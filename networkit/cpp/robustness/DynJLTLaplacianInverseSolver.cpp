/*
 *  DynJLTLaplacianInverseSolver.cpp
 *
 *  Created on: 27.06.2023
 *     Authors: Maria Predari <predarimaria@gmail.com>
 *              Lukas Berner <Lukas.Berner@hu-berlin.de>
 */

#include <networkit/robustness/DynJLTLaplacianInverseSolver.hpp>

#include <networkit/algebraic/CSRMatrix.hpp>
#include <networkit/numerics/LAMG/Lamg.hpp>

namespace NetworKit {

int jltDimension(int perRound, double epsilon) {
    return std::max(std::log(perRound) / (epsilon * epsilon), 1.);
}

template <class NK_Matrix>
inline NK_Matrix nk_matrix_from_expr(count n_rows, count n_cols,
                                     std::function<double(count, count)> expr) {
    std::vector<Triplet> triplets;
    triplets.reserve(n_rows * n_cols);
    for (count i = 0; i < n_rows; i++) {
        for (count j = 0; j < n_cols; j++) {
            double v = expr(i, j);
            if (v != 0.) {
                triplets.push_back({i, j, v});
            }
        }
    }
    return NK_Matrix{n_rows, n_cols, triplets};
}

inline CSRMatrix nk_dense_to_csr(DenseMatrix dense) {
    auto entry = [&](count i, count j) -> double { return dense(i, j); };
    return nk_matrix_from_expr<CSRMatrix>(dense.numberOfRows(), dense.numberOfColumns(), entry);
}

DynJLTLaplacianInverseSolver::DynJLTLaplacianInverseSolver(const Graph &G, double tolerance,
                                                           count eqnPerRound, count roundsPerSolver,
                                                           count roundsPerColumn)
    : DynLaplacianInverseSolver(G), l(jltDimension(eqnPerRound, tolerance)), tolerance(tolerance),
      solver(G, 0.0001, 2 * l + 2, roundsPerSolver, roundsPerColumn) {
    if (!G.hasEdgeIds())
        throw std::runtime_error(
            "Error: call Graph.indexEdges() before initializing DynJLTLaplacianInverseSolver!");
}

void DynJLTLaplacianInverseSolver::run() {
    assert(G.hasEdgeIds());

    solver.run();

    incidence = CSRMatrix::incidenceMatrix(G);
    computeIntermediateMatrices();

    hasRun = true;
}

void DynJLTLaplacianInverseSolver::update(GraphEvent ev) {
    assureFinished();
    assureUpdated(ev);

    solver.update(ev);
    if (ev.type != GraphEvent::EDGE_ADDITION && ev.type != GraphEvent::EDGE_REMOVAL)
        throw std::runtime_error(
            "Error: GraphEvents other than Edge addition and removal are not supported!");

    // TODO: check if updating this instead of re-computing is faster?
    incidence = CSRMatrix::incidenceMatrix(G);
    computeIntermediateMatrices();
}

double DynJLTLaplacianInverseSolver::totalResistanceDifference(const GraphEvent &ev) const {
    assureFinished();

    const auto R = effR(ev.u, ev.v);
    const auto phiNormSq = phiNormSquared(ev.u, ev.v);

    double w;
    if (ev.type == GraphEvent::EDGE_ADDITION)
        w = 1.0 / (1.0 + R);
    else if (ev.type == GraphEvent::EDGE_REMOVAL)
        w = 1.0 / (1.0 - R);
    else
        throw std::logic_error(
            "totalResistanceDifference cannot be computed for events other than edge "
            "addition or deletion!");
    return G.numberOfNodes() * w * phiNormSq;
}

double DynJLTLaplacianInverseSolver::effR(node u, node v) const {
    const auto r = PBL.row(u) - PBL.row(v);
    return Vector::innerProduct(r, r);
}

double DynJLTLaplacianInverseSolver::phiNormSquared(node u, node v) const {
    const auto r = PL.row(u) - PL.row(v);
    return Vector::innerProduct(r, r);
}

void DynJLTLaplacianInverseSolver::computeIntermediateMatrices() {
    // Generate projection matrices

    auto random_projection = [&](count n_rows, count n_cols) -> DenseMatrix {
        auto normal = [&]() { return d(Aux::Random::getURNG()) / std::sqrt(n_rows); };
        auto normal_expr = [&](count, count) { return normal(); };
        DenseMatrix P = nk_matrix_from_expr<DenseMatrix>(n_rows, n_cols, normal_expr);
        Vector avg = P * Vector(n_cols, 1. / n_cols);
        P -= Vector::outerProduct<DenseMatrix>(avg, Vector(n_cols, 1.));
        return P;
    };

    auto P_n = random_projection(l, G.numberOfNodes());
    auto P_m = random_projection(l, G.numberOfEdges());

    assert(G.numberOfEdges() == incidence.numberOfColumns());

    // Compute columns of P L^\dagger and P' B^T L^\dagger where B is the
    // incidence matrix of G We first compute the transposes of the targets. For
    // the first, solve LX = P^T, for the second solve LX = B P^T

    // CSRMatrix rhs1 = P_n.transpose();
    CSRMatrix rhs2_mat = incidence * nk_dense_to_csr(P_m.transpose());
    std::vector<Vector> rhss1;
    std::vector<Vector> rhss2;

    for (count i = 0; i < l; i++) {
        rhss1.push_back(P_n.row(i).transpose());
        rhss2.push_back(rhs2_mat.column(i));
    }

    auto xs1 = solver.parallelSolve(rhss1);
    auto xs2 = solver.parallelSolve(rhss2);

    PL = DenseMatrix(l, G.numberOfNodes());
    PBL = DenseMatrix(l, G.numberOfNodes());

    for (count i = 0; i < l; i++) {
        for (count j = 0; j < G.numberOfNodes(); j++) {
            PL.setValue(i, j, xs1[i][j]);
            PBL.setValue(i, j, xs2[i][j]);
        }
    }
    PL = PL.transpose();
    PBL = PBL.transpose();
}

} // namespace NetworKit
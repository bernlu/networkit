/*
 *  DynLazyLaplacianInverseSolver.cpp
 *
 *  Created on: 27.06.2023
 *     Authors: Maria Predari <predarimaria@gmail.com>
 *              Lukas Berner <Lukas.Berner@hu-berlin.de>
 */

#include <networkit/robustness/DynLazyLaplacianInverseSolver.hpp>

namespace NetworKit {

DynLazyLaplacianInverseSolver::DynLazyLaplacianInverseSolver(const Graph &G, double tolerance,
                                                             count eqnPerRound,
                                                             count roundsPerSolver,
                                                             count roundsPerColumn)
    : DynLaplacianInverseSolver(G), n(G.numberOfNodes()), tolerance(tolerance),
      eqnPerRound(eqnPerRound), lamg(tolerance), roundsPerSolver(roundsPerSolver),
      roundsPerColumn(roundsPerColumn) {}

void DynLazyLaplacianInverseSolver::run() {
    laplacian = CSRMatrix::laplacianMatrix(G);

    updateW.clear();
    updateVec.clear();
    cols.clear();
    colAge.clear();

    colAge.resize(n, std::numeric_limits<count>::max());
    cols.resize(n);
    solverAge = 0;
    round = 0;

    lamg.~Lamg<CSRMatrix>();
    new (&lamg) Lamg<CSRMatrix>(tolerance);
    lamg.setup(laplacian);

    hasRun = true;
}

const Vector &DynLazyLaplacianInverseSolver::getColumn(node u) const {
    computeColumns({u});
    return cols[u];
}

void DynLazyLaplacianInverseSolver::computeColumns(std::vector<node> nodes) const {
    // Determine which nodes need the solver
    std::vector<node> nodes_to_solve;
    for (auto u : nodes) {
        if (round - colAge[u] >= roundsPerColumn
            || colAge[u] == std::numeric_limits<count>::max()) {
            nodes_to_solve.push_back(u);
        }
    }

    // Solve
    auto nodes_to_solve_count = nodes_to_solve.size();
    std::vector<NetworKit::Vector> rhss(nodes_to_solve_count,
                                        NetworKit::Vector(n, -1. / static_cast<double>(n)));
    for (count i = 0; i < nodes_to_solve_count; i++) {
        auto u = nodes_to_solve[i];
        rhss[i][u] += 1.;
    }
    std::vector<NetworKit::Vector> xs(nodes_to_solve_count, NetworKit::Vector(n));
    lamg.parallelSolve(rhss, xs);

    // Ensure slns average 0
    for (count i = 0; i < nodes_to_solve_count; i++) {
        auto &x = xs[i];
        auto u = nodes_to_solve[i];
        double avg =
            NetworKit::Vector::innerProduct(x, NetworKit::Vector(n, 1.0 / static_cast<double>(n)));
        x -= avg;

        cols[u] = x;
        colAge[u] = solverAge;
    }

    // Update
    for (count i = 0; i < nodes.size(); i++) {
        auto u = nodes[i];
        auto &col = cols[u];
        for (auto &r = colAge[u]; r < round; r++) {
            auto &upv = updateVec[r];
            col -= upv * (upv[u] * updateW[r]);
        }
        colAge[u] = round;
    }
}

// assume that the event has already happend (G has been modified before the call to this function)
void DynLazyLaplacianInverseSolver::update(GraphEvent ev) {
    assureFinished();
    assureUpdated(ev);

    const node u = ev.u;
    const node v = ev.v;

    computeColumns({u, v});
    const auto colU = getColumn(u);
    const auto colV = getColumn(v);

    double R = colU[u] + colV[v] - 2 * colU[v];
    double w;
    if (ev.type == GraphEvent::EDGE_ADDITION)
        w = 1.0 / (1.0 + R);
    else if (ev.type == GraphEvent::EDGE_REMOVAL)
        w = -1.0 / (1.0 - R);
    else
        throw std::logic_error("update cannot be computed for events other than edge "
                               "addition or deletion!");
    assert(w < 1.);
    auto upv = colU - colV;

    if (ev.type == GraphEvent::EDGE_ADDITION) {
        laplacian.setValue(u, u, laplacian(u, u) + 1.);
        laplacian.setValue(v, v, laplacian(v, v) + 1.);
        laplacian.setValue(u, v, laplacian(u, v) - 1.);
        laplacian.setValue(v, u, laplacian(v, u) - 1.);
    } else if (ev.type == GraphEvent::EDGE_REMOVAL) {
        laplacian.setValue(u, u, laplacian(u, u) - 1.);
        laplacian.setValue(v, v, laplacian(v, v) - 1.);
        laplacian.setValue(u, v, laplacian(u, v) + 1.);
        laplacian.setValue(v, u, laplacian(v, u) + 1.);
    } else
        throw std::logic_error("update cannot be computed for events other than edge "
                               "addition or deletion!");

    updateVec.push_back(upv);
    updateW.push_back(w);
    round++;

    if (round % roundsPerSolver == 0) {
        lamg.~Lamg<CSRMatrix>();
        new (&lamg) Lamg<CSRMatrix>(tolerance);
        lamg.setup(laplacian);

        solverAge = round;
    }
}

double DynLazyLaplacianInverseSolver::totalResistanceDifference(const GraphEvent &ev) const {
    assureFinished();

    const node i = ev.u;
    const node j = ev.v;

    const auto col_i = getColumn(i);
    const auto col_j = getColumn(j);
    const double R_ij = col_i[i] + col_j[j] - 2 * col_i[j];
    double w;
    if (ev.type == GraphEvent::EDGE_ADDITION)
        w = 1.0 / (1.0 + R_ij);
    else if (ev.type == GraphEvent::EDGE_REMOVAL)
        w = 1.0 / (1.0 - R_ij);
    else
        throw std::logic_error(
            "totalResistanceDifference cannot be computed for events other than edge "
            "addition or deletion!");
    const auto norm = (col_i - col_j).length();
    return norm * norm * w * col_i.getDimension();
}

std::vector<Vector> DynLazyLaplacianInverseSolver::parallelSolve(std::vector<Vector> &rhss) const {
    auto size = rhss.size();
    std::vector<Vector> xs(size, Vector(n));
    // lamg.parallelSolve(rhss, xs);
    for (count i = 0; i < size; ++i) {
        lamg.solve(rhss[i], xs[i]);
    }

#pragma omp parallel for
    for (count i = 0; i < size; i++) {
        auto &x = xs[i];
        auto &rhs = rhss[i];
        double avg = x.transpose() * Vector(n, 1.0 / static_cast<double>(n));
        x -= avg;

        for (count r = solverAge; r < round; r++) {
            auto upv = updateVec[r];
            x -= upv * (NetworKit::Vector::innerProduct(upv, rhs) * updateW[r]);
        }
    }

    return xs;
}

} // namespace NetworKit
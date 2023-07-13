/*
 *  RobustnessGreedy.cpp
 *
 *  Created on: 27.06.2023
 *     Authors: Maria Predari <predarimaria@gmail.com>
 *              Lukas Berner <Lukas.Berner@hu-berlin.de>
 */

#include <networkit/robustness/RobustnessGreedy.hpp>

#include <networkit/algebraic/CSRMatrix.hpp>
#include <networkit/numerics/LAMG/Lamg.hpp>

namespace NetworKit {

RobustnessGreedy::RobustnessGreedy(const Graph &G, count k, Problem robustnessProblem,
                                   Metric metric, node focusNode)
    : G(G), k(k), robustnessProblem(robustnessProblem),
      metric(metric == Metric::none ? defaultMetric(robustnessProblem) : metric),
      focusNode(focusNode) {}

RobustnessGreedy::Metric RobustnessGreedy::defaultMetric(Problem problem) {
    if (problem == Problem::GLOBAL_REDUCTION)
        return Metric::FOREST;
    else
        return Metric::RESISTANCE;
}

void RobustnessGreedy::resetFocus(node focusNode) {
    this->focusNode = focusNode;
    this->hasRun = false;
}

DenseMatrix FullLpinv::setupLpinv(RobustnessGreedy::Metric metric, count n) {
    // return DenseMatrix(n);
    switch (metric) {
    case RobustnessGreedy::Metric::FOREST:
        return DenseMatrix(n + 1);
    case RobustnessGreedy::Metric::RESISTANCE:
        return DenseMatrix(n);
    case RobustnessGreedy::Metric::none:
        throw std::logic_error("metric cannot be none");
    }
    throw std::logic_error("unreachable code in setupLpinv");
}

FullLpinv::FullLpinv(const Graph &G, RobustnessGreedy::Metric metric)
    : lpinv(setupLpinv(metric, G.numberOfNodes())) {}

void FullLpinv::setupLaplacianPseudoinverse(const Graph &G, RobustnessGreedy::Metric metric,
                                            RobustnessGreedy::Problem robustnessProblem) {
    if (lpinvOriginal) { // if original has been set, re-use
        lpinv = lpinvOriginal.value();
        return;
    }
    DEBUG("running setup laplacian");
    auto lap = CSRMatrix::laplacianMatrix(G);
    if (metric == RobustnessGreedy::Metric::FOREST) {
        // modify lap to add node for forest distance
        const count n = G.numberOfNodes();
        CSRMatrix lap2(n + 1, -1.0);
        lap2.setValue(n, n, n);
        std::vector<index> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        lap2.assign(indices, indices, lap);
        // lap2.sort();
        for (count i = 0; i < lap2.numberOfColumns(); i++)
            lap2.setValue(i, i, lap2(i, i) + 1);
        lap = lap2;
        // lap.sort();
        // lap = std::move(lap2); // optimize this copy?
    }
    DEBUG("laplacian:\n", lap);
    Lamg<CSRMatrix> lamg;
    lamg.setupConnected(lap);

    const count n = lap.numberOfColumns();
    const count maxThreads = static_cast<count>(omp_get_max_threads());

    // Solution vectors: one per thread
    std::vector<Vector> solutions(maxThreads, Vector(n));

    // Right hand side vectors: one per thread
    std::vector<Vector> rhss(maxThreads, Vector(n));

    const count iters = (n % maxThreads == 0) ? n / maxThreads : n / maxThreads + 1;
    for (count i = 0; i < iters; ++i) {
        // Index of the next vertex to process
        const index base = i * maxThreads;

#pragma omp parallel
        {
            // Each thread solves a linear system from `base` to `base + #threads - 1`
            const index thread = omp_get_thread_num();
            const node v = base + thread;
            if (v < n) {
                // Reset solution and rhs vector of the current thread
                solutions[thread].fill(0.0);

                // Set up system to compute the diagonal column
                rhss[thread].fill(-1. / static_cast<double>(n));
                // rhss[thread].fill(0);
                rhss[thread][v] += 1.;
            }
        }

        if (base + maxThreads >= n) {
            // Last iteration: some threads cannot be used.
            // Resize rhss and solutions to the number of vertices left to be processed.
            rhss.resize(n - base);
            solutions.resize(rhss.size());
        }

        lamg.parallelSolve(rhss, solutions);

        // Store the results
        for (index idx = 0; idx < solutions.size(); ++idx) {
            const node v = base + idx;
            if (v < n)
                for (index row = 0; row < n; ++row)
                    // lpinv.setValue(row, v, solutions[idx][row] - 1.0 / n);
                    lpinv.setValue(row, v, solutions[idx][row]);
            else
                break;
        }
    }

    if (robustnessProblem == RobustnessGreedy::Problem::LOCAL_IMPROVEMENT) // store lpinv for re-use
        lpinvOriginal = lpinv;
}

/// computes absolute difference of Lpinv(G) for the edge addition or deletion event @a ev
/// i.e. abs (|Lpinv[:,i] - Lpinv[:,j]|)^2 / (1 +- r(i,j))
double FullLpinv::laplacianPseudoinverseTraceDifference(const GraphEvent &ev) const {
    const node i = ev.u;
    const node j = ev.v;
    const auto col_i = lpinv.column(i);
    const auto col_j = lpinv.column(j);
    double R_ij = col_i[i] + col_j[j] - 2 * col_i[j];
    double w;
    if (ev.type == GraphEvent::EDGE_ADDITION)
        w = 1.0 / (1.0 + R_ij);
    else if (ev.type == GraphEvent::EDGE_REMOVAL)
        w = 1.0 / (1.0 - R_ij);
    else
        throw std::logic_error(
            "Trace difference cannot be computed for events other than edge addition or deletion!");
    auto norm = (col_i - col_j).length();
    return norm * norm * w;
}

void FullLpinv::updateLaplacianPseudoinverse(const GraphEvent &ev) {
    auto i = ev.u;
    auto j = ev.v;
    double R_ij = lpinv(i, i) + lpinv(j, j) - 2 * lpinv(i, j);

    double w_negative;
    if (ev.type == GraphEvent::EDGE_ADDITION)
        w_negative = -1.0 / (1.0 + R_ij);
    else if (ev.type == GraphEvent::EDGE_REMOVAL)
        w_negative = -1.0 / (1.0 - R_ij);
    else
        throw std::logic_error("updateLaplacianPseudoinverse does not support events other than "
                               "edge addition or deletion!");
    auto v = lpinv.column(i) - lpinv.column(j);

    // #pragma omp parallel for
    const auto n = lpinv.numberOfRows();
    for (index i = 0; i < n; i++) {
        auto updateVec = v[i] * w_negative * v;
        for (index j = 0; j < n; j++) {
            lpinv.setValue(j, i, lpinv(i, j) + updateVec[j]);
        }
    }
}

// ------------------------------------------------------------------------------------

void DynLapSolver::setupSolver() {
    laplacian = CSRMatrix::laplacianMatrix(G_);
    n = G_.numberOfNodes();
    m = G_.numberOfEdges();

    updateW.clear();
    updateVec.clear();
    cols.clear();
    colAge.clear();

    // TODO: deal with case n+1
    if (useJLT) {
        incidence = CSRMatrix::incidenceMatrix(G_);

        // std::cout << "incidence rows: " << incidence.numberOfRows() << std::endl;
        // std::cout << "incidence cols: " << incidence.numberOfColumns() << std::endl;
        // for (count i = 0; i < incidence.numberOfRows(); i++) {
        //     for (count j = 0; j < incidence.numberOfColumns(); j++)
        //         std::cout << incidence(i, j) << ",";
        //     std::cout << std::endl;
        // }
    }

    // if (metric == RobustnessGreedy::Metric::FOREST) {
    //     // modify lap to add node for forest distance
    //     const count n = G.numberOfNodes();
    //     CSRMatrix lap2(n + 1);
    //     lap2.setValue(n, n, n);
    //     std::vector<index> indices(n);
    //     std::iota(indices.begin(), indices.end(), 0);
    //     for (auto i : indices) {
    //         lap2.setValue(n, i, -1);
    //         lap2.setValue(i, n, -1);
    //     }
    //     lap2.assign(indices, indices, laplacian);
    //     lap2.sort();

    //     laplacian = lap2;
    //     // lap = std::move(lap2); // optimize this copy?
    // }
    const count n = G_.numberOfNodes();

    colAge.resize(n, std::numeric_limits<count>::max());
    cols.resize(n);
    solverAge = 0;
    round = 0;

    lamg.~Lamg<CSRMatrix>();
    new (&lamg) Lamg<CSRMatrix>(tolerance);
    lamg.setup(laplacian);

    if (useJLT)
        computeIntermediateMatrices();
}

const Vector &DynLapSolver::getColumn(node u) {
    computeColumns({u});
    return cols[u];
}

void DynLapSolver::computeColumns(std::vector<node> nodes) {
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
void DynLapSolver::updateEdge(const GraphEvent &ev) {
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
        throw std::logic_error("Trace difference cannot be computed for events other than edge "
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
        throw std::logic_error("Trace difference cannot be computed for events other than edge "
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

    if (useJLT) {
        // TODO: integrate into this code
        // G.addEdge(u, v);
        // solver.addEdge(u, v); // done above

        G_.indexEdges();
        incidence = CSRMatrix::incidenceMatrix(G_);
        m = G_.numberOfEdges();

        computeIntermediateMatrices();
    }
}

double DynLapSolver::totalResistanceDifferenceApprox(const GraphEvent &ev) {

    const node i = ev.u;
    const node j = ev.v;

    if (useJLT) {
        double R = effR(i, j);
        double phiNormSq = phiNormSquared(i, j);
        if (ev.type == GraphEvent::EDGE_ADDITION)
            return n / (1. + R) * phiNormSq;
        if (ev.type == GraphEvent::EDGE_REMOVAL)
            return n / (1. - R) * phiNormSq;
    }

    const auto col_i = getColumn(i);
    const auto col_j = getColumn(j);
    const double R_ij = col_i[i] + col_j[j] - 2 * col_i[j];
    double w;
    if (ev.type == GraphEvent::EDGE_ADDITION)
        w = 1.0 / (1.0 + R_ij);
    else if (ev.type == GraphEvent::EDGE_REMOVAL)
        w = 1.0 / (1.0 - R_ij);
    else
        throw std::logic_error("Trace difference cannot be computed for events other than edge "
                               "addition or deletion!");
    const auto norm = (col_i - col_j).length();
    return norm * norm * w * col_i.getDimension();
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

void DynLapSolver::computeIntermediateMatrices() {

    // Generate projection matrices

    std::normal_distribution<> d{0, 1};

    auto random_projection = [&](count n_rows, count n_cols) -> NetworKit::DenseMatrix {
        auto normal = [&]() { return d(Aux::Random::getURNG()) / std::sqrt(n_rows); };
        auto normal_expr = [&](NetworKit::count, NetworKit::count) { return normal(); };
        DenseMatrix P = nk_matrix_from_expr<DenseMatrix>(n_rows, n_cols, normal_expr);
        NetworKit::Vector avg = P * NetworKit::Vector(n_cols, 1. / n_cols);
        P -= NetworKit::Vector::outerProduct<DenseMatrix>(avg, NetworKit::Vector(n_cols, 1.));
        return P;
    };

    auto P_n = random_projection(l, n);
    auto P_m = random_projection(l, m);

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

    auto parallelSolve = [&](std::vector<Vector> &rhss) {
        auto size = rhss.size();
        std::vector<Vector> xs(size, Vector(n));
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
                x -= upv * (Vector::innerProduct(upv, rhs) * updateW[r]);
            }
        }
        return xs;
    };

    auto xs1 = parallelSolve(rhss1);
    auto xs2 = parallelSolve(rhss2);

    PL = DenseMatrix(l, n);
    PBL = DenseMatrix(l, n);

    for (count i = 0; i < l; i++) {
        for (count j = 0; j < n; j++) {
            PL.setValue(i, j, xs1[i][j]);
            PBL.setValue(i, j, xs2[i][j]);
        }
    }
}

double DynLapSolver::effR(node u, node v) {
    auto r = PBL.column(u) - PBL.column(v);
    return Vector::innerProduct(r, r);
}

double DynLapSolver::phiNormSquared(node u, node v) {
    auto r = PL.column(u) - PL.column(v);
    return Vector::innerProduct(r, r);
}

void printAllEdgeGains(Graph &G) {
    auto pairs = {// std::make_pair(RobustnessGreedy::Metric::RESISTANCE,
                  //                  RobustnessGreedy::Problem::GLOBAL_IMPROVEMENT),
                  //   std::make_pair(RobustnessGreedy::Metric::RESISTANCE,
                  //                  RobustnessGreedy::Problem::GLOBAL_REDUCTION),
                  //   std::make_pair(RobustnessGreedy::Metric::FOREST,
                  //                  RobustnessGreedy::Problem::GLOBAL_IMPROVEMENT),
                  std::make_pair(RobustnessGreedy::Metric::FOREST,
                                 RobustnessGreedy::Problem::GLOBAL_REDUCTION)};

    for (auto [metric, problem] : pairs) {
        std::cout << "Metric: "
                  << (metric == RobustnessGreedy::Metric::RESISTANCE ? "resistance" : "forest")
                  << std::endl;
        std::cout << "Problem: "
                  << (problem == RobustnessGreedy::Problem::GLOBAL_IMPROVEMENT ? "improvement"
                                                                               : "reduction")
                  << std::endl;
        FullLpinv lp(G, metric);
        lp.setupLaplacianPseudoinverse(G, metric, problem);
        std::cout << "lpinv: " << std::endl;
        std::cout << lp.lpinv;

        if (problem == RobustnessGreedy::Problem::GLOBAL_IMPROVEMENT) {
            G.forNodePairs([&](node u, node v) {
                if (!G.hasEdge(u, v))
                    std::cout << "(" << u << " ," << v << "): "
                              << lp.laplacianPseudoinverseTraceGain(
                                     GraphEvent(GraphEvent::EDGE_ADDITION, u, v))
                              << std::endl;
            });
        }

        if (problem == RobustnessGreedy::Problem::GLOBAL_REDUCTION) {
            G.forNodePairs([&](node u, node v) {
                if (G.hasEdge(u, v))
                    std::cout << "(" << u << " ," << v << "): "
                              << lp.laplacianPseudoinverseTraceGain(
                                     GraphEvent(GraphEvent::EDGE_REMOVAL, u, v))
                              << std::endl;
            });
        }
    }
}

std::ostream &operator<<(std::ostream &os, const DenseMatrix &M) {
    for (index i = 0; i < M.numberOfRows(); i++) {
        for (index j = 0; j < M.numberOfColumns(); j++)
            os << M(i, j) << ", ";
        os << std::endl;
    }
    return os;
}

std::ostream &operator<<(std::ostream &os, const CSRMatrix &M) {
    for (index i = 0; i < M.numberOfRows(); i++) {
        for (index j = 0; j < M.numberOfColumns(); j++)
            os << M(i, j) << ", ";
        os << std::endl;
    }
    return os;
}

} // namespace NetworKit
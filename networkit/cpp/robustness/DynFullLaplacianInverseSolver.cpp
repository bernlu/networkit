/*
 *  DynFullLaplacianInverseSolver.cpp
 *
 *  Created on: 27.06.2023
 *     Authors: Maria Predari <predarimaria@gmail.com>
 *              Lukas Berner <Lukas.Berner@hu-berlin.de>
 */

#include <networkit/robustness/DynFullLaplacianInverseSolver.hpp>

#include <networkit/algebraic/CSRMatrix.hpp>
#include <networkit/numerics/LAMG/Lamg.hpp>

namespace NetworKit {

DynFullLaplacianInverseSolver::DynFullLaplacianInverseSolver(const Graph &G)
    : DynLaplacianInverseSolver(G), lpinv(G.numberOfNodes()) {}

void DynFullLaplacianInverseSolver::run() {
    auto lap = CSRMatrix::laplacianMatrix(G);
    DEBUG(lap);
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

        // TODO: change this to have the row access in the outer loop ?
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
    hasRun = true;
}

double DynFullLaplacianInverseSolver::totalResistanceDifference(const GraphEvent &ev) const {
    assureFinished();

    const node i = ev.u;
    const node j = ev.v;
    const auto col_i = lpinv.row(i); // lpinv is symmetric, Take row instead of column, because
                                     // DenseMatrix is stored row-major
    const auto col_j = lpinv.row(j);
    const double R_ij = col_i[i] + col_j[j] - 2 * col_i[j];
    double w;
    if (ev.type == GraphEvent::EDGE_ADDITION)
        w = 1.0 / (1.0 + R_ij);
    else if (ev.type == GraphEvent::EDGE_REMOVAL)
        w = 1.0 / (1.0 - R_ij);
    else
        throw std::logic_error(
            "Trace difference cannot be computed for events other than edge addition or deletion!");
    const auto norm = (col_i - col_j).length();
    return norm * norm * w * G.numberOfNodes();
}

void DynFullLaplacianInverseSolver::update(GraphEvent ev) {
    assureFinished();
    assureUpdated(ev);

    if (ev.type == GraphEvent::EDGE_ADDITION)
        assert(G.hasEdge(ev.u, ev.v));
    if (ev.type == GraphEvent::EDGE_REMOVAL)
        assert(!G.hasEdge(ev.u, ev.v));

    const auto i = ev.u;
    const auto j = ev.v;
    const double R_ij = lpinv(i, i) + lpinv(j, j) - 2 * lpinv(i, j);

    double w_negative;
    if (ev.type == GraphEvent::EDGE_ADDITION)
        w_negative = -1.0 / (1.0 + R_ij);
    else if (ev.type == GraphEvent::EDGE_REMOVAL)
        w_negative = -1.0 / (1.0 - R_ij);
    else
        throw std::logic_error("update does not support events other than "
                               "edge addition or deletion!");
    const auto v = lpinv.row(i) - lpinv.row(j);

    const auto n = lpinv.numberOfRows();
#pragma omp parallel for
    for (index i = 0; i < n; i++) {
        const auto updateVec = v[i] * w_negative * v;
        for (index j = 0; j < n; j++) {
            lpinv.setValue(j, i, lpinv(j, i) + updateVec[j]);
        }
    }
}

} // namespace NetworKit
/*
 *  DynLaplacianInverseSolver.cpp
 *
 *  Created on: 27.06.2023
 *     Authors: Maria Predari <predarimaria@gmail.com>
 *              Lukas Berner <Lukas.Berner@hu-berlin.de>
 */

#include <networkit/robustness/DynLaplacianInverseSolver.hpp>

#include <networkit/algebraic/DynamicMatrix.hpp>
#include <networkit/numerics/LAMG/Lamg.hpp>
#include <networkit/robustness/DynFullLaplacianInverseSolver.hpp>
#include <networkit/robustness/RobustnessGreedy.hpp>

namespace NetworKit {

void DynLaplacianInverseSolver::assureUpdated(const GraphEvent &ev) const {
    if (ev.type == GraphEvent::EDGE_ADDITION)
        assert(G.hasEdge(ev.u, ev.v));
    if (ev.type == GraphEvent::EDGE_REMOVAL)
        assert(!G.hasEdge(ev.u, ev.v));
}

void printAllEdgeGains(Graph &G) {
    using Metric = RobustnessGreedy::Metric;
    using Problem = RobustnessGreedy::Problem;
    auto pairs = {

        std::make_pair(Metric::RESISTANCE, Problem::GLOBAL_IMPROVEMENT),
        std::make_pair(Metric::RESISTANCE, Problem::GLOBAL_REDUCTION),
        std::make_pair(Metric::FOREST, Problem::GLOBAL_IMPROVEMENT),
        std::make_pair(Metric::FOREST, Problem::GLOBAL_REDUCTION)

    };

    for (auto [metric, problem] : pairs) {
        std::cout << "Metric: " << (metric == Metric::RESISTANCE ? "resistance" : "forest")
                  << std::endl;
        std::cout << "Problem: "
                  << (problem == Problem::GLOBAL_IMPROVEMENT ? "improvement" : "reduction")
                  << std::endl;

        node forestNode = none;
        if (metric == Metric::FOREST) {
            forestNode = G.addNode();
            G.forNodes([&](node u) {
                if (u != forestNode)
                    G.addEdge(u, forestNode);
            });
        }

        DynFullLaplacianInverseSolver lp(G);
        lp.run();
        std::cout << "lpinv: " << std::endl;
        std::cout << lp.lpinv;

        if (problem == RobustnessGreedy::Problem::GLOBAL_IMPROVEMENT) {
            G.forNodePairs([&](node u, node v) {
                if (!G.hasEdge(u, v) && u != forestNode && v != forestNode)
                    std::cout << "(" << u << " ," << v << "): "
                              << lp.totalResistanceDifference(
                                     GraphEvent(GraphEvent::EDGE_ADDITION, u, v))
                              << std::endl;
            });
        }

        if (problem == RobustnessGreedy::Problem::GLOBAL_REDUCTION) {
            G.forNodePairs([&](node u, node v) {
                if (G.hasEdge(u, v) && u != forestNode && v != forestNode)
                    std::cout << "(" << u << " ," << v << "): "
                              << lp.totalResistanceDifference(
                                     GraphEvent(GraphEvent::EDGE_REMOVAL, u, v))
                              << std::endl;
            });
        }
        if (metric == Metric::FOREST)
            G.removeNode(forestNode);
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

std::ostream &operator<<(std::ostream &os, const DynamicMatrix &M) {
    for (index i = 0; i < M.numberOfRows(); i++) {
        for (index j = 0; j < M.numberOfColumns(); j++)
            os << M(i, j) << ", ";
        os << std::endl;
    }
    return os;
}

} // namespace NetworKit
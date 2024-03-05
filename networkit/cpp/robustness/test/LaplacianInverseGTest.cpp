/*
 * LaplacianInverseGTest.cpp
 *
 *  Created on: 27.06.2023
 *      Author: Lukas Berner (Lukas.Berner@hu-berlin.de)
 */

#include <functional>
#include <gtest/gtest.h>

#include <networkit/graph/Graph.hpp>

#include <networkit/robustness/DynFullLaplacianInverseSolver.hpp>
#include <networkit/robustness/DynJLTLaplacianInverseSolver.hpp>
#include <networkit/robustness/DynLaplacianInverseSolver.hpp>
#include <networkit/robustness/DynLazyLaplacianInverseSolver.hpp>
// #include <networkit/robustness/DynSpectralLaplacianInverseSolver.hpp>

namespace NetworKit {

std::unique_ptr<DynLaplacianInverseSolver> full(const Graph &G) {
    return std::make_unique<DynFullLaplacianInverseSolver>(G);
}
std::unique_ptr<DynLaplacianInverseSolver> lazy(const Graph &G) {
    return std::make_unique<DynLazyLaplacianInverseSolver>(G, 0.99);
}
std::unique_ptr<DynLaplacianInverseSolver> jlt(const Graph &G) {
    return std::make_unique<DynJLTLaplacianInverseSolver>(G, 0.99);
}

using generatorFn = decltype(full);

class LaplacianInverseGTest : public testing::Test {
public:
    Graph smallGraph() {
        Graph G;
        G.addNodes(6);
        std::vector<std::pair<unsigned long, unsigned long>> edges = {
            {0, 1}, {0, 2}, {1, 3}, {2, 3}, {1, 2}, {1, 4}, {3, 5}, {4, 5}};
        /* G=

            0
           / \
          1 - 2
          |\ /
          4 3
          |/
          5

        */
        for (auto p : edges) {
            G.addEdge(p.first, p.second);
        }
        G.indexEdges();
        return G;
    }

    std::vector<std::unique_ptr<DynLaplacianInverseSolver>> setupSolvers(const Graph &G) {
        std::vector<std::unique_ptr<DynLaplacianInverseSolver>> solvers;
        solvers.push_back(std::make_unique<DynFullLaplacianInverseSolver>(G));
        // solvers.push_back(std::make_unique<DynLazyLaplacianInverseSolver>(G, 1e-6));
        // solvers.push_back(std::make_unique<DynJLTLaplacianInverseSolver>(G, 0.55));
        // solvers.push_back(std::make_unique<DynSpectralLaplacianInverseSolver>(G, 3));
        return solvers;
    }
};

TEST_F(LaplacianInverseGTest, testRun) {
    Graph G = smallGraph();

    auto solvers = setupSolvers(G);

    for (auto &solver : solvers)
        solver->run();

    G.forNodePairs([&](node u, node v) {
        GraphEvent ev;
        if (G.hasEdge(u, v)) {
            ev = GraphEvent(GraphEvent::EDGE_REMOVAL, u, v);
            // std::cout << "removal " << u << ", " << v << std::endl;
        } else {

            ev = GraphEvent(GraphEvent::EDGE_ADDITION, u, v);
            // std::cout << "addition " << u << ", " << v << std::endl;
        }
        auto &solver0 = solvers[0];
        for (auto &solver : solvers)
            EXPECT_NEAR(solver->totalResistanceDifference(ev),
                        solver0->totalResistanceDifference(ev), 0.1);
    });
}

TEST_F(LaplacianInverseGTest, testUpdates) {
    Graph G = smallGraph();

    auto solvers = setupSolvers(G);
    for (auto &solver : solvers)
        solver->run();

    std::vector<Edge> updates = {{0, 5}, {1, 5}, {3, 5}, {2, 0}, {1, 0}, {4, 2}};

    for (auto update : updates) {
        const auto u = update.u;
        const auto v = update.v;
        GraphEvent upd;
        if (G.hasEdge(u, v)) {
            G.removeEdge(u, v);
            upd = GraphEvent(GraphEvent::EDGE_REMOVAL, u, v);
        } else {
            G.addEdge(u, v);
            upd = GraphEvent(GraphEvent::EDGE_ADDITION, u, v);
        }

        for (auto &solver : solvers)
            solver->update(upd);

        G.forNodePairs([&](node u, node v) {
            GraphEvent ev;
            if (G.hasEdge(u, v)) {
                ev = GraphEvent(GraphEvent::EDGE_REMOVAL, u, v);
                // std::cout << "removal " << u << ", " << v << std::endl;
            } else {

                ev = GraphEvent(GraphEvent::EDGE_ADDITION, u, v);
                // std::cout << "addition " << u << ", " << v << std::endl;
            }
            auto &solver0 = solvers[0];
            for (auto &solver : solvers)
                EXPECT_NEAR(solver->totalResistanceDifference(ev),
                            solver0->totalResistanceDifference(ev), 0.1);
        });
    }
}

} /* namespace NetworKit */

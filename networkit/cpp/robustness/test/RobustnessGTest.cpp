/*
 * RobustnessGTest.cpp
 *
 *  Created on: 27.06.2023
 *      Author: Lukas Berner (Lukas.Berner@hu-berlin.de)
 */

#include <functional>
#include <gtest/gtest.h>

#include <networkit/graph/Graph.hpp>
#include <networkit/robustness/ColStoch.hpp>
#include <networkit/robustness/SimplStoch.hpp>
#include <networkit/robustness/StGreedy.hpp>

#include <networkit/robustness/RobustnessGreedy.hpp>

#include <networkit/io/NetworkitBinaryReader.hpp>

namespace NetworKit {

class RobustnessGTest : public testing::Test {
protected:
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
        return G;
    }
};

// TEST_F(RobustnessGTest, CSRMAtrixBug) {
//     CSRMatrix A(2);
//     A.setValue(0, 0, 1);
//     A.setValue(0, 1, -1);
//     A.setValue(1, 0, -1);
//     A.setValue(1, 1, 1);
//     // A =
//     // 1 -1
//     // -1 1
//     std::cout << "CSRMatrix A: " << std::endl;
//     std::cout << A << std::endl;

//     CSRMatrix B(3, -1.0);
//     B.setValue(2, 2, 2);
//     // -1 -1 -1
//     // -1 -1 -1
//     // -1 -1 2

//     std::vector<index> indices(2);
//     std::iota(indices.begin(), indices.end(), 0);
//     std::cout << "indices: ";
//     for (auto ind : indices)
//         std::cout << ind << ", "; // 0, 1
//     std::cout << std::endl;

//     B.assign(indices, indices, A);
//     // should result in:
//     // 1 -1 -1
//     // -1 1 -1
//     // -1 -1 2
//     std::cout << "B: " << std::endl;
//     std::cout << B << std::endl;

//     for (count i = 0; i < B.numberOfColumns() - 1; i++)
//         B.setValue(i, i, B(i, i) + 1);
//     // should result in:
//     // 2 -1 -1
//     // -1 2 -1
//     // -1 -1 2
//     std::cout << "B: " << std::endl;
//     std::cout << B << std::endl;
// }

// TEST_F(RobustnessGTest, laplacianMatrixBug) {
//     Graph G(4);
//     G.addEdge(0, 1);
//     G.addEdge(0, 2);
//     G.addEdge(0, 3);
//     G.addEdge(1, 2);

//     std::cout << CSRMatrix::laplacianMatrix(G) << std::endl;
//     node newnode = G.addNode();
//     std::cout << CSRMatrix::laplacianMatrix(G) << std::endl;
//     G.removeNode(newnode);
//     G.compactEdges();
//     G.checkConsistency();
//     std::cout << CSRMatrix::laplacianMatrix(G) << std::endl;
// }

TEST_F(RobustnessGTest, printAllGains) {
    Graph G = smallGraph();
    printAllEdgeGains(G);
    G.removeEdge(1, 0);
    printAllEdgeGains(G);
}

TEST_F(RobustnessGTest, testStGreedy_GRIP_smallgraph) {
    Graph G = smallGraph();

    StGreedy greedy(G, 2, StGreedy::Problem::GLOBAL_IMPROVEMENT);
    greedy.run();
    auto result = greedy.getResultItems();
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].u, 0);
    EXPECT_EQ(result[0].v, 5);
    EXPECT_EQ(result[1].u, 2);
    EXPECT_EQ(result[1].v, 4);
}

// TEST_F(RobustnessGTest, testInfpower_stGreedy) {
//     NetworkitBinaryReader reader;
//     auto g = reader.read(
//         "/home/berneluk/Dokumente/Robustness/robustness-extend/instances/inf-power.nkb");
//     StGreedy greedy(g, 5, StGreedy::Problem::GLOBAL_IMPROVEMENT);
//     INFO("l 131");
//     greedy.run();
//     auto result = greedy.getResultItems();
//     for (auto r : result) {
//         std::cout << r.u << ", " << r.v << std::endl;
//     }
//     std::cout << "total: " << greedy.getResultValue() << std::endl;
// }

TEST_F(RobustnessGTest, testStGreedy_LRIP_smallgraph) {
    Graph G = smallGraph();

    StGreedy greedy(G, 2, StGreedy::Problem::LOCAL_IMPROVEMENT, StGreedy::Metric::RESISTANCE, 0);
    greedy.run();
    auto result = greedy.getResultItems();
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].u, 0);
    EXPECT_EQ(result[0].v, 5);
    EXPECT_EQ(result[1].u, 0);
    EXPECT_EQ(result[1].v, 4);

    greedy.resetFocus(2);
    greedy.run();
    result = greedy.getResultItems();
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].u, 2);
    EXPECT_EQ(result[0].v, 5);
    EXPECT_EQ(result[1].u, 2);
    EXPECT_EQ(result[1].v, 4);
}

TEST_F(RobustnessGTest, testStGreedy_GDEL_smallgraph) {
    Graph G = smallGraph();

    StGreedy greedy(G, 2, StGreedy::Problem::GLOBAL_REDUCTION);
    greedy.run();
    auto result = greedy.getResultItems();
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].u, 5);
    EXPECT_EQ(result[0].v, 4);
    EXPECT_EQ(result[1].u, 5);
    EXPECT_EQ(result[1].v, 3);
    EXPECT_NEAR(greedy.getResultValue(), 5., 0.1);
}

TEST_F(RobustnessGTest, testSimplStoch_GRIP_smallgraph) {
    Aux::Random::setSeed(1, true);
    Graph G = smallGraph();

    SimplStoch greedy(G, 2, SimplStoch::Problem::GLOBAL_IMPROVEMENT, 0.99);
    greedy.run();
    auto result = greedy.getResultItems();
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].u, 4);
    EXPECT_EQ(result[0].v, 0);
    EXPECT_EQ(result[1].u, 3);
    EXPECT_EQ(result[1].v, 0);
}

TEST_F(RobustnessGTest, testSimplStoch_LRIP_smallgraph) {
    Aux::Random::setSeed(1, true);
    Graph G = smallGraph();

    SimplStoch greedy(G, 2, SimplStoch::Problem::LOCAL_IMPROVEMENT, 0.9, false, {},
                      SimplStoch::Metric::RESISTANCE, 0);
    greedy.run();
    auto result = greedy.getResultItems();
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].u, 0);
    EXPECT_EQ(result[0].v, 4);
    EXPECT_EQ(result[1].u, 0);
    EXPECT_EQ(result[1].v, 5);

    greedy.resetFocus(2);
    greedy.run();
    result = greedy.getResultItems();
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].u, 2);
    EXPECT_EQ(result[0].v, 4);
    EXPECT_EQ(result[1].u, 2);
    EXPECT_EQ(result[1].v, 5);
}

TEST_F(RobustnessGTest, testSimplStoch_GDEL_smallgraph) {
    Aux::Random::setSeed(1, true);
    Graph G = smallGraph();

    SimplStoch greedy(G, 2, SimplStoch::Problem::GLOBAL_REDUCTION, 0.99);
    greedy.run();
    auto result = greedy.getResultItems();
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].u, 2);
    EXPECT_EQ(result[0].v, 1);
    EXPECT_EQ(result[1].u, 2);
    EXPECT_EQ(result[1].v, 0);
    EXPECT_NEAR(greedy.getResultValue(), 6.3, 0.1);
}

TEST_F(RobustnessGTest, testSimplStochJLT_GRIP_smallgraph) {
    Aux::Random::setSeed(1, true);
    Graph G = smallGraph();

    SimplStoch greedy(G, 2, SimplStoch::Problem::GLOBAL_IMPROVEMENT, 0.99, true);
    greedy.run();
    auto result = greedy.getResultItems();
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].u, 1);
    EXPECT_EQ(result[0].v, 5);
    EXPECT_EQ(result[1].u, 0);
    EXPECT_EQ(result[1].v, 3);
}

TEST_F(RobustnessGTest, testSimplStochJLT_LRIP_smallgraph) {
    Aux::Random::setSeed(1, true);
    Graph G = smallGraph();

    SimplStoch greedy(G, 2, SimplStoch::Problem::LOCAL_IMPROVEMENT, 0.9, true);
    greedy.resetFocus(0);
    greedy.run();
    auto result = greedy.getResultItems();
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].u, 0);
    EXPECT_EQ(result[0].v, 4);
    EXPECT_EQ(result[1].u, 0);
    EXPECT_EQ(result[1].v, 5);

    greedy.resetFocus(2);
    greedy.run();
    result = greedy.getResultItems();
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].u, 2);
    EXPECT_EQ(result[0].v, 5);
    EXPECT_EQ(result[1].u, 2);
    EXPECT_EQ(result[1].v, 4);
}

TEST_F(RobustnessGTest, testSimplStochJLT_GDEL_smallgraph) {
    Aux::Random::setSeed(1, true);
    Graph G = smallGraph();

    SimplStoch greedy(G, 2, SimplStoch::Problem::GLOBAL_REDUCTION, 0.99, true);
    greedy.run();
    auto result = greedy.getResultItems();
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].u, 1);
    EXPECT_EQ(result[0].v, 0);
    EXPECT_EQ(result[1].u, 3);
    EXPECT_EQ(result[1].v, 2);
    EXPECT_NEAR(greedy.getResultValue(), 12.2, 0.1);
}

TEST_F(RobustnessGTest, testColStoch_GRIP_smallgraph) {
    Aux::Random::setSeed(1, true);
    Graph G = smallGraph();

    ColStoch greedy(G, 2, ColStoch::Problem::GLOBAL_IMPROVEMENT, 0.99);
    greedy.run();
    auto result = greedy.getResultItems();
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].u, 1);
    EXPECT_EQ(result[0].v, 5);
    EXPECT_EQ(result[1].u, 0);
    EXPECT_EQ(result[1].v, 3);
}

TEST_F(RobustnessGTest, testColStoch_LRIP_smallgraph) {
    Aux::Random::setSeed(1, true);
    Graph G = smallGraph();

    ColStoch greedy(G, 2, ColStoch::Problem::LOCAL_IMPROVEMENT, 0.9);
    greedy.resetFocus(0);
    greedy.run();
    auto result = greedy.getResultItems();
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].u, 0);
    EXPECT_EQ(result[0].v, 4);
    EXPECT_EQ(result[1].u, 0);
    EXPECT_EQ(result[1].v, 5);

    greedy.resetFocus(2);
    greedy.run();
    result = greedy.getResultItems();
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].u, 2);
    EXPECT_EQ(result[0].v, 5);
    EXPECT_EQ(result[1].u, 2);
    EXPECT_EQ(result[1].v, 4);
}

TEST_F(RobustnessGTest, testColStoch_GDEL_smallgraph) {
    Aux::Random::setSeed(1, true);
    Graph G = smallGraph();

    ColStoch greedy(G, 2, ColStoch::Problem::GLOBAL_REDUCTION, 0.99);
    greedy.run();
    auto result = greedy.getResultItems();
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].u, 1);
    EXPECT_EQ(result[0].v, 0);
    EXPECT_EQ(result[1].u, 3);
    EXPECT_EQ(result[1].v, 2);
    EXPECT_NEAR(greedy.getResultValue(), 12.2, 0.1);
}

TEST_F(RobustnessGTest, testColStochJLT_GRIP_smallgraph) {
    Aux::Random::setSeed(1, true);
    Graph G = smallGraph();

    ColStoch greedy(G, 2, ColStoch::Problem::GLOBAL_IMPROVEMENT, 0.99, true);
    greedy.run();
    auto result = greedy.getResultItems();
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].u, 1);
    EXPECT_EQ(result[0].v, 5);
    EXPECT_EQ(result[1].u, 0);
    EXPECT_EQ(result[1].v, 3);
}

TEST_F(RobustnessGTest, testColStochJLT_LRIP_smallgraph) {
    Aux::Random::setSeed(1, true);
    Graph G = smallGraph();

    ColStoch greedy(G, 2, ColStoch::Problem::LOCAL_IMPROVEMENT, 0.9, true);
    greedy.resetFocus(0);
    greedy.run();
    auto result = greedy.getResultItems();
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].u, 0);
    EXPECT_EQ(result[0].v, 4);
    EXPECT_EQ(result[1].u, 0);
    EXPECT_EQ(result[1].v, 5);

    greedy.resetFocus(2);
    greedy.run();
    result = greedy.getResultItems();
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].u, 2);
    EXPECT_EQ(result[0].v, 5);
    EXPECT_EQ(result[1].u, 2);
    EXPECT_EQ(result[1].v, 4);
}

TEST_F(RobustnessGTest, testColStochJLT_GDEL_smallgraph) {
    Aux::Random::setSeed(1, true);
    Graph G = smallGraph();

    ColStoch greedy(G, 2, ColStoch::Problem::GLOBAL_REDUCTION, 0.99, true);
    greedy.run();
    auto result = greedy.getResultItems();
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].u, 1);
    EXPECT_EQ(result[0].v, 0);
    EXPECT_EQ(result[1].u, 3);
    EXPECT_EQ(result[1].v, 2);
    EXPECT_NEAR(greedy.getResultValue(), 12.2, 0.1);
}

} /* namespace NetworKit */

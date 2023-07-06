/*
 * RobustnessGTest.cpp
 *
 *  Created on: 27.06.2023
 *      Author: Lukas Berner (Lukas.Berner@hu-berlin.de)
 */

#include <functional>
#include <gtest/gtest.h>

#include <networkit/graph/Graph.hpp>
#include <networkit/robustness/StochasticGreedy.hpp>
#include <networkit/robustness/SubmodularGreedy.hpp>

namespace NetworKit {

class GreedyGTest : public testing::Test {};

TEST_F(GreedyGTest, testSubmodularGreedy) {
    std::vector<int> candidates{1, 2, 3, 4, 5};

    auto sg = SubmodularGreedy<int>(candidates, 2);
    sg.setGainFunction([](const int &i) { return i; });
    sg.setPickedItemCallback([](int) {});
    sg.run();
    auto result = sg.getResultItems();
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0], 5);
    EXPECT_EQ(result[1], 4);
    EXPECT_EQ(sg.getResultValue(), 9);
}

TEST_F(GreedyGTest, testSubmodularGreedy_all) {
    std::vector<int> candidates{4, 5};

    auto sg = SubmodularGreedy<int>(
        candidates, 2, [](int i) { return i; }, [](int) {});
    sg.run();
    auto result = sg.getResultItems();
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0], 5);
    EXPECT_EQ(result[1], 4);
    EXPECT_EQ(sg.getResultValue(), 9);
}

TEST_F(GreedyGTest, testSubmodularGreedy_k_greater_n) {
    std::vector<int> candidates{4, 5};

    auto sg = SubmodularGreedy<int>(
        candidates, 3, [](int i) { return i; }, [](int) {});
    sg.run();
    auto result = sg.getResultItems();
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0], 5);
    EXPECT_EQ(result[1], 4);
    EXPECT_EQ(sg.getResultValue(), 9);
}

TEST_F(GreedyGTest, testSubmodularGreedy_changingGain) {
    std::vector<int> candidates{1, 2, 3};

    // gain in the first iteration:  1, 2, 3 (pick 3)
    // gain in the second iteration: 4, 3, 2 (pick 1)
    // submodular greedy will update the value of 2, then 2 will be top of the PQ and get picked
    // (gain of 1 is not evaluated) expect total value of 3 + 3 = 6

    bool once_added = false;

    auto sg = SubmodularGreedy<int>(
        candidates, 2,
        [&](int i) {
            if (!once_added)
                return i;
            else
                return 5 - i;
        },
        [&](int) { once_added = true; });
    sg.run();
    auto result = sg.getResultItems();
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0], 3);
    EXPECT_EQ(result[1], 2);
    EXPECT_EQ(sg.getResultValue(), 6);
}

TEST_F(GreedyGTest, testSubmodularGreedy_nonprintable) {
    std::vector<std::pair<int, int>> items{{1, 2}, {3, 4}, {5, 6}};
    auto sg2 = SubmodularGreedy<std::pair<int, int>>(items, 2);
    sg2.setGainFunction([&](auto &x) { return x.first; });
    sg2.setPickedItemCallback([&](auto) {});
    sg2.run();
}

TEST_F(GreedyGTest, testStochasticGreedy) {
    Aux::Random::setSeed(1, true);
    std::vector<int> candidates{1, 2, 3, 4, 5};

    auto sg = StochasticGreedy<int>(candidates, 2, 0.5);
    sg.setGainFunction([](const int &i) {
        DEBUG(i);
        return i;
    });
    sg.setPickedItemCallback([](int) {});
    sg.run();
    auto result = sg.getResultItems();
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0], 4);
    EXPECT_EQ(result[1], 5);
    EXPECT_EQ(sg.getResultValue(), 9);
}

} /* namespace NetworKit */

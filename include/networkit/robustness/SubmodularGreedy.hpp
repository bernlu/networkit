/*
 *  SubmodularGreedy.hpp
 *
 *  Created on: 27.06.2023
 *     Authors: Maria Predari <predarimaria@gmail.com>
 *              Lukas Berner <Lukas.Berner@hu-berlin.de>
 */

#ifndef NETWORKIT_ROBUSTNESS_SUBMODULAR_GREEDY_HPP_
#define NETWORKIT_ROBUSTNESS_SUBMODULAR_GREEDY_HPP_

#include <iostream>
#include <queue>
#include <vector>

#include <omp.h>

#include <networkit/auxiliary/Log.hpp>
#include <networkit/graph/Graph.hpp>
#include <networkit/robustness/GreedyOptimizer.hpp>

namespace NetworKit {

// TODO: write a operator< for the case when Item is not comparable!

template <class Item>
struct _ItemWrapperType {
    Item item;
    double value;
    count lastUpdated;
};

template <class Item>
bool operator<(const _ItemWrapperType<Item> &left, const _ItemWrapperType<Item> &right) {
    return left.value < right.value; // || (right.value == left.value && left.item < right.item); //
                                     // the second check is not required for the PQ to work (?)
}

template <class Item>
class SubmodularGreedy : public GreedyOptimizer<Item> {
public:
    using typename GreedyOptimizer<Item>::gainFnType;
    using typename GreedyOptimizer<Item>::pickedItemCallbackType;

    SubmodularGreedy(const std::vector<Item> &items, count k, gainFnType gainFn,
                     pickedItemCallbackType pickedItemCallback)
        : GreedyOptimizer<Item>(items, k, std::move(gainFn), std::move(pickedItemCallback)){};
    SubmodularGreedy(const std::vector<Item> &items, count k) : GreedyOptimizer<Item>(items, k){};
    SubmodularGreedy(count k) : GreedyOptimizer<Item>(k){};
    virtual void run() override;

    void summarize();

protected:
    using ItemWrapper = _ItemWrapperType<Item>;

    void initializeRun();
    void initializeRound(count){};

    std::priority_queue<ItemWrapper> itemQueue;
};

template <class Item>
void SubmodularGreedy<Item>::summarize() {
    std::cout << "Greedy Results Summary. ";
    if (!this->hasRun) {
        std::cout << "Not executed yet!";
    }
    std::cout << "Result Size: " << this->result.size() << std::endl;
    if (this->result.size() < 1000) {
        for (auto e : this->getResultItems()) {
            std::cout << "(" << e.u << ", " << e.v << "), ";
        }
    }
    std::cout << std::endl;
    std::cout << "Total Value: " << this->getTotalValue() << std::endl;
}

template <class Item>
void SubmodularGreedy<Item>::initializeRun() {
    itemQueue = std::priority_queue<ItemWrapper>();
    unsigned int threads = omp_get_max_threads();
    std::vector<std::vector<ItemWrapper>> items_per_thread{threads, std::vector<ItemWrapper>{0}};

#pragma omp parallel for
    for (unsigned int ind = 0; ind < this->items.size(); ind++) {
        auto i = this->items[ind];
        ItemWrapper it{i, this->gainFn(i), 0};
        items_per_thread[omp_get_thread_num()].push_back(it);
    }
    for (unsigned int i = 0; i < threads; i++) {
        for (auto &it : items_per_thread[i]) {
            itemQueue.push(it);
        }
    }
}

template <class Item>
void SubmodularGreedy<Item>::run() {
    this->assureCallbacksSet();
    count round = 0;
    this->result.clear();
    initializeRun();

    bool candidatesLeft = true;

    while (!itemQueue.empty()) {
        initializeRound(round);

        // Get top updated entry from queue
        ItemWrapper c;
        while (true) {
            if (itemQueue.empty()) {
                candidatesLeft = false;
                break;
            } else {
                c = itemQueue.top();
                itemQueue.pop();
            }

            DEBUG(" INSPECTING candidate value = ", c.value, " lastupdated = ", c.lastUpdated);
            // debug_print_sg(c.item);

            if (c.lastUpdated == round)
                break; // top updated entry found.
            else {
                c.value = this->gainFn(c.item);
                c.lastUpdated = round;
                itemQueue.push(c);
            }
        }
        if (candidatesLeft) {
            this->result.push_back(c.item);
            this->totalGain += c.value;

            DEBUG(" >>> TotalGain = ", this->totalGain, " <<< ");
            // if constexpr (is_debug_printable_v<Item>)
            //     DEBUG(" SELECTED value = ", c.value, " of item = ", c.item);
            // else
            DEBUG(" SELECTED value = ", c.value, " picked item:");
            // debug_print_sg<Item>(c.item);

            this->pickedItemCallback(c.item);
            round++;
            if (round == this->k)
                break;
        }
    }
    this->hasRun = true;
}

} // namespace NetworKit

#endif // NETWORKIT_ROBUSTNESS_SUBMODULAR_GREEDY_HPP_

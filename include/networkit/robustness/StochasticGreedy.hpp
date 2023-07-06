/*
 *  StochasticGreedy.hpp
 *
 *  Created on: 27.06.2023
 *     Authors: Maria Predari <predarimaria@gmail.com>
 *              Lukas Berner <Lukas.Berner@hu-berlin.de>
 */

#ifndef NETWORKIT_ROBUSTNESS_STOCHASTIC_GREEDY_HPP_
#define NETWORKIT_ROBUSTNESS_STOCHASTIC_GREEDY_HPP_

#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <set>
#include <vector>

#include <omp.h>
#include <networkit/auxiliary/Log.hpp>
#include <networkit/auxiliary/Random.hpp>
#include <networkit/robustness/GreedyOptimizer.hpp>

// Stochastic Greedy
// Implementation of
// Baharan Mirzasoleiman, Ashwinkumar Badanidiyuru, Amin Karbasi, Jan Vondrak,
// Andreas Krause:  Lazier Than Lazy Greedy. https://arxiv.org/abs/1409.7938

namespace NetworKit {

template <class Item>
struct _ItemWrapper {
    Item item;
    double value;
    std::optional<count> lastUpdated;
    count index;
    bool selected;
};

template <class Item>
bool operator<(const _ItemWrapper<Item> &left, const _ItemWrapper<Item> &right) {
    return left.value < right.value; // || (right.value == left.value && left.item < right.item);
                                     // this second part is not required for the pq to work. (?)
}

template <class Item>
class StochasticGreedy : public GreedyOptimizer<Item> {
public:
    using typename GreedyOptimizer<Item>::gainFnType;
    using typename GreedyOptimizer<Item>::pickedItemCallbackType;

    /**
     * @param epsilon [0-1] <desc TODO>
     */
    StochasticGreedy(std::vector<Item> &items, count k, double epsilon, gainFnType gainFn,
                     pickedItemCallbackType pickedItemCallback)
        : GreedyOptimizer<Item>(items, k, std::move(gainFn), std::move(pickedItemCallback)),
          epsilon(epsilon){};
    StochasticGreedy(std::vector<Item> &items, count k, double epsilon)
        : GreedyOptimizer<Item>(items, k), epsilon(epsilon){};
    StochasticGreedy(count k, double epsilon) : GreedyOptimizer<Item>(k), epsilon(epsilon){};

    virtual void run() override;

    void summarize() {
        std::cout << "Stochastic Submodular Greedy Results Summary. ";
        if (!this->hasRun) {
            std::cout << "Not executed yet!";
            return;
        }
        std::cout << "Result Size: " << this->result.size() << std::endl;
        if (this->result.size() < 1000) {
            for (auto e : this->getResultItems()) {
                std::cout << "(" << e.u << ", " << e.v << "), ";
            }
        }
        std::cout << std::endl;
        std::cout << "Total Value: " << this->getResultValue() << std::endl;
    }

protected:
    using ItemWrapper = _ItemWrapper<Item>;

    void initializeRun();
    count N;
    double epsilon = 0.1;

    std::vector<ItemWrapper> itemsWrapped;
};

template <class Item>
void StochasticGreedy<Item>::initializeRun() {
    itemsWrapped = std::vector<ItemWrapper>();
    for (count i = 0; i < this->items.size(); i++) {
        auto it = this->items[i];
        ItemWrapper qe{it, std::numeric_limits<double>::infinity(), -1, i, false};
        itemsWrapped.push_back(qe);
    }
    this->N = itemsWrapped.size();
}

template <class Item>
void StochasticGreedy<Item>::run() {
    this->assureCallbacksSet();
    count round = 0;
    this->result.clear();

    initializeRun();

    bool candidatesLeft = true;

    while (candidatesLeft) {
        std::priority_queue<ItemWrapper> R;
        count s = (count)(1.0 * this->N / this->k * std::log(1.0 / epsilon)) + 1;
        s = std::min(s, this->itemsWrapped.size() - round);

        // Get a random subset of the items of size s.
        // Do this via selecting individual elements resp via shuffling,
        // depending on wether s is large or small.
        // ==========================================================================
        // Populating R set. What is the difference between small and large s?
        if (s > N / 4) { // This is not a theoretically justified estimate
            std::vector<unsigned int> allIndices = std::vector<unsigned int>(N);
            std::iota(allIndices.begin(), allIndices.end(), 0);
            std::shuffle(allIndices.begin(), allIndices.end(), Aux::Random::getURNG());

            auto itemCount = itemsWrapped.size();
            for (count i = 0; i < itemCount; i++) {
                auto &item = this->itemsWrapped[allIndices[i]];
                if (!item.selected) {
                    // DEBUG("ITEM:(", item.item.u, " ,",  item.item.v, ") value = ",
                    // item.value, " lastUpdated = ", item.lastUpdated, " index = ",
                    // item.index, " selected = ", item.selected);
                    R.push(item);
                    if (R.size() >= s) {
                        break;
                    }
                }
            }
        } else {
            while (R.size() < s) {
                std::set<unsigned int> indicesSet;
                count v = Aux::Random::index(N);
                if (indicesSet.count(v) == 0) {
                    indicesSet.insert(v);
                    auto item = this->itemsWrapped[v];
                    if (!item.selected) {
                        // DEBUG("ITEM:(", item.item.u, " ,",  item.item.v, ") value = ",
                        // item.value, " lastUpdated = ", item.lastUpdated, " index = ",
                        // item.index, " selected = ", item.selected);
                        R.push(item);
                    }
                }
            }
        }
        // ==========================================================================

        DEBUG("AFTER POPULATING PRIORITY QUEUE. R size: ", R.size());

        // Get top updated entry from R
        ItemWrapper c;
        while (true) {
            if (R.empty()) {
                candidatesLeft = false;
                break;
            } else {
                c = R.top();
                R.pop();
            }

            if (c.lastUpdated == round) {
                break; // top updated entry found.
            } else {
                auto &item = this->itemsWrapped[c.index];
                c.value = this->gainFn(c.item);
                // DEBUG(" TOP :(", c.item.u, " ,",  c.item.v, ") value = ", c.value,
                // " lastUpdated = ", c.lastUpdated, " index = ", c.index, " selected
                // = ", c.selected);
                item.value = c.value;
                c.lastUpdated = round;
                item.lastUpdated = round;
                R.push(c);
            }

            // DEBUG("PRINTING ITEMS (START).");
            // printItems();
            // DEBUG("PRINTING ITEMS (END).");
        }
        if (candidatesLeft) {
            this->result.push_back(c.item);
            this->totalGain += c.value;
            // DEBUG(" >>> TOTALVALUE = ", this->totalValue, " <<< ");
            // DEBUG(" SELECTED value = ", c.value, " of edge = (", c.item.u, ", ",
            // c.item.v, ")");
            this->pickedItemCallback(c.item);
            this->itemsWrapped[c.index].selected = true;

            round++;
            if (round == this->k)
                break;
        }
    }
    this->hasRun = true;
}

} // namespace NetworKit

#endif // NETWORKIT_ROBUSTNESS_STOCHASTIC_GREEDY_HPP_
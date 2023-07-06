/*
 *  GreedyOptimizer.hpp
 *
 *  Created on: 27.06.2023
 *     Authors: Maria Predari <predarimaria@gmail.com>
 *              Lukas Berner <Lukas.Berner@hu-berlin.de>
 */

#ifndef NETWORKIT_ROBUSTNESS_GREEDY_OPTIMIZER_HPP_
#define NETWORKIT_ROBUSTNESS_GREEDY_OPTIMIZER_HPP_

#include <functional>

#include <networkit/Globals.hpp>
#include <networkit/base/Algorithm.hpp>

namespace NetworKit {

/**
 * @ingroup robustness
 * Abstract base class for greedy optimization algorithms.
 * The greedy algorithm attempts to maximize the total gain over all items picked, under the
 * constraint that the number of picked items is at most @a k.
 */
template <class Item>
class GreedyOptimizer : public Algorithm {
public:
    using gainFnType = std::function<double(const Item &)>;
    using pickedItemCallbackType = std::function<void(const Item &)>;
    /**
     * @param items all candidate items
     * @param k the maximum number of items to pick
     * @param gainFn lambda function that returns the gain for a given Item. This function may be
     * called in parallel! The gain may change after an Item is picked.
     * @param pickedItemCallback callback function that will be run every time an Item is picked to
     * allow updating of relevant data structures and updating the gain function
     */
    GreedyOptimizer(const std::vector<Item> &items, count k, gainFnType gainFn,
                    pickedItemCallbackType pickedItemCallback);

    GreedyOptimizer(const std::vector<Item> &items, count k);
    GreedyOptimizer(count k);

    // void setItems(const std::vector<Item> &items);
    void setGainFunction(gainFnType gainFn);
    void setPickedItemCallback(pickedItemCallbackType pickedItemCallback);

    virtual double getResultValue() {
        assureFinished();
        return totalGain;
    };

    virtual std::vector<Item> &getResultItems() {
        assureFinished();
        return result;
    };

protected:
    gainFnType gainFn;                         // function that computes the gain for an item
    pickedItemCallbackType pickedItemCallback; // callback that is run for each picked item
    const std::vector<Item> &items;            // all candidate items
    const count k;                             // maximum number of items to add
    std::vector<Item> result;                  // vector of picked items
    double totalGain = 0;                      // sum of all gains in result

    void assureCallbacksSet() {
        if (!gainFn)
            throw std::runtime_error("Error, gainFn must be set first");
        if (!pickedItemCallback)
            throw std::runtime_error("Error, pickedItemCallback must be set first");
    };
};

template <class Item>
GreedyOptimizer<Item>::GreedyOptimizer(const std::vector<Item> &items, count k, gainFnType gainFn,
                                       pickedItemCallbackType pickedItemCallback)
    : gainFn(std::move(gainFn)), pickedItemCallback(std::move(pickedItemCallback)), items(items),
      k(k){};

template <class Item>
GreedyOptimizer<Item>::GreedyOptimizer(const std::vector<Item> &items, count k)
    : items(items), k(k){};

template <class Item>
GreedyOptimizer<Item>::GreedyOptimizer(count k) : k(k){};

template <class Item>
void GreedyOptimizer<Item>::setGainFunction(gainFnType gainFn) {
    this->gainFn = std::move(gainFn);
};

template <class Item>
void GreedyOptimizer<Item>::setPickedItemCallback(pickedItemCallbackType pickedItemCallback) {
    this->pickedItemCallback = std::move(pickedItemCallback);
};

} // namespace NetworKit

#endif // NETWORKIT_ROBUSTNESS_GREEDY_OPTIMIZER_HPP_
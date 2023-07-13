/*
 *  stGreedy.hpp
 *
 *  Created on: 27.06.2023
 *     Authors: Maria Predari <predarimaria@gmail.com>
 *              Lukas Berner <Lukas.Berner@hu-berlin.de>
 */

#ifndef NETWORKIT_ROBUSTNESS_ST_GREEDY_HPP_
#define NETWORKIT_ROBUSTNESS_ST_GREEDY_HPP_

#include <networkit/algebraic/CSRMatrix.hpp>
#include <networkit/algebraic/DenseMatrix.hpp>
#include <networkit/dynamics/GraphEvent.hpp>
#include <networkit/numerics/LAMG/Lamg.hpp>
#include <networkit/robustness/RobustnessGreedy.hpp>
#include <networkit/robustness/SubmodularGreedy.hpp>

#include <optional>

namespace NetworKit {

class StGreedy final : public RobustnessGreedy, FullLpinv {
public:
    StGreedy(const Graph &G, count k, Problem robustnessProblem, Metric metric = Metric::none,
             node focusNode = none);

    virtual void run() override;

private:
    std::vector<Edge> buildCandidateSet();
};

} // namespace NetworKit

#endif // NETWORKIT_ROBUSTNESS_ST_GREEDY_HPP_
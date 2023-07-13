/*
 *  SimplStoch.hpp
 *
 *  Created on: 27.06.2023
 *     Authors: Maria Predari <predarimaria@gmail.com>
 *              Lukas Berner <Lukas.Berner@hu-berlin.de>
 */
#ifndef NETWORKIT_ROBUSTNESS_SIMPL_STOCH_HPP_
#define NETWORKIT_ROBUSTNESS_SIMPL_STOCH_HPP_

#include <networkit/algebraic/DenseMatrix.hpp>
#include <networkit/dynamics/GraphEvent.hpp>
#include <networkit/robustness/RobustnessGreedy.hpp>

#include <optional>

namespace NetworKit {

class SimplStoch final : public RobustnessGreedy, FullLpinv {
public:
    enum class CandidateSetSize { SMALL, LARGE };
    SimplStoch(const Graph &G, count k, Problem robustnessProblem, double epsilon,
               Metric metric = Metric::none, node focusNode = none,
               CandidateSetSize candidatesize = CandidateSetSize::SMALL);

    virtual void run() override;

private:
    std::vector<Edge> buildCandidateSet();
    const CandidateSetSize candidatesize;
    const double epsilon;
};

} // namespace NetworKit

#endif // NETWORKIT_ROBUSTNESS_SIMPL_STOCH_HPP_
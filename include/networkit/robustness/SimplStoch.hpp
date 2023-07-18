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

class SimplStoch final : public RobustnessGreedy {
public:
    SimplStoch(Graph &G, count k, Problem robustnessProblem, double epsilon, bool useJLT = false,
               std::optional<double> solverEpsilon = {}, Metric metric = Metric::none,
               node focusNode = none);

    virtual void run() override;

private:
    const bool useJLT;
    const double epsilon;
    const double solverEpsilon;
};

} // namespace NetworKit

#endif // NETWORKIT_ROBUSTNESS_SIMPL_STOCH_HPP_
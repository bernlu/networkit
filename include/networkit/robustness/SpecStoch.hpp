/*
 *  SpecStoch.hpp
 *
 *  Created on: 27.06.2023
 *     Authors: Maria Predari <predarimaria@gmail.com>
 *              Lukas Berner <Lukas.Berner@hu-berlin.de>
 */
#ifndef NETWORKIT_ROBUSTNESS_SPEC_STOCH_HPP_
#define NETWORKIT_ROBUSTNESS_SPEC_STOCH_HPP_

#include <networkit/algebraic/DenseMatrix.hpp>
#include <networkit/dynamics/GraphEvent.hpp>
#include <networkit/robustness/RobustnessGreedy.hpp>

#include <optional>

namespace NetworKit {

class SpecStoch final : public RobustnessGreedy {
public:
    SpecStoch(Graph &G, count k, Problem robustnessProblem, double epsilon,
              count numberOfEigenpairs, Metric metric = Metric::AUTOMATIC, node focusNode = none);

    virtual void run() override;

private:
    const double epsilon;
    const count numberOfEigenpairs;
};

} // namespace NetworKit

#endif // NETWORKIT_ROBUSTNESS_SPEC_STOCH_HPP_

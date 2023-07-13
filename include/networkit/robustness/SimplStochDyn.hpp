/*
 *  SimplStochDyn.hpp
 *
 *  Created on: 27.06.2023
 *     Authors: Maria Predari <predarimaria@gmail.com>
 *              Lukas Berner <Lukas.Berner@hu-berlin.de>
 */
#ifndef NETWORKIT_ROBUSTNESS_SIMPL_STOCH_DYN_HPP_
#define NETWORKIT_ROBUSTNESS_SIMPL_STOCH_DYN_HPP_

#include <networkit/dynamics/GraphEvent.hpp>
#include <networkit/robustness/RobustnessGreedy.hpp>

#include <optional>

namespace NetworKit {

class SimplStochDyn final : public RobustnessGreedy, DynLapSolver {
public:
    SimplStochDyn(Graph &G, count k, Problem robustnessProblem, double epsilon, bool useJLT = false,
                  double solverEpsilon = 1e-6, Metric metric = Metric::none, node focusNode = none);

    virtual void run() override;

private:
    Graph &G;
    std::vector<Edge> buildCandidateSet(node forestCenter = none);
    const double epsilon;
};

} // namespace NetworKit

#endif // NETWORKIT_ROBUSTNESS_SIMPL_STOCH_DYN_HPP_
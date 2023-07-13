/*
 *  ColStoch.hpp
 *
 *  Created on: 27.06.2023
 *     Authors: Maria Predari <predarimaria@gmail.com>
 *              Lukas Berner <Lukas.Berner@hu-berlin.de>
 */
#ifndef NETWORKIT_ROBUSTNESS_COL_STOCH_HPP_
#define NETWORKIT_ROBUSTNESS_COL_STOCH_HPP_

#include <networkit/centrality/DynApproxElectricalCloseness.hpp>
#include <networkit/dynamics/GraphEvent.hpp>
#include <networkit/robustness/RobustnessGreedy.hpp>

#include <optional>

namespace NetworKit {

class ColStoch final : public RobustnessGreedy, DynLapSolver {
public:
    ColStoch(Graph &G, count k, Problem robustnessProblem, double epsilon, bool useJLT = false,
             double solverEpsilon = 1e-6, double diagEpsilon = 10, Metric metric = Metric::none,
             node focusNode = none);

    virtual void run() override;

private:
    Graph &G;
    const double epsilon;
    DynApproxElectricalCloseness apx;

    std::vector<Edge> buildCandidateSet();
    count numberOfNodeCandidates() const;
    std::optional<GraphEvent> makeEvent(node u, node v = none) const;
};

} // namespace NetworKit

#endif // NETWORKIT_ROBUSTNESS_COL_STOCH_HPP_
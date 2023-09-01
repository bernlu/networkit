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

class ColStoch final : public RobustnessGreedy {
public:
    ColStoch(Graph &G, count k, Problem robustnessProblem, double epsilon, double diagEpsilon = 10,
             bool useJLT = false, bool jltLossCorrection = true,
             std::optional<double> solverEpsilon = {}, Metric metric = Metric::AUTOMATIC,
             node focusNode = none);

    virtual void run() override;

private:
    const double epsilon;
    const double solverEpsilon;
    const double diagEpsilon;
    const bool useJLT;
    const bool jltLossCorrection;
    std::unique_ptr<DynApproxElectricalCloseness> apx;
    std::unique_ptr<DynApproxElectricalCloseness> apxCopy;

    count numberOfNodeCandidates() const;
    std::optional<GraphEvent> makeEvent(node u, node v = none) const;
};

} // namespace NetworKit

#endif // NETWORKIT_ROBUSTNESS_COL_STOCH_HPP_

/*
 *  stGreedy.cpp
 *
 *  Created on: 27.06.2023
 *     Authors: Maria Predari <predarimaria@gmail.com>
 *              Lukas Berner <Lukas.Berner@hu-berlin.de>
 */

#include <networkit/robustness/StGreedy.hpp>

#include <networkit/robustness/DynFullLaplacianInverseSolver.hpp>

namespace NetworKit {

StGreedy::StGreedy(Graph &G, count k, Problem robustnessProblem, Metric metric, node focusNode)
    : RobustnessGreedy(G, k, robustnessProblem, metric, focusNode) {}

void StGreedy::run() {
    // Compute pseudoinverse of laplacian
    // setupLaplacianPseudoinverse(G, metric, robustnessProblem);

    prepareGraph();
    result.clear();
    resultValue = 0;

    // setup solver
    setupSolver<DynFullLaplacianInverseSolver>();

    // candidates
    std::vector<Edge> items = buildCandidateSet();

    SubmodularGreedy<Edge> greedy(items, k);

    if (robustnessProblem == Problem::GLOBAL_REDUCTION) {
        greedy.setGainFunction([&](const Edge &e) {
            GraphEvent ev(GraphEvent::EDGE_REMOVAL, e.u, e.v);
            double gain = 0;
            if (metric == Metric::RESISTANCE)
                gain = lapSolver->totalResistanceDifference(ev);
            if (metric == Metric::FOREST)
                gain = lapSolver->totalForestDistanceDifference(ev);
            return gain;
        });
        greedy.setPickedItemCallback([&](const Edge &e) {
            GraphEvent ev(GraphEvent::EDGE_REMOVAL, e.u, e.v);
            G.removeEdge(ev.u, ev.v);
            lapSolver->update(ev);
        });
    } else {
        greedy.setGainFunction([&](const Edge &e) {
            GraphEvent ev(GraphEvent::EDGE_ADDITION, e.u, e.v);
            double gain = 0;
            if (metric == Metric::RESISTANCE)
                gain = lapSolver->totalResistanceDifference(ev);
            if (metric == Metric::FOREST)
                gain = lapSolver->totalForestDistanceDifference(ev);
            return gain;
        });
        greedy.setPickedItemCallback([&](const Edge &e) {
            GraphEvent ev(GraphEvent::EDGE_ADDITION, e.u, e.v);
            G.addEdge(ev.u, ev.v);
            lapSolver->update(ev);
        });
    }

    greedy.run();

    resultValue = greedy.getResultValue();
    result = greedy.getResultItems();

    // reset G to original state
    restoreGraph();

    this->hasRun = true;
}

} // namespace NetworKit
/*
 *  SimplStoch.cpp
 *
 *  Created on: 27.06.2023
 *     Authors: Maria Predari <predarimaria@gmail.com>
 *              Lukas Berner <Lukas.Berner@hu-berlin.de>
 */

#include <networkit/robustness/SimplStoch.hpp>

#include <networkit/robustness/DynFullLaplacianInverseSolver.hpp>
#include <networkit/robustness/DynJLTLaplacianInverseSolver.hpp>
#include <networkit/robustness/StochasticGreedy.hpp>

namespace NetworKit {
SimplStoch::SimplStoch(Graph &G, count k, Problem robustnessProblem, double epsilon, bool useJLT,
                       std::optional<double> solverEpsilon, Metric metric, node focusNode)
    : RobustnessGreedy(G, k, robustnessProblem, metric, focusNode), useJLT(useJLT),
      epsilon(epsilon),
      solverEpsilon(solverEpsilon ? solverEpsilon.value() : (useJLT ? 0.55 : 1e-5)) {}

void SimplStoch::run() {
    // Compute pseudoinverse of laplacian
    prepareGraph();
    result.clear();
    resultValue = 0;

    // setup solver or take copy
    if (useJLT) {
        G.indexEdges();
        setupSolver<DynJLTLaplacianInverseSolver>(solverEpsilon);
    } else
        setupSolver<DynFullLaplacianInverseSolver>(solverEpsilon);

    // candidates
    std::vector<Edge> items = buildCandidateSet();

    DEBUG("candidate size: ", items.size());

    StochasticGreedy<Edge> greedy(items, k, epsilon);

    if (robustnessProblem == Problem::GLOBAL_REDUCTION) {
        greedy.setGainFunction([&](const Edge &e) {
            GraphEvent ev(GraphEvent::EDGE_REMOVAL, e.u, e.v);
            auto gain = lapSolver->totalResistanceDifference(ev);
            return gain;
        });
        greedy.setPickedItemCallback([&](const Edge &e) {
            GraphEvent ev(GraphEvent::EDGE_REMOVAL, e.u, e.v);
            G.removeEdge(ev.u, ev.v);
            if (useJLT)
                G.indexEdges(true);
            lapSolver->update(ev);
        });
    } else {
        greedy.setGainFunction([&](const Edge &e) {
            GraphEvent ev(GraphEvent::EDGE_ADDITION, e.u, e.v);
            auto gain = lapSolver->totalResistanceDifference(ev);
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

    restoreGraph();
    this->hasRun = true;
}

} // namespace NetworKit
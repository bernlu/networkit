/*
 *  SimplStochDyn.cpp
 *
 *  Created on: 27.06.2023
 *     Authors: Maria Predari <predarimaria@gmail.com>
 *              Lukas Berner <Lukas.Berner@hu-berlin.de>
 */

#include <networkit/robustness/SimplStochDyn.hpp>

#include <networkit/robustness/StochasticGreedy.hpp>

namespace NetworKit {
SimplStochDyn::SimplStochDyn(Graph &G, count k, Problem robustnessProblem, double epsilon,
                             bool useJLT, double solverEpsilon, Metric metric, node focusNode)
    : RobustnessGreedy(G, k, robustnessProblem, metric, focusNode),
      DynLapSolver(G, solverEpsilon, useJLT), G(G), epsilon(epsilon) {}

std::vector<Edge> SimplStochDyn::buildCandidateSet(node forestCenter) {
    std::vector<Edge> items;
    switch (robustnessProblem) {
    case Problem::GLOBAL_IMPROVEMENT:
        G.forNodePairs([&](node i, node j) {
            if (!this->G.hasEdge(i, j) && i != forestCenter && j != forestCenter)
                items.push_back(Edge(i, j));
        });
        return items;
    case Problem::LOCAL_IMPROVEMENT:
        G.forNodes([&](node i) {
            if (focusNode != i && !this->G.hasEdge(focusNode, i) && i != forestCenter)
                items.push_back(Edge(focusNode, i));
        });
        return items;
    case Problem::GLOBAL_REDUCTION:
        G.forEdges([&](node i, node j) {
            if (i != forestCenter && j != forestCenter)
                items.push_back(Edge(i, j));
        });
        return items;
    }
    assert(false
           && "unreachable switch statement"); // every case is handled in the switch statement
}

void SimplStochDyn::run() {
    node forestCenter = none;
    if (metric == Metric::FOREST) {
        forestCenter = G.addNode();
        G.forNodes([&](node i) {
            if (i != forestCenter)
                G.addEdge(i, forestCenter);
        });
    }
    result.clear();
    resultValue = 0;
    setupSolver();

    // candidates
    std::vector<Edge> items = buildCandidateSet(forestCenter);

    DEBUG("candidate size: ", items.size());

    StochasticGreedy<Edge> greedy(items, k, epsilon);

    if (robustnessProblem == Problem::GLOBAL_REDUCTION) {
        greedy.setGainFunction([&](const Edge &e) {
            GraphEvent ev(GraphEvent::EDGE_REMOVAL, e.u, e.v);
            auto gain = totalResistanceDifferenceApprox(ev);
            return gain;
        });
        greedy.setPickedItemCallback([&](const Edge &e) {
            GraphEvent ev(GraphEvent::EDGE_REMOVAL, e.u, e.v);
            G.addEdge(e.u, e.v);
            updateEdge(ev);
            // DEBUG("updated LPINV");
            // DEBUG(lpinv);
            // G.addEdge(e.u, e.v);
        });
    } else {
        greedy.setGainFunction([&](const Edge &e) {
            GraphEvent ev(GraphEvent::EDGE_ADDITION, e.u, e.v);
            auto gain = totalResistanceDifferenceApprox(ev);
            return gain;
        });
        greedy.setPickedItemCallback([&](const Edge &e) {
            GraphEvent ev(GraphEvent::EDGE_ADDITION, e.u, e.v);
            G.addEdge(e.u, e.v);
            updateEdge(ev);
            // DEBUG("updated LPINV");
            // DEBUG(lpinv);
            // G.addEdge(e.u, e.v);
        });
    }

    greedy.run();

    resultValue = greedy.getResultValue();
    result = greedy.getResultItems();

    // after run: remove forest center again
    if (metric == Metric::FOREST)
        G.removeNode(forestCenter);

    this->hasRun = true;
}

} // namespace NetworKit
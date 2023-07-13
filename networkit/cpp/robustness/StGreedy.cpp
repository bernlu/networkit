/*
 *  stGreedy.cpp
 *
 *  Created on: 27.06.2023
 *     Authors: Maria Predari <predarimaria@gmail.com>
 *              Lukas Berner <Lukas.Berner@hu-berlin.de>
 */

#include <networkit/robustness/StGreedy.hpp>

namespace NetworKit {

StGreedy::StGreedy(const Graph &G, count k, Problem robustnessProblem, Metric metric,
                   node focusNode)
    : RobustnessGreedy(G, k, robustnessProblem, metric, focusNode), FullLpinv(G, this->metric) {}

std::vector<Edge> StGreedy::buildCandidateSet() {
    std::vector<Edge> items;
    switch (robustnessProblem) {
    case Problem::GLOBAL_IMPROVEMENT:
        for (size_t i = 0; i < G.numberOfNodes(); i++) {
            for (size_t j = 0; j < i; j++) {
                if (i != j && !this->G.hasEdge(i, j)) {
                    items.push_back(Edge(i, j));
                }
            }
        }
        return items;
    case Problem::LOCAL_IMPROVEMENT:
        G.forNodes([&](node v) {
            if (focusNode != v && !this->G.hasEdge(focusNode, v))
                items.push_back(Edge(focusNode, v));
        });
        return items;
    case Problem::GLOBAL_REDUCTION:
        G.forEdges([&](node u, node v) { items.push_back(Edge(u, v)); });
        return items;
    }
    assert(false
           && "unreachable switch statement"); // every case is handled in the switch statement
}

void StGreedy::run() {
    // Compute pseudoinverse of laplacian
    setupLaplacianPseudoinverse(G, metric, robustnessProblem);

    // candidates
    std::vector<Edge> items = buildCandidateSet();

    SubmodularGreedy<Edge> greedy(items, k);

    if (robustnessProblem == Problem::GLOBAL_REDUCTION) {
        greedy.setGainFunction([&](const Edge &e) {
            GraphEvent ev(GraphEvent::EDGE_REMOVAL, e.u, e.v);
            auto gain = laplacianPseudoinverseTraceGain(ev);
            return gain;
        });
        greedy.setPickedItemCallback([&](const Edge &e) {
            GraphEvent ev(GraphEvent::EDGE_REMOVAL, e.u, e.v);
            updateLaplacianPseudoinverse(ev);
            // DEBUG("updated LPINV");
            // DEBUG(lpinv);
            // G.addEdge(e.u, e.v);
        });
    } else {
        greedy.setGainFunction([&](const Edge &e) {
            GraphEvent ev(GraphEvent::EDGE_ADDITION, e.u, e.v);
            auto gain = laplacianPseudoinverseTraceGain(ev);
            return gain;
        });
        greedy.setPickedItemCallback([&](const Edge &e) {
            GraphEvent ev(GraphEvent::EDGE_ADDITION, e.u, e.v);
            updateLaplacianPseudoinverse(ev);
            // DEBUG("updated LPINV");
            // DEBUG(lpinv);
            // G.addEdge(e.u, e.v);
        });
    }

    greedy.run();

    resultValue = greedy.getResultValue();
    result = greedy.getResultItems();

    this->hasRun = true;
}

} // namespace NetworKit
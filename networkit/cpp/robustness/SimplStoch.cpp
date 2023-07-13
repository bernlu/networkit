/*
 *  SimplStoch.cpp
 *
 *  Created on: 27.06.2023
 *     Authors: Maria Predari <predarimaria@gmail.com>
 *              Lukas Berner <Lukas.Berner@hu-berlin.de>
 */

#include <networkit/robustness/SimplStoch.hpp>

#include <networkit/robustness/StochasticGreedy.hpp>

namespace NetworKit {
SimplStoch::SimplStoch(const Graph &G, count k, Problem robustnessProblem, double epsilon,
                       Metric metric, node focusNode, CandidateSetSize candidatesize)
    : RobustnessGreedy(G, k, robustnessProblem, metric, focusNode), FullLpinv(G, this->metric),
      candidatesize(candidatesize), epsilon(epsilon) {}

std::vector<Edge> SimplStoch::buildCandidateSet() {
    std::vector<Edge> items;
    const count n = G.numberOfNodes();
    count lim = std::numeric_limits<count>::max();
    if (this->candidatesize == CandidateSetSize::SMALL)
        lim = std::ceil(std::log(n));
    switch (robustnessProblem) {
    case Problem::GLOBAL_IMPROVEMENT:
        for (count i = 0; i < n; i++) {
            for (count j = 0; j < i; j++) {
                if (!this->G.hasEdge(i, j))
                    items.push_back(Edge(i, j));
                if (items.size() > lim)
                    break;
            }
        }
        return items;
    case Problem::LOCAL_IMPROVEMENT:
        for (size_t i = 0; i < n; i++) {
            if (focusNode != i && !this->G.hasEdge(focusNode, i))
                items.push_back(Edge(focusNode, i));
            if (items.size() > lim) {
                break;
            }
        }
        return items;
    case Problem::GLOBAL_REDUCTION:
        G.forEdges([&](node u, node v) {
            if (items.size() <= lim)
                items.push_back(Edge(u, v));
        });
        return items;
    }
    assert(false
           && "unreachable switch statement"); // every case is handled in the switch statement
}

void SimplStoch::run() {
    // Compute pseudoinverse of laplacian
    setupLaplacianPseudoinverse(G, metric, robustnessProblem);

    // candidates
    std::vector<Edge> items = buildCandidateSet();

    DEBUG("candidate size: ", items.size());

    StochasticGreedy<Edge> greedy(items, k, epsilon);

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
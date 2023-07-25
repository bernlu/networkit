/*
 *  RobustnessGreedy.cpp
 *
 *  Created on: 27.06.2023
 *     Authors: Maria Predari <predarimaria@gmail.com>
 *              Lukas Berner <Lukas.Berner@hu-berlin.de>
 */

#include <networkit/robustness/RobustnessGreedy.hpp>

#include <networkit/algebraic/CSRMatrix.hpp>
#include <networkit/numerics/LAMG/Lamg.hpp>

namespace NetworKit {

RobustnessGreedy::RobustnessGreedy(Graph &G, count k, Problem robustnessProblem, Metric metric,
                                   node focusNode)
    : G(G), k(k), robustnessProblem(robustnessProblem),
      metric(metric == Metric::AUTOMATIC ? defaultMetric(robustnessProblem) : metric),
      focusNode(focusNode) {}

RobustnessGreedy::Metric RobustnessGreedy::defaultMetric(Problem problem) {
    if (problem == Problem::GLOBAL_REDUCTION)
        return Metric::FOREST;
    else
        return Metric::RESISTANCE;
}

void RobustnessGreedy::resetFocus(node focusNode) {
    this->focusNode = focusNode;
    this->hasRun = false;
}

double RobustnessGreedy::getResultValue() const {
    assureFinished();
    return resultValue;
}

std::vector<Edge> RobustnessGreedy::buildCandidateSet() const {
    if (metric == Metric::FOREST)
        assert(forestCenter != none);
    if (robustnessProblem == Problem::LOCAL_IMPROVEMENT)
        assert(focusNode != none);

    std::vector<Edge> items;
    switch (robustnessProblem) {
    case Problem::GLOBAL_IMPROVEMENT:
        G.forNodePairs([&](node u, node v) {
            if (u != forestCenter && v != forestCenter && !G.hasEdge(u, v))
                items.push_back(Edge(u, v));
        });
        return items;
    case Problem::LOCAL_IMPROVEMENT:
        G.forNodes([&](node v) {
            if (focusNode != v && v != forestCenter && !G.hasEdge(focusNode, v))
                items.push_back(Edge(focusNode, v));
        });
        return items;
    case Problem::GLOBAL_REDUCTION:
        G.forEdges([&](node u, node v) {
            if (u != forestCenter && v != forestCenter)
                items.push_back(Edge(u, v));
        });
        return items;
    }
    assert(false
           && "unreachable switch statement"); // every case is handled in the switch statement
    throw std::runtime_error("Error: unhandled enum case in RobustnessGreedy::buildCandidateSet");
}

void RobustnessGreedy::prepareGraph() {
    if (metric == Metric::FOREST) {
        forestCenter = G.addNode();
        G.forNodes([&](node u) {
            if (u != forestCenter)
                G.addEdge(u, forestCenter);
        });
    }
}

void RobustnessGreedy::restoreGraph() {
    // reset G to original state
    for (auto &edge : result) {
        if (robustnessProblem == Problem::GLOBAL_REDUCTION)
            G.addEdge(edge.u, edge.v);
        else
            G.removeEdge(edge.u, edge.v);
    }

    if (metric == Metric::FOREST) {
        G.removeNode(forestCenter);
    }
}

} // namespace NetworKit
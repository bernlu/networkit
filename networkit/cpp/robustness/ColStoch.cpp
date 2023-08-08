/*
 *  ColStoch.cpp
 *
 *  Created on: 27.06.2023
 *     Authors: Maria Predari <predarimaria@gmail.com>
 *              Lukas Berner <Lukas.Berner@hu-berlin.de>
 */

#include <networkit/robustness/ColStoch.hpp>

#include <networkit/robustness/DynJLTLaplacianInverseSolver.hpp>
#include <networkit/robustness/DynLazyLaplacianInverseSolver.hpp>
#include <networkit/robustness/StochasticGreedy.hpp>

namespace NetworKit {
ColStoch::ColStoch(Graph &G, count k, Problem robustnessProblem, double epsilon, double diagEpsilon,
                   bool useJLT, std::optional<double> solverEpsilon, Metric metric, node focusNode)
    : RobustnessGreedy(G, k, robustnessProblem, metric, focusNode), epsilon(epsilon),
      solverEpsilon(solverEpsilon ? solverEpsilon.value() : (useJLT ? 0.55 : 1e-5)), useJLT(useJLT),
      apx(std::make_unique<DynApproxElectricalCloseness>(G, diagEpsilon)) {}

count ColStoch::numberOfNodeCandidates() const {
    const auto n = G.numberOfNodes();
    count s = 0;
    switch (robustnessProblem) {
    case Problem::GLOBAL_IMPROVEMENT:
    case Problem::GLOBAL_REDUCTION:
        s = std::ceil(n * std::sqrt(1. / (double)k * std::log(1.0 / epsilon)));
        if (s < 2)
            s = 2;
        if (s > n)
            s = n / 2;
        break;
    case Problem::LOCAL_IMPROVEMENT:
        s = std::ceil((n - G.degree(focusNode)) / k * std::log(1.0 / epsilon)) + 1;
        break;
    }
    INFO("computed number of node candidates: ", s, "for n=", n, " k=", k, "epsilon=", epsilon);
    return s;
}

std::optional<GraphEvent> ColStoch::makeEvent(node u, node v) const {
    switch (robustnessProblem) {
    case Problem::LOCAL_IMPROVEMENT:
        if (G.hasEdge(focusNode, u))
            return GraphEvent(GraphEvent::EDGE_ADDITION, focusNode, u);
        break;
    case Problem::GLOBAL_IMPROVEMENT:
        if (G.hasEdge(u, v))
            return GraphEvent(GraphEvent::EDGE_ADDITION, u, v);
        break;
    case Problem::GLOBAL_REDUCTION:
        if (!G.hasEdge(u, v))
            return GraphEvent(GraphEvent::EDGE_REMOVAL, u, v);
        break;
    }
    return {};
}

void ColStoch::run() {
    prepareGraph();

    const count n = G.numberOfNodes();
    result.clear();
    resultValue = 0;

    if (useJLT) {
        G.indexEdges();
        setupSolver<DynJLTLaplacianInverseSolver>(solverEpsilon);
    } else
        setupSolver<DynLazyLaplacianInverseSolver>(solverEpsilon);
    INFO("ColStoch: solver setup done");

    if (apxCopy)
        apx = std::make_unique<DynApproxElectricalCloseness>(*apxCopy);
    else {
        apx->run();
        if (robustnessProblem == Problem::LOCAL_IMPROVEMENT)
            apxCopy = std::make_unique<DynApproxElectricalCloseness>(*apx);
    }
    INFO("ColStoch: apx setup done");

    if (k + G.numberOfEdges() > (n * (n - 1) / 8 * 3)) { // 3/4 of the complete graph.
        this->hasRun = true;
        std::cout << "Bad call to TreeGreedy, adding this many edges is not "
                     "supported! Attempting to have "
                  << k + G.numberOfEdges() << " edges, limit is " << n * (n - 1) / 8 * 3;
        return;
    }

    for (count round = 0; round < k; round++) {
        INFO("ColStoch: main loop round ", round);
        double bestGain = -std::numeric_limits<double>::infinity();
        GraphEvent bestEdge;

        int it = 0;

        do {
            INFO("ColStoch: starting node set collection");
            // Collect nodes set for current round
            std::set<node> nodes;
            const auto s = numberOfNodeCandidates();

            double min = std::numeric_limits<double>::infinity();
            double max = -std::numeric_limits<double>::infinity();

            std::vector<double> nodeWeights(n);

            // the random choice following this may fail if all the vertex pairs are
            // already present as edges, we use heuristic information the first
            // time, uniform distribution if it fails
            if (it < 2) {
                G.forNodes([&](node u) {
                    double val = apx->getDiagonal()[u];
                    if (val < min)
                        min = val;
                    if (val > max)
                        max = val;
                    nodeWeights[u] = val;
                });
                for (auto &v : nodeWeights) {
                    double u = 0;
                    switch (robustnessProblem) {
                    case Problem::GLOBAL_IMPROVEMENT:
                    case Problem::LOCAL_IMPROVEMENT:
                        u = v - min; // pick nodes with largest diag
                        break;
                    case Problem::GLOBAL_REDUCTION:
                        u = max - v; // pick nodes with smallest diag
                        break;
                    }
                    v = u * u;
                }
            } else {
                G.forNodes([&](node u) { nodeWeights[u] = 1.; });
            }
            it++;

            INFO("ColStoch: node weights computed");

            std::discrete_distribution<> distribution_nodes_heuristic(nodeWeights.begin(),
                                                                      nodeWeights.end());

            nodeWeights.clear();
            for (count i = 0; i < n; i++) {
                nodeWeights.push_back(1.0);
            }
            std::discrete_distribution<> distribution_nodes_uniform(nodeWeights.begin(),
                                                                    nodeWeights.end());

            while (nodes.size() < s / 2) {
                node randomNode; // make sure not to pick the forest center node
                do {
                    randomNode = distribution_nodes_heuristic(Aux::Random::getURNG());
                } while (randomNode == forestCenter);
                nodes.insert(randomNode);
            }

            while (nodes.size() < s) {
                node randomNode;
                do {
                    randomNode = distribution_nodes_uniform(Aux::Random::getURNG());
                } while (randomNode == forestCenter);
                nodes.insert(randomNode);
            }
            std::vector<node> nodesVec{nodes.begin(), nodes.end()};

            INFO("ColStoch: nodes vec sampled");

            // computeColumns(nodesVec);	// this call is not required (?)

            // Determine best edge between nodes from node set

            for (count i = 0; i < nodesVec.size(); i++) {
                auto u = nodesVec[i];
                if (robustnessProblem == Problem::LOCAL_IMPROVEMENT) {
                    const auto ev = makeEvent(u);
                    if (ev) {
                        double gain = lapSolver->totalResistanceDifference(ev.value());
                        if (gain > bestGain) {
                            bestEdge = ev.value();
                            bestGain = gain;
                        }
                    }
                } else {
                    for (count j = 0; j < i; j++) {
                        auto v = nodesVec[j];
                        const auto ev = makeEvent(u, v);
                        if (ev) {
                            double gain = lapSolver->totalResistanceDifference(ev.value());
                            if (gain > bestGain) {
                                bestEdge = ev.value();
                                bestGain = gain;
                            }
                        }
                    }
                }
            }
            INFO("ColStoch: edges inspected.");

        } while (bestGain == -std::numeric_limits<double>::infinity());

        INFO("ColStoch: best edge found");

        // Accept edge
        resultValue += bestGain;
        auto u = bestEdge.u;
        auto v = bestEdge.v;
        if (bestEdge.type == GraphEvent::EDGE_ADDITION)
            G.addEdge(u, v);
        else
            G.removeEdge(u, v);
        result.push_back(Edge(u, v));

        INFO("ColStoch: edge accepted");

        if (round < k - 1) {
            lapSolver->update(bestEdge);
            apx->update(bestEdge);
        }

        INFO("ColStoch: solver and apx updated");
    }
    // after run: remove forest center again
    restoreGraph();
    INFO("ColStoch: main loop done");
    this->hasRun = true;
    // INFO("Computed columns: ", solver.getComputedColumnCount());
}

} // namespace NetworKit
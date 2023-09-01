/*
 *  RobustnessGreedy.hpp
 *
 *  Created on: 27.06.2023
 *     Authors: Maria Predari <predarimaria@gmail.com>
 *              Lukas Berner <Lukas.Berner@hu-berlin.de>
 */
#ifndef NETWORKIT_ROBUSTNESS_ROBUSTNESS_GREEDY_HPP_
#define NETWORKIT_ROBUSTNESS_ROBUSTNESS_GREEDY_HPP_

#include <networkit/base/Algorithm.hpp>
#include <networkit/graph/Graph.hpp>

#include <networkit/robustness/DynJLTLaplacianInverseSolver.hpp>
#include <networkit/robustness/DynLaplacianInverseSolver.hpp>

#include <optional>

namespace NetworKit {

class RobustnessGreedy : public Algorithm {
public:
    enum Problem { GLOBAL_IMPROVEMENT, LOCAL_IMPROVEMENT, GLOBAL_REDUCTION };
    enum Metric { RESISTANCE, FOREST, AUTOMATIC };

    void resetFocus(node focusNode);

    double getResultValue() const;

    const std::vector<Edge> &getResultItems() const {
        assureFinished();
        return result;
    }

protected:
    RobustnessGreedy(Graph &G, count k, Problem robustnessProblem,
                     Metric metric = Metric::AUTOMATIC, node focusNode = none);
    Graph &G;
    const count k;
    const Problem robustnessProblem;
    const Metric metric;
    node focusNode;
    node forestCenter = none;
    std::unique_ptr<DynLaplacianInverseSolver> lapSolver;
    std::unique_ptr<DynLaplacianInverseSolver> lapSolverOriginal;

    std::vector<Edge> result;
    double resultValue;

    // precond:
    // - G is augmented with forest node if metric==forest
    std::vector<Edge> buildCandidateSet() const;

    template <class Solver>
    void setupSolver(double solverEpsilon, bool jltComputeForestLossCorrection = false) {
        if (lapSolverOriginal)
            lapSolver = std::make_unique<Solver>(static_cast<Solver &>(*lapSolverOriginal));
        else {
            if constexpr (std::is_same_v<Solver, DynJLTLaplacianInverseSolver>)
                lapSolver =
                    std::make_unique<Solver>(G, solverEpsilon, jltComputeForestLossCorrection);
            else
                lapSolver = std::make_unique<Solver>(G, solverEpsilon);
            lapSolver->run();
            if (robustnessProblem == Problem::LOCAL_IMPROVEMENT)
                lapSolverOriginal = std::make_unique<Solver>(static_cast<Solver &>(*lapSolver));
        }
    }

    template <class Solver>
    void setupSolver() {
        if (lapSolverOriginal)
            lapSolver = std::make_unique<Solver>(static_cast<Solver &>(*lapSolverOriginal));
        else {
            lapSolver = std::make_unique<Solver>(G);
            lapSolver->run();
            if (robustnessProblem == Problem::LOCAL_IMPROVEMENT)
                lapSolverOriginal = std::make_unique<Solver>(static_cast<Solver &>(*lapSolver));
        }
    }

    void prepareGraph();

    void restoreGraph();

private:
    static Metric defaultMetric(Problem p);
};

} // namespace NetworKit

#endif // NETWORKIT_ROBUSTNESS_ROBUSTNESS_GREEDY_HPP_

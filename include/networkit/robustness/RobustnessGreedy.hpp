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

#include <networkit/algebraic/DenseMatrix.hpp>
#include <networkit/dynamics/GraphEvent.hpp>

#include <networkit/algebraic/CSRMatrix.hpp>
#include <networkit/algebraic/DenseMatrix.hpp>
#include <networkit/numerics/LAMG/Lamg.hpp>

#include <optional>

namespace NetworKit {

std::ostream &operator<<(std::ostream &os, const DenseMatrix &M);

std::ostream &operator<<(std::ostream &os, const CSRMatrix &M);

class RobustnessGreedy : public Algorithm {
public:
    enum class Problem { GLOBAL_IMPROVEMENT, LOCAL_IMPROVEMENT, GLOBAL_REDUCTION };
    enum class Metric { RESISTANCE, FOREST, none };

    RobustnessGreedy(const Graph &G, count k, Problem robustnessProblem,
                     Metric metric = Metric::none, node focusNode = none);

    void resetFocus(node focusNode);

    double getResultValue() const {
        assureFinished();
        return resultValue;
    };

    const std::vector<Edge> &getResultItems() const {
        assureFinished();
        return result;
    }

protected:
    const Graph &G;
    const count k;
    const Problem robustnessProblem;
    const Metric metric;
    node focusNode;

    std::vector<Edge> result;
    double resultValue;

private:
    static Metric defaultMetric(Problem p);
};

class FullLpinv {
public:
    friend void printAllEdgeGains(Graph &G);

protected:
    FullLpinv(const Graph &G, RobustnessGreedy::Metric metric);

    void setupLaplacianPseudoinverse(const Graph &G, RobustnessGreedy::Metric metric,
                                     RobustnessGreedy::Problem robustnessProblem);

    double laplacianPseudoinverseTraceDifference(const GraphEvent &ev) const;

    double laplacianPseudoinverseTraceGain(const GraphEvent &ev) const {
        return this->lpinv.numberOfColumns() * laplacianPseudoinverseTraceDifference(ev);
    }

    void updateLaplacianPseudoinverse(const GraphEvent &ev);

private:
    DenseMatrix lpinv;
    std::optional<DenseMatrix> lpinvOriginal;

    static DenseMatrix setupLpinv(RobustnessGreedy::Metric metric, count n);
};

void printAllEdgeGains(Graph &G);

class DynLapSolver {
protected:
    DynLapSolver(Graph &G, double tolerance, bool useJLT = false, count eqnPerRound = 200,
                 count roundsPerSolver = 10, count roundsPerColumn = 25)
        : useJLT(useJLT), l(std::max(std::log(eqnPerRound) / (tolerance * tolerance), 1.)), G_(G),
          tolerance(tolerance), eqnPerRound(useJLT ? 2 * l + 2 : eqnPerRound), n(G.numberOfNodes()),
          roundsPerSolver(roundsPerSolver), roundsPerColumn(roundsPerColumn), lamg(tolerance) {
        if (useJLT)
            G.indexEdges();
    }

    void setupSolver();

    double totalResistanceDifferenceApprox(const GraphEvent &ev);
    void updateEdge(const GraphEvent &ev);
    void computeColumns(std::vector<node> nodes);

private:
    // for JLT
    const bool useJLT;
    const count l;
    Graph &G_;

    // generic solver vars
    const double tolerance;
    const count eqnPerRound;
    count n;

    std::vector<count> colAge;
    std::vector<Vector> cols;
    std::vector<Vector> updateVec;
    std::vector<double> updateW;

    count round = 0;
    count solverAge = 0;
    CSRMatrix laplacian;

    // Update solver every n rounds
    const count roundsPerSolver;
    // If a round was last computed this many rounds ago or more, compute by
    // solving instead of updating.
    const count roundsPerColumn;
    Lamg<CSRMatrix> lamg;

    const Vector &getColumn(node u);
    void computeIntermediateMatrices();
    double effR(node u, node v);
    double phiNormSquared(node u, node v);

    // for JLT
    count m;

    DenseMatrix PL, PBL;
    CSRMatrix incidence;
};

} // namespace NetworKit

#endif // NETWORKIT_ROBUSTNESS_SIMPL_STOCH_HPP_
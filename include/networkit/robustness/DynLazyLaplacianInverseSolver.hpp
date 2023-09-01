/*
 *  DynLazyLaplacianInverseSolver.hpp
 *
 *  Created on: 27.06.2023
 *     Authors: Maria Predari <predarimaria@gmail.com>
 *              Lukas Berner <Lukas.Berner@hu-berlin.de>
 */
#ifndef NETWORKIT_ROBUSTNESS_DYN_LAZY_LAPLACIAN_INVERSE_SOLVER_HPP_
#define NETWORKIT_ROBUSTNESS_DYN_LAZY_LAPLACIAN_INVERSE_SOLVER_HPP_

#include <networkit/base/Algorithm.hpp>
#include <networkit/base/DynAlgorithm.hpp>

#include <networkit/algebraic/CSRMatrix.hpp>
#include <networkit/algebraic/DenseMatrix.hpp>
#include <networkit/numerics/LAMG/Lamg.hpp>

#include <networkit/robustness/DynLaplacianInverseSolver.hpp>

namespace NetworKit {

class DynLazyLaplacianInverseSolver : public DynLaplacianInverseSolver {
public:
    DynLazyLaplacianInverseSolver(const Graph &G, double tolerance, count eqnPerRound = 200,
                                  count roundsPerSolver = 10, count roundsPerColumn = 25);

    void run() override;

    double totalResistanceDifference(const GraphEvent &ev) const;
    double totalForestDistanceDifference(const GraphEvent &ev) const override;
    void update(GraphEvent ev) override;
    void computeColumns(std::vector<node> nodes) const;

    std::vector<Vector> parallelSolve(std::vector<Vector> &rhss) const;
    const Vector &getColumn(node u) const;

private:
    const count n;
    const double tolerance;
    const count eqnPerRound;

    mutable std::vector<count> colAge;
    mutable std::vector<Vector> cols;
    mutable Lamg<CSRMatrix> lamg;

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
};

} // namespace NetworKit

#endif // NETWORKIT_ROBUSTNESS_DYN_LAZY_LAPLACIAN_INVERSE_SOLVER_HPP_

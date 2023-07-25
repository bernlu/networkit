/*
 *  DynJLTLaplacianInverseSolver.hpp
 *
 *  Created on: 27.06.2023
 *     Authors: Maria Predari <predarimaria@gmail.com>
 *              Lukas Berner <Lukas.Berner@hu-berlin.de>
 */
#ifndef NETWORKIT_ROBUSTNESS_DYN_JLT_LAPLACIAN_INVERSE_SOLVER_HPP_
#define NETWORKIT_ROBUSTNESS_DYN_JLT_LAPLACIAN_INVERSE_SOLVER_HPP_

#include <networkit/algebraic/DenseMatrix.hpp>
#include <networkit/algebraic/DynamicMatrix.hpp>
#include <networkit/base/Algorithm.hpp>
#include <networkit/base/DynAlgorithm.hpp>
#include <networkit/robustness/DynLaplacianInverseSolver.hpp>
#include <networkit/robustness/DynLazyLaplacianInverseSolver.hpp>

namespace NetworKit {

class DynJLTLaplacianInverseSolver : public DynLaplacianInverseSolver {
public:
    DynJLTLaplacianInverseSolver(const Graph &G, double tolerance, count eqnPerRound = 200,
                                 count roundsPerSolver = 10, count roundsPerColumn = 25);

    void run() override;
    void update(GraphEvent) override;
    double totalResistanceDifference(const GraphEvent &ev) const;

private:
    const count n;
    const count l;
    const double tolerance;
    count m;
    void computeIntermediateMatrices();
    double effR(node u, node v) const;
    double phiNormSquared(node u, node v) const;

    // for JLT
    DynLazyLaplacianInverseSolver solver;

    DenseMatrix PL, PBL;

    std::normal_distribution<> d{0, 1};

    CSRMatrix incidence;
};

} // namespace NetworKit

#endif // NETWORKIT_ROBUSTNESS_DYN_JLT_LAPLACIAN_INVERSE_SOLVER_HPP_

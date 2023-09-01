/*
 *  DynFullLaplacianInverseSolver.hpp
 *
 *  Created on: 27.06.2023
 *     Authors: Maria Predari <predarimaria@gmail.com>
 *              Lukas Berner <Lukas.Berner@hu-berlin.de>
 */
#ifndef NETWORKIT_ROBUSTNESS_DYN_FULL_LAPLACIAN_INVERSE_SOLVER_HPP_
#define NETWORKIT_ROBUSTNESS_DYN_FULL_LAPLACIAN_INVERSE_SOLVER_HPP_

#include <networkit/algebraic/DenseMatrix.hpp>
#include <networkit/base/Algorithm.hpp>
#include <networkit/base/DynAlgorithm.hpp>
#include <networkit/robustness/DynLaplacianInverseSolver.hpp>

namespace NetworKit {

class DynFullLaplacianInverseSolver : public DynLaplacianInverseSolver {
public:
    friend void printAllEdgeGains(Graph &G);

    DynFullLaplacianInverseSolver(const Graph &G);

    void run() override;
    void update(GraphEvent ev) override;
    virtual double totalResistanceDifference(const GraphEvent &ev) const override;
    virtual double totalForestDistanceDifference(const GraphEvent &ev) const override;

private:
    DenseMatrix lpinv;
};

} // namespace NetworKit

#endif // NETWORKIT_ROBUSTNESS_DYN_FULL_LAPLACIAN_INVERSE_SOLVER_HPP_

/*
 *  DynSpectralLaplacianInverseSolver.hpp
 *
 *  Created on: 27.06.2023
 *     Authors: Maria Predari <predarimaria@gmail.com>
 *              Lukas Berner <Lukas.Berner@hu-berlin.de>
 */
#ifndef NETWORKIT_ROBUSTNESS_DYN_SPECTRAL_LAPLACIAN_INVERSE_SOLVER_HPP_
#define NETWORKIT_ROBUSTNESS_DYN_SPECTRAL_LAPLACIAN_INVERSE_SOLVER_HPP_

#include <networkit/base/Algorithm.hpp>
#include <networkit/base/DynAlgorithm.hpp>

#include <networkit/algebraic/DenseMatrix.hpp>
#include <networkit/robustness/DynLaplacianInverseSolver.hpp>

#include <slepc.h>

namespace NetworKit {

class DynSpectralLaplacianInverseSolver : public DynLaplacianInverseSolver {
public:
    /** Sets up the eigendecomposition of an input graph @a g. It creates the
     * necessary matrix format sets the corresponding options and creates the
     * eigendecomposition environment. Given the number of requested eigenparis @a
     * numberOfEigenpairs, it performs a truncated eigendecompositing of the
     * Laplacian matrix of @a g as: L = U\LambdaU^T. Since our matrices are
     * considered to be of Laplacian type by default, we deflate the eigenvector
     * that corresponds to the zero eigenvalue to boost the convergence of the
     * involved methods.
     *
     * @param	G			the graph
     * @param	k			the number of expected edge additions so that we
     * include it in the preallocation of the matrix.
     * @param	numberOfEigenpairs	number of requested eigenpairs on the lower
     * end of the spectrum.
     */
    DynSpectralLaplacianInverseSolver(const Graph &G, count numberOfEigenpairs);

    // copy constructor
    DynSpectralLaplacianInverseSolver(const DynSpectralLaplacianInverseSolver &other);

    DynSpectralLaplacianInverseSolver(DynSpectralLaplacianInverseSolver &&other) noexcept = default;

    ~DynSpectralLaplacianInverseSolver();

    DynSpectralLaplacianInverseSolver &operator=(DynSpectralLaplacianInverseSolver other) noexcept;

    friend void swap(DynSpectralLaplacianInverseSolver &first,
                     DynSpectralLaplacianInverseSolver &second) noexcept;

    /** Updates the eigensolver environment. To do so it resets the (updated)
     matrix, the initial space with the current solution (to speed up convergence)
     and the deflation space.
     * Finally reruns the eigensolver with the new setting.

     * @return Petsc error code in case of a failure of the EPSSolve().
     */

    void update_eigensolver();

    double totalResistanceDifference(const GraphEvent &ev) const;
    double totalForestDistanceDifference(const GraphEvent &ev) const override;
    void update(GraphEvent ev) override;
    void run() override;

    /**
     * Runs the eigensolver.
     * Computes c eigenpairs in the lower side of the spectrum and one on the
     right side.

     * @return Petsc error code in case of a failure of the EPSSolve().
     */

    void run_eigensolver();

    /**
     * Prints information regarding the eigensolver.
     */

    void info_eigensolver() {

        EPSType type;        /* type of solver */
        PetscReal tol;       /* error and tolerance of the solver */
        PetscInt maxit, its; /* max iterations, actual iterations */
        PetscInt nev, nc;    /* # of computed values, # of converged values */
        DEBUG(" -------------- INFO ----------------- ");
        EPSGetType(eps, &type);
        DEBUG(" SOLUTION METHOD: ", type);
        EPSGetIterationNumber(eps, &its);
        DEBUG(" ITERATION COUNT: ", its);
        EPSGetTolerances(eps, &tol, &maxit);
        DEBUG(" STOP COND: tol= ", (double)tol, " maxit= ", maxit);
        EPSGetDimensions(eps, &nev, NULL, NULL);
        DEBUG(" COMPUT EVALUES: ", nev);
        EPSGetConverged(eps, &nc);
        DEBUG(" CONVRG EVALUES: ", nc);
        PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_INFO_DETAIL);
        EPSConvergedReasonView(eps, PETSC_VIEWER_STDOUT_WORLD);
        EPSErrorView(eps, EPS_ERROR_RELATIVE, PETSC_VIEWER_STDOUT_WORLD);
        PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);
        EPSView(eps, PETSC_VIEWER_STDOUT_WORLD);
    }

private:
    /**
     * Computes gain(a,b) for two vertices @a a, @a b using
     * separate averages for the effective resistance and the biharmonic
     * distances. First computes the averages: D_appx = (upD + lowD)/2, R_appx =
     * (upR + lowR)/2. Then computes the gain approximation as gain(a,b) = D_appx
     * / (1.0 + R_approx).
     * @param      a      vertex
     * @param      b      vertex
     * @return approximation of gain difference.
     */
    double SpectralApproximationGainDifference2(node a, node b) const;

    /** Adds the elements that correspond to edge @a u, @a v into the operator
          matrix.
          * Currently implemented for unweighted entries.
          * @param      u       a valid vertex value
          * @param      v       a valid vertex value

         */
    void addEdge(NetworKit::node u, NetworKit::node v);

    /** Removes the elements that correspond to edge @a u, @a v into the operator
             matrix.
             * Currently implemented for unweighted entries.
             * @param      u       a valid vertex value
             * @param      v       a valid vertex value

            */
    void removeEdge(NetworKit::node u, NetworKit::node v);
    /**
     * Sets values of a Petsc matrix @a A based on an input graph information.
     * The values are inserted row by row.
     *
     * @param       g            Graph from witch we set the values of @a A
     * @param       nnz           the nnz/degree information for each row/node.
     * @param	A	     Petsc type matrix.
     */
    void MatSetValuesROW(NetworKit::Graph const &g, PetscInt *nnz, Mat *A);

    EPS eps;               /* eigenproblem solver context*/
    Mat A;                 /* operator matrix */
    PetscInt n;            /* size of matrix */
    Vec x;                 /* vector representing the nullspace */
    PetscInt c, nconv = 0; /* # requested values, # converged low values */
    double *e_vectors;     /* stores the eigenvectors (of size n*nconv) */
    double *e_values;      /* stores eigenvalues (of size nconv + 1) */
    Vec *Q;
};

} // namespace NetworKit

#endif // NETWORKIT_ROBUSTNESS_DYN_SPECTRAL_LAPLACIAN_INVERSE_SOLVER_HPP_

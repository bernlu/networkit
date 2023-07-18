/*
 *  DynSpectralLaplacianInverseSolver.cpp
 *
 *  Created on: 27.06.2023
 *     Authors: Maria Predari <predarimaria@gmail.com>
 *              Lukas Berner <Lukas.Berner@hu-berlin.de>
 */

#include <networkit/robustness/DynSpectralLaplacianInverseSolver.hpp>

#include <networkit/algebraic/DynamicMatrix.hpp>
#include <networkit/numerics/LAMG/Lamg.hpp>

#include <cstdlib>
#include <cstring>

namespace NetworKit {

DynSpectralLaplacianInverseSolver::DynSpectralLaplacianInverseSolver(const Graph &G,
                                                                     count numberOfEigenpairs)
    : DynLaplacianInverseSolver(G), n(G.numberOfNodes()), c(numberOfEigenpairs) {

    if (!numberOfEigenpairs) {
        throw std::runtime_error("Requesting no eigenpairs!");
    }

    int argc = 0;
    SlepcInitialize(&argc, nullptr, nullptr, nullptr);

    DEBUG("GRAPH INPUT: (n = ", n, " m = ", G.numberOfEdges(), ")\n");

    // TODO: ADJUST FOR ALLOCATING MORE SPACE BASED ON k!
    // INSTEAD OF DEGREE(V) + 1, ALLOCATE DEGREE(V) + 1 + k
    // TO AVOID ANOTHER MALLOC (k IS SMALL COMPARED TO AVG DEGREE).

    PetscInt *nnz = (PetscInt *)malloc(n * sizeof(PetscInt));
    G.forNodes([&](NetworKit::node v) { nnz[v] = (PetscInt)G.degree(v) + 1; });

    // includes preallocation
    MatCreateSeqAIJ(PETSC_COMM_WORLD, n, n, 0, nnz, &A);
    MatSetOption(A, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
    MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetType(A, MATSEQAIJ);
    MatSetUp(A);

    MatSetValuesROW(G, nnz, &A);
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    free(nnz);

    // storage for eigenpairs
    e_vectors = (double *)calloc(1, n * c * sizeof(double));
    e_values = (double *)calloc(1, (c + 1) * sizeof(double));
    // Vec x for deflation
    MatCreateVecs(A, &x, NULL);
    VecSet(x, 1.0);

    // create eps environment
    EPSCreate(PETSC_COMM_WORLD, &eps);
    EPSSetOperators(eps, A, NULL);
    EPSSetProblemType(eps, EPS_HEP);
    EPSSetFromOptions(eps);
    // set deflation space
    EPSSetDeflationSpace(eps, 1, &x);

    run_eigensolver();
}

// copy constructor
DynSpectralLaplacianInverseSolver::DynSpectralLaplacianInverseSolver(
    const DynSpectralLaplacianInverseSolver &other)
    : DynLaplacianInverseSolver(other.G), n(other.n), c(other.c), nconv(other.nconv) {
    e_vectors = (double *)calloc(1, n * c * sizeof(double));
    e_values = (double *)calloc(1, (c + 1) * sizeof(double));
    std::memcpy(e_vectors, other.e_vectors, n * c * sizeof(double));
    std::memcpy(e_values, other.e_values, (c + 1) * sizeof(double));

    VecDuplicate(other.x, &x);
    VecCopy(other.x, x);

    MatConvert(other.A, MATSAME, MAT_INITIAL_MATRIX, &A);

    VecDuplicateVecs(other.Q[0], nconv + 1, &Q);
    for (int i = 0; i < nconv + 1; i++) {
        VecCopy(other.Q[i], Q[i]);
    }

    // from setup:
    // create eps environment
    EPSCreate(PETSC_COMM_WORLD, &eps);
    EPSSetOperators(eps, A, NULL);
    EPSSetProblemType(eps, EPS_HEP);
    EPSSetFromOptions(eps);
    // set deflation space
    EPSSetDeflationSpace(eps, 1, &x);

    // from update_eigensolver:
    EPSSetInitialSpace(eps, nconv + 1, Q);
}

DynSpectralLaplacianInverseSolver::~DynSpectralLaplacianInverseSolver() {
    free(e_vectors);
    free(e_values);
    EPSDestroy(&eps);
    MatDestroy(&A);
    VecDestroy(&x);
    VecDestroyVecs(nconv + 1, &Q);
}

DynSpectralLaplacianInverseSolver &
DynSpectralLaplacianInverseSolver::operator=(DynSpectralLaplacianInverseSolver other) noexcept {
    swap(*this, other);
    return *this;
}

void swap(DynSpectralLaplacianInverseSolver &first,
          DynSpectralLaplacianInverseSolver &second) noexcept {
    using std::swap;

    swap(first.eps, second.eps);
    swap(first.A, second.A);
    swap(first.n, second.n);
    swap(first.x, second.x);
    swap(first.c, second.c);
    swap(first.nconv, second.nconv);
    swap(first.e_vectors, second.e_vectors);
    swap(first.e_values, second.e_values);
    swap(first.Q, second.Q);
}

void DynSpectralLaplacianInverseSolver::update_eigensolver() {

    EPSSetOperators(eps, A, NULL);
    EPSSetDeflationSpace(eps, 1, &x);
    EPSSetInitialSpace(eps, nconv + 1, Q);
    run_eigensolver();
}

void DynSpectralLaplacianInverseSolver::run_eigensolver() {

    // PetscReal error;
    // PetscReal norm;

    PetscScalar val;
    Vec vec;
    PetscInt i;
    // allocate for eigenvector
    MatCreateVecs(A, NULL, &vec);

    EPSSetDimensions(eps, c, PETSC_DEFAULT, PETSC_DEFAULT);
    EPSSetWhichEigenpairs(eps, EPS_SMALLEST_MAGNITUDE);
    EPSSolve(eps);
    EPSGetConverged(eps, &nconv);
    if (nconv > c)
        nconv = c;
    // allocation eigenvectors (one more for largest eigenvector)
    VecDuplicateVecs(vec, nconv + 1, &Q);

    for (i = 0; i < nconv; i++) {
        EPSGetEigenpair(eps, i, &val, NULL, vec, NULL);
        // compute relative error associated to each eigenpair
        // EPSComputeError(eps, i, EPS_ERROR_RELATIVE, &error);
        // PetscPrintf(PETSC_COMM_WORLD,"   %12f      %12g \n", (double)val,
        // (double)error); PetscPrintf(PETSC_COMM_WORLD,"\n"); VecNorm(vec,
        // NORM_2, &norm); DEBUG("Norm of evector %d : %g \n", i, norm);
        // VecView(vec, PETSC_VIEWER_STDOUT_WORLD);
        e_values[i] = (double)val;
        VecCopy(vec, Q[i]);
        for (PetscInt j = 0; j < n; j++) {
            PetscScalar w;
            VecGetValues(vec, 1, &j, &w);
            *(e_vectors + i + j * c) = (double)w;
        }
    }

    // reset for largest eigenpair
    EPSSetDimensions(eps, 1, PETSC_DEFAULT, PETSC_DEFAULT);
    EPSSetWhichEigenpairs(eps, EPS_LARGEST_MAGNITUDE);
    EPSSolve(eps);
    PetscInt nconv_l;
    EPSGetConverged(eps, &nconv_l);
    if (!nconv_l) {
        throw std::runtime_error("Largest eigenvalue has not converged!");
    }
    EPSGetEigenpair(eps, 0, &val, NULL, vec, NULL);
    // DEBUG("           k          ||Ax-kx||/||kx||\n"
    //"   ----------------- ------------------\n");
    // EPSComputeError(eps, 0, EPS_ERROR_RELATIVE, &error);
    // DEBUG("   %12f       %12g\n",(double)val,(double)error, "\n");
    //  store top eigenvector at the end of Q.
    VecCopy(vec, Q[i]);
    e_values[i] = val;
    VecDestroy(&vec);
}

/**
 * Sets values of a Petsc matrix @a A based on an input graph information.
 * The values are inserted row by row.
 *
 * @param       g            Graph from witch we set the values of @a A
 * @param       nnz           the nnz/degree information for each row/node.
 * @param	A	     Petsc type matrix.
 */
void DynSpectralLaplacianInverseSolver::MatSetValuesROW(Graph const &g, PetscInt *nnz, Mat *A) {
    g.forNodes([&](const node v) {
        double weightedDegree = 0.0;
        PetscInt *col = (PetscInt *)malloc(nnz[v] * sizeof(PetscInt));
        PetscScalar *val = (PetscScalar *)malloc(nnz[v] * sizeof(PetscScalar));
        unsigned int idx = 0;
        g.forNeighborsOf(v, [&](const node u, double w) { // - adj  mat
            // exclude diag. (would be subtracted by adj weight
            if (u != v) {
                weightedDegree += w;
            }
            col[idx] = (PetscInt)u;
            val[idx] = -(PetscScalar)w;
            idx++;
        });
        col[idx] = v;
        val[idx] = weightedDegree;
        PetscInt a = (PetscInt)v;
        MatSetValues(*A, 1, &a, nnz[v], col, val, INSERT_VALUES);
    });
}

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

double DynSpectralLaplacianInverseSolver::SpectralApproximationGainDifference2(node a,
                                                                               node b) const {

    double upD = 0.0, upR = 0.0, lowD = 0.0, lowR = 0.0;
    double lambda_n = 1.0 / (e_values[nconv] * e_values[nconv]);
    double lambda_c = 1.0 / (e_values[nconv - 1] * e_values[nconv - 1]);
    double sq_diff;

    for (int i = 0; i < nconv; i++) {
        sq_diff = *(e_vectors + a * c + i) - *(e_vectors + b * c + i);
        sq_diff *= sq_diff;

        lowD += (1.0 / (e_values[i] * e_values[i]) - lambda_n) * sq_diff;
        upR += (1.0 / e_values[i] - 1.0 / e_values[nconv - 1]) * sq_diff;

        upD += (1.0 / (e_values[i] * e_values[i]) - lambda_c) * sq_diff;
        lowR += (1.0 / e_values[i] - 1.0 / e_values[nconv]) * sq_diff;
    }

    return (upD + 2.0 * lambda_c + lowD + 2.0 * lambda_n)
           / (2.0 + 2.0 / e_values[nconv] + 2.0 / e_values[nconv - 1] + upR + lowR);
}

/** Adds the elements that correspond to edge @a u, @a v into the operator
  matrix.
  * Currently implemented for unweighted entries.
  * @param      u       a valid vertex value
  * @param      v       a valid vertex value

 */
void DynSpectralLaplacianInverseSolver::addEdge(node u, node v) {

    if (u == v) {
        std::cout << "Warning: Graph has edge with equal target and destination!";
        return;
    }

    PetscInt a = (PetscInt)u;
    PetscInt b = (PetscInt)v;
    PetscScalar w = 1.0;
    PetscScalar nw = -1.0;

    MatSetValues(A, 1, &a, 1, &a, &w, ADD_VALUES);
    MatSetValues(A, 1, &b, 1, &b, &w, ADD_VALUES);
    MatSetValues(A, 1, &a, 1, &b, &nw, ADD_VALUES);
    MatSetValues(A, 1, &b, 1, &a, &nw, ADD_VALUES);

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}

void DynSpectralLaplacianInverseSolver::removeEdge(node u, node v) {

    if (u == v) {
        std::cout << "Warning: Graph has edge with equal target and destination!";
        return;
    }

    PetscInt a = (PetscInt)u;
    PetscInt b = (PetscInt)v;
    PetscScalar w = 1.0;
    PetscScalar nw = -1.0;

    MatSetValues(A, 1, &a, 1, &a, &nw, ADD_VALUES);
    MatSetValues(A, 1, &b, 1, &b, &nw, ADD_VALUES);
    MatSetValues(A, 1, &a, 1, &b, &w, ADD_VALUES);
    MatSetValues(A, 1, &b, 1, &a, &w, ADD_VALUES);

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}

double DynSpectralLaplacianInverseSolver::totalResistanceDifference(const GraphEvent &ev) const {
    return SpectralApproximationGainDifference2(ev.u, ev.v) * static_cast<double>(n);
}

void DynSpectralLaplacianInverseSolver::update(GraphEvent ev) {
    if (ev.type == GraphEvent::EDGE_ADDITION)
        addEdge(ev.u, ev.v);
    else if (ev.type == GraphEvent::EDGE_REMOVAL)
        removeEdge(ev.u, ev.v);
    else
        throw std::logic_error(
            "Error: DynSpectralLaplacianInverseSolver onl supports edge additions and removals.");
    update_eigensolver();
}

void DynSpectralLaplacianInverseSolver::run() {
    run_eigensolver();
}

} // namespace NetworKit
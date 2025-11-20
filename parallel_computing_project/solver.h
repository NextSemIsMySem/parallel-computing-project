#ifndef SOLVER_H
#define SOLVER_H

#include <vector>

// --- Function Declarations for jacobi_serial.cpp ---

/**
 * @brief Solves the linear system Ax = b using the sequential Weighted Jacobi method.
 * 
 * @param A The NxN matrix.
 * @param b The N-dimensional vector.
 * @param N The size of the matrix and vectors.
 * @param max_iterations The maximum number of iterations to perform.
 * @param tolerance The convergence tolerance for the L2 norm of the error.
 * @param omega The weight for the Jacobi method.
 * @param iterations_taken Pointer to store the number of iterations performed.
 * @return The solution vector x.
 */
std::vector<double> jacobi_serial(
    const std::vector<double>& A, 
    const std::vector<double>& b, 
    int N, 
    int max_iterations, 
    double tolerance, 
    double omega,
    int& iterations_taken
);

// --- Function Declarations for jacobi_omp.cpp ---

/**
 * @brief Solves the linear system Ax = b using the OpenMP parallel Weighted Jacobi method.
 * 
 * @param A The NxN matrix.
 * @param b The N-dimensional vector.
 * @param N The size of the matrix and vectors.
 * @param max_iterations The maximum number of iterations to perform.
 * @param tolerance The convergence tolerance for the L2 norm of the error.
 * @param omega The weight for the Jacobi method.
 * @param iterations_taken Pointer to store the number of iterations performed.
 * @return The solution vector x.
 */
std::vector<double> jacobi_omp(
    const std::vector<double>& A, 
    const std::vector<double>& b, 
    int N, 
    int max_iterations, 
    double tolerance, 
    double omega,
    int& iterations_taken
);

// --- Function Declarations for utils.cpp ---

/**
 * @brief Generates a strictly diagonally dominant NxN matrix.
 * 
 * This ensures that the Jacobi method will converge.
 * The matrix is flattened into a 1D vector in row-major order.
 * 
 * @param N The dimension of the matrix.
 * @return A std::vector<double> representing the flattened matrix.
 */
std::vector<double> generate_diagonally_dominant_matrix(int N);

/**
 * @brief Generates an N-dimensional vector with random values.
 * 
 * @param N The dimension of the vector.
 * @return A std::vector<double> with random values.
 */
std::vector<double> generate_vector_b(int N);

/**
 * @brief Calculates the L2 Norm (Euclidean norm) of the error vector.
 * 
 * The error is calculated as Ax - b.
 * 
 * @param A The NxN matrix.
 * @param x The current solution vector.
 * @param b The right-hand side vector.
 * @param N The size of the system.
 * @return The L2 norm of the error.
 */
double calculate_l2_norm(
    const std::vector<double>& A, 
    const std::vector<double>& x, 
    const std::vector<double>& b, 
    int N
);

#endif // SOLVER_H

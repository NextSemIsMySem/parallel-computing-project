#include "solver.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm> // For std::swap

/**
 * @brief Solves the linear system Ax = b using the sequential Weighted Jacobi method.
 *
 * This function iteratively computes the solution to Ax = b.
 * The iteration formula is:
 * x_new[i] = (1 - omega) * x_old[i] + (omega / A[i][i]) * (b[i] - sigma)
 * where sigma is the sum of A[i][j] * x_old[j] for j != i.
 *
 * @param A The NxN matrix (flattened).
 * @param b The N-dimensional vector.
 * @param N The size of the matrix and vectors.
 * @param max_iterations The maximum number of iterations.
 * @param tolerance The convergence tolerance.
 * @param omega The weight factor.
 * @param iterations_taken Reference to store the number of iterations performed.
 * @return The solution vector x.
 */
std::vector<double> jacobi_serial(
    const std::vector<double>& A, 
    const std::vector<double>& b, 
    int N, 
    int max_iterations, 
    double tolerance, 
    double omega,
    int& iterations_taken) 
{
    // Initialize solution vector x with zeros
    std::vector<double> x_current(N, 0.0);
    std::vector<double> x_next(N, 0.0);

    double error = tolerance + 1.0; // Ensure the loop starts
    iterations_taken = 0;

    while (iterations_taken < max_iterations && error > tolerance) {
        for (int i = 0; i < N; ++i) {
            double sigma = 0.0;
            for (int j = 0; j < N; ++j) {
                if (i != j) {
                    sigma += A[i * N + j] * x_current[j];
                }
            }
            
            double A_ii = A[i * N + i];
            if (std::abs(A_ii) < 1e-9) { // Avoid division by zero
                // Handle singular or near-singular matrix case
                // For this assignment, we assume A is well-behaved
                A_ii = 1e-9;
            }

            x_next[i] = (1.0 - omega) * x_current[i] + (omega / A_ii) * (b[i] - sigma);
        }

        // Efficiency: Swap pointers instead of copying data.
        // x_current becomes the new x_next for the next iteration,
        // and the old x_next (which now holds the new values) becomes x_current.
        std::swap(x_current, x_next);

        // Convergence check is done less frequently to reduce overhead
        if (iterations_taken % 10 == 0) {
            error = calculate_l2_norm(A, x_current, b, N);
        }
        
        iterations_taken++;
    }
    
    // Final error calculation
    error = calculate_l2_norm(A, x_current, b, N);
    std::cout << "Final Error (L2 Norm): " << error << std::endl;

    return x_current;
}

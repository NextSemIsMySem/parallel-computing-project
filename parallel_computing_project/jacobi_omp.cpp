#include "solver.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <algorithm> // For std::swap

using namespace std;

/**
 * @brief Solves the linear system Ax = b using the OpenMP parallel Weighted Jacobi method.
 *
 * This function parallelizes the main loop of the Jacobi iteration using OpenMP.
 * The iteration formula is the same as the serial version.
 *
 * Parallelization Strategy:
 * - The outer `for` loop over `i` (rows) is parallelized using `#pragma omp parallel for`.
 * - Each thread computes a subset of the `x_next` elements independently.
 * - Data Dependencies: The calculation of `x_next[i]` depends only on `x_current` values,
 *   which are read-only within the loop. This avoids race conditions on `x_current`.
 * - Race Conditions: Since each thread writes to a different `x_next[i]`, there are no
 *   write-write race conditions. `x_next` is the shared resource being updated.
 * - Scheduling: We use the default `schedule(static)`. This divides the iterations
 *   (rows) into chunks of equal size and distributes them among threads. This is
*   efficient when the workload for each iteration is uniform, which is the case here.
 *   For non-uniform workloads, `schedule(dynamic)` could be investigated for better
 *   load balancing.
 * - Pointer Swapping: `std::swap` is used to avoid expensive array copying between
 *   iterations, which is crucial for performance. This happens in the serial part
 *   of the code.
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
vector<double> jacobi_omp(
    const vector<double>& A, 
    const vector<double>& b, 
    int N, 
    int max_iterations, 
    double tolerance, 
    double omega,
    int& iterations_taken) 
{
    vector<double> x_current(N, 0.0);
    vector<double> x_next(N, 0.0);

    double error = tolerance + 1.0; // Ensure the loop starts
    iterations_taken = 0;

    while (iterations_taken < max_iterations && error > tolerance) {
        // Parallelize the main computation loop.
        // Each thread will handle a portion of the rows of the matrix.
        // The default schedule is `static`, which is suitable for this problem
        // as the workload for each row `i` is roughly the same.
        // For investigation, one could change this to `schedule(dynamic)`.
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            double sigma = 0.0;
            // This inner loop is not parallelized to avoid overhead and complexity.
            // The outer loop provides sufficient granularity for parallelization.
            for (int j = 0; j < N; ++j) {
                if (i != j) {
                    // Reading from x_current is safe as it's not modified in this loop.
                    sigma += A[i * N + j] * x_current[j];
                }
            }
            
            double A_ii = A[i * N + i];
            if (abs(A_ii) < 1e-9) {
                A_ii = 1e-9;
            }

            // Writing to x_next[i] is safe because each thread writes to a unique index.
            x_next[i] = (1.0 - omega) * x_current[i] + (omega / A_ii) * (b[i] - sigma);
        }

        // Pointer swap is a serial operation, performed by the master thread.
        // This is a critical optimization to avoid deep copying the vectors.
        swap(x_current, x_next);

        // Convergence check is performed serially and periodically.
        if (iterations_taken % 10 == 0) {
            error = calculate_l2_norm(A, x_current, b, N);
        }
        
        iterations_taken++;
    }
    
    // Final error calculation
    error = calculate_l2_norm(A, x_current, b, N);
    cout << "Final Error (L2 Norm): " << error << endl;

    return x_current;
}

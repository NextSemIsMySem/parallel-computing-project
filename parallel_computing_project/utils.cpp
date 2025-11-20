#include "solver.h"
#include <vector>
#include <cmath>
#include <random>
#include <numeric>

using namespace std;

/**
 * @brief Generates a strictly diagonally dominant NxN matrix.
 *
 * A matrix is strictly diagonally dominant if for every row, the absolute
 * value of the diagonal element is greater than the sum of the absolute
 * values of all other elements in that row. This property guarantees
 * that the Jacobi method will converge.
 * The matrix is flattened into a 1D vector (row-major) for better cache performance.
 *
 * @param N The dimension of the matrix.
 * @return A std::vector<double> representing the flattened matrix.
 */
vector<double> generate_diagonally_dominant_matrix(int N) {
    vector<double> A(N * N);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < N; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < N; ++j) {
            if (i != j) {
                A[i * N + j] = dis(gen);
                row_sum += abs(A[i * N + j]);
            }
        }
        // Set the diagonal element to be greater than the sum of other elements
        A[i * N + i] = row_sum + dis(gen) + 1.0; 
    }
    return A;
}

/**
 * @brief Generates an N-dimensional vector b with random values.
 *
 * @param N The dimension of the vector.
 * @return A std::vector<double> with random values between 0 and 1.
 */
vector<double> generate_vector_b(int N) {
    vector<double> b(N);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < N; ++i) {
        b[i] = dis(gen);
    }
    return b;
}

/**
 * @brief Calculates the L2 Norm (Euclidean norm) of the residual vector (b - Ax).
 *
 * The L2 norm is sqrt(sum(residual[i]^2)). It is used to check for convergence.
 *
 * @param A The NxN matrix (flattened).
 * @param x The current solution vector.
 * @param b The right-hand side vector.
 * @param N The size of the system.
 * @return The L2 norm of the residual.
 */
double calculate_l2_norm(const vector<double>& A, const vector<double>& x, const vector<double>& b, int N) {
    double norm = 0.0;
    for (int i = 0; i < N; ++i) {
        double residual_i = -b[i];
        for (int j = 0; j < N; ++j) {
            residual_i += A[i * N + j] * x[j];
        }
        norm += residual_i * residual_i;
    }
    return sqrt(norm);
}

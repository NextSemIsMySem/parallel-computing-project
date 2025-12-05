#ifndef JACOBI_H
#define JACOBI_H

#include <vector>
#include <cmath>
#include <random>
#include <omp.h>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

// Helper struct to hold the linear system
struct LinearSystem {
    Matrix A;
    Vector b;
    Vector x;
    int n;
    
    LinearSystem(int size) : n(size) {
        A.resize(n, Vector(n, 0.0));
        b.resize(n, 0.0);
        x.resize(n, 0.0);
    }
};

// Function declarations
void solveSerial(const Matrix& A, const Vector& b, Vector& x, int maxIter = 10000, double tol = 1e-6);
void solveParallel(const Matrix& A, const Vector& b, Vector& x, int maxIter = 10000, double tol = 1e-6);

// ============================================================================
// SERIAL VERSIONS OF HELPER FUNCTIONS
// ============================================================================

// Generate a strictly diagonally dominant system (SERIAL)
inline void generateSystemSerial(LinearSystem& sys) {
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    
    // Generate random matrix elements
    for (int i = 0; i < sys.n; ++i) {
        double rowSum = 0.0;
        
        // Generate off-diagonal elements
        for (int j = 0; j < sys.n; ++j) {
            if (i != j) {
                sys.A[i][j] = dis(gen);
                rowSum += std::abs(sys.A[i][j]);
            }
        }
        
        // Make diagonal element strictly dominant
        sys.A[i][i] = rowSum + std::abs(dis(gen)) + 1.0;
    }
    
    // Generate random solution vector
    Vector trueSolution(sys.n);
    for (int i = 0; i < sys.n; ++i) {
        trueSolution[i] = dis(gen);
    }
    
    // Compute b = A * trueSolution
    for (int i = 0; i < sys.n; ++i) {
        sys.b[i] = 0.0;
        for (int j = 0; j < sys.n; ++j) {
            sys.b[i] += sys.A[i][j] * trueSolution[j];
        }
    }
    
    // Initialize x to zero
    for (int i = 0; i < sys.n; ++i) {
        sys.x[i] = 0.0;
    }
}

// Helper function to compute L2 norm of residual (SERIAL)
inline double computeResidualNormSerial(const Matrix& A, const Vector& x, const Vector& b) {
    int n = x.size();
    double norm = 0.0;
    
    for (int i = 0; i < n; ++i) {
        double residual = -b[i];
        for (int j = 0; j < n; ++j) {
            residual += A[i][j] * x[j];
        }
        norm += residual * residual;
    }
    
    return std::sqrt(norm);
}

// ============================================================================
// PARALLEL VERSIONS OF HELPER FUNCTIONS (OpenMP)
// ============================================================================

// Generate a strictly diagonally dominant system (PARALLEL)
inline void generateSystemParallel(LinearSystem& sys) {
    // Generate random matrix elements (parallelized)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < sys.n; ++i) {
        // Each thread needs its own RNG to avoid race conditions
        std::mt19937 thread_gen(42 + i);
        std::uniform_real_distribution<> thread_dis(-1.0, 1.0);
        
        double rowSum = 0.0;
        
        // Generate off-diagonal elements
        for (int j = 0; j < sys.n; ++j) {
            if (i != j) {
                sys.A[i][j] = thread_dis(thread_gen);
                rowSum += std::abs(sys.A[i][j]);
            }
        }
        
        // Make diagonal element strictly dominant
        sys.A[i][i] = rowSum + std::abs(thread_dis(thread_gen)) + 1.0;
    }
    
    // Generate random solution vector (parallelized)
    Vector trueSolution(sys.n);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < sys.n; ++i) {
        std::mt19937 thread_gen(42 + sys.n + i);
        std::uniform_real_distribution<> thread_dis(-1.0, 1.0);
        trueSolution[i] = thread_dis(thread_gen);
    }
    
    // Compute b = A * trueSolution (parallelized)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < sys.n; ++i) {
        sys.b[i] = 0.0;
        for (int j = 0; j < sys.n; ++j) {
            sys.b[i] += sys.A[i][j] * trueSolution[j];
        }
    }
    
    // Initialize x to zero (parallelized)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < sys.n; ++i) {
        sys.x[i] = 0.0;
    }
}

// Helper function to compute L2 norm of residual (PARALLEL)
inline double computeResidualNormParallel(const Matrix& A, const Vector& x, const Vector& b) {
    int n = x.size();
    double norm = 0.0;
    
    // Parallelize with reduction to safely accumulate norm
    #pragma omp parallel for reduction(+:norm) schedule(static)
    for (int i = 0; i < n; ++i) {
        double residual = -b[i];
        for (int j = 0; j < n; ++j) {
            residual += A[i][j] * x[j];
        }
        norm += residual * residual;
    }
    
    return std::sqrt(norm);
}

// ============================================================================
// BACKWARD COMPATIBILITY - Default to serial versions
// ============================================================================

inline void generateSystem(LinearSystem& sys) {
    generateSystemSerial(sys);
}

inline double computeResidualNorm(const Matrix& A, const Vector& x, const Vector& b) {
    return computeResidualNormSerial(A, x, b);
}

#endif // JACOBI_H

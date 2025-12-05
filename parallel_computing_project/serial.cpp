#include "jacobi.h"
#include <iostream>

void solveSerial(const Matrix& A, const Vector& b, Vector& x, int maxIter, double tol) {
    int n = x.size();
    
    // Allocate two vectors for double buffering
    Vector x_old = x;  // Current iteration values
    Vector x_new(n);   // Next iteration values
    
    for (int iter = 0; iter < maxIter; ++iter) {
        // Jacobi iteration: x_new[i] = (b[i] - sum(A[i][j] * x_old[j], j != i)) / A[i][i]
        for (int i = 0; i < n; ++i) {
            double sigma = 0.0;
            
            // Sum all off-diagonal terms
            for (int j = 0; j < n; ++j) {
                if (j != i) {
                    sigma += A[i][j] * x_old[j];
                }
            }
            
            // Compute new value
            x_new[i] = (b[i] - sigma) / A[i][i];
        }
        
        // Check convergence every 10 iterations to reduce overhead
        if (iter % 10 == 0) {
            double norm = computeResidualNormSerial(A, x_new, b);
            
            if (norm < tol) {
                // Copy final result to output vector
                x = x_new;
                return;
            }
        }
        
        // Swap vectors (double buffering)
        std::swap(x_old, x_new);
    }
    
    // Copy final result to output vector
    x = x_old;
}

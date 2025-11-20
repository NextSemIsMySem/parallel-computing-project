// parallel_computing_project.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <omp.h>
#include "solver.h"

using namespace std;

// --- Configuration ---
const int N = 1000; // Matrix size (e.g., 1000x1000)
const int MAX_ITERATIONS = 10000;
const double TOLERANCE = 1e-6;
const double OMEGA = 0.66; // Weight for Jacobi method

int main()
{
    cout << "Setting up problem with matrix size N = " << N << endl;

    // --- 1. Problem Setup ---
    // Generate a diagonally dominant matrix A and vector b to ensure convergence.
    // This part is not timed as it's considered setup.
    vector<double> A = generate_diagonally_dominant_matrix(N);
    vector<double> b = generate_vector_b(N);
    int iterations_taken;

    // --- 2. Serial Execution ---
    cout << "\n--- Running Serial Jacobi Solver ---" << endl;
    double start_time_serial = omp_get_wtime();
    
    vector<double> x_serial = jacobi_serial(A, b, N, MAX_ITERATIONS, TOLERANCE, OMEGA, iterations_taken);
    
    double end_time_serial = omp_get_wtime();
    double time_serial = end_time_serial - start_time_serial;

    cout << "Matrix Size (N): " << N << endl;
    cout << "Iterations Taken: " << iterations_taken << endl;
    cout << "Execution Time (Serial): " << time_serial << " seconds" << endl;
    // The final error is printed inside the jacobi_serial function.

    // --- 3. Parallel Execution (OpenMP) ---
    cout << "\n--- Running Parallel (OpenMP) Jacobi Solver ---" << endl;
    double start_time_omp = omp_get_wtime();

    vector<double> x_omp = jacobi_omp(A, b, N, MAX_ITERATIONS, TOLERANCE, OMEGA, iterations_taken);

    double end_time_omp = omp_get_wtime();
    double time_omp = end_time_omp - start_time_omp;

    cout << "Matrix Size (N): " << N << endl;
    cout << "Iterations Taken: " << iterations_taken << endl;
    cout << "Execution Time (OpenMP): " << time_omp << " seconds" << endl;
    // The final error is printed inside the jacobi_omp function.

    // --- 4. Report Speedup ---
    if (time_serial > 0) {
        cout << "\nSpeedup (Serial / OpenMP): " << time_serial / time_omp << "x" << endl;
    }

    return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file

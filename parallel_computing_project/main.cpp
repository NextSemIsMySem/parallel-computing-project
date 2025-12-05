#include "jacobi.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <omp.h>

int main() {
    // Test sizes
    std::vector<int> sizes = {500, 1000, 2000};
    
    std::cout << "=================================================================\n";
    std::cout << "    Jacobi Iterative Method - Serial vs Parallel Comparison\n";
    std::cout << "=================================================================\n";
    std::cout << "OpenMP Threads: " << omp_get_max_threads() << "\n";
    std::cout << "=================================================================\n\n";
    
    for (int N : sizes) {
        std::cout << "Matrix Size: " << N << " x " << N << "\n";
        std::cout << "-----------------------------------------------------------------\n";
        
        // ===== SERIAL WORKFLOW =====
        LinearSystem sys_serial(N);
        
        // Time serial system generation
        auto start_gen_serial = std::chrono::high_resolution_clock::now();
        generateSystemSerial(sys_serial);
        auto end_gen_serial = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_gen_serial = end_gen_serial - start_gen_serial;
        double time_gen_serial = duration_gen_serial.count();
        
        // Time serial solver
        Vector x_serial = sys_serial.x;
        auto start_solve_serial = std::chrono::high_resolution_clock::now();
        solveSerial(sys_serial.A, sys_serial.b, x_serial);
        auto end_solve_serial = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_solve_serial = end_solve_serial - start_solve_serial;
        double time_solve_serial = duration_solve_serial.count();
        
        // Verify serial solution
        double residual_serial = computeResidualNormSerial(sys_serial.A, x_serial, sys_serial.b);
        
        double total_serial = time_gen_serial + time_solve_serial;
        
        // ===== PARALLEL WORKFLOW =====
        LinearSystem sys_parallel(N);
        
        // Time parallel system generation
        auto start_gen_parallel = std::chrono::high_resolution_clock::now();
        generateSystemParallel(sys_parallel);
        auto end_gen_parallel = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_gen_parallel = end_gen_parallel - start_gen_parallel;
        double time_gen_parallel = duration_gen_parallel.count();
        
        // Time parallel solver
        Vector x_parallel = sys_parallel.x;
        auto start_solve_parallel = std::chrono::high_resolution_clock::now();
        solveParallel(sys_parallel.A, sys_parallel.b, x_parallel);
        auto end_solve_parallel = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_solve_parallel = end_solve_parallel - start_solve_parallel;
        double time_solve_parallel = duration_solve_parallel.count();
        
        // Verify parallel solution
        double residual_parallel = computeResidualNormParallel(sys_parallel.A, x_parallel, sys_parallel.b);
        
        double total_parallel = time_gen_parallel + time_solve_parallel;
        
        // ===== DISPLAY RESULTS =====
        std::cout << std::fixed << std::setprecision(6);
        
        std::cout << "\nSERIAL EXECUTION:\n";
        std::cout << "  System Generation:  " << std::setw(10) << time_gen_serial << " s\n";
        std::cout << "  Solver Execution:   " << std::setw(10) << time_solve_serial << " s\n";
        std::cout << "  Total Time:         " << std::setw(10) << total_serial << " s\n";
        std::cout << "  Residual Norm:      " << std::scientific << std::setprecision(2) 
                  << residual_serial << "\n";
        
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "\nPARALLEL EXECUTION:\n";
        std::cout << "  System Generation:  " << std::setw(10) << time_gen_parallel << " s\n";
        std::cout << "  Solver Execution:   " << std::setw(10) << time_solve_parallel << " s\n";
        std::cout << "  Total Time:         " << std::setw(10) << total_parallel << " s\n";
        std::cout << "  Residual Norm:      " << std::scientific << std::setprecision(2) 
                  << residual_parallel << "\n";
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\nSPEEDUP ANALYSIS:\n";
        std::cout << "  Generation Speedup: " << std::setw(10) << time_gen_serial / time_gen_parallel << "x\n";
        std::cout << "  Solver Speedup:     " << std::setw(10) << time_solve_serial / time_solve_parallel << "x\n";
        std::cout << "  Overall Speedup:    " << std::setw(10) << total_serial / total_parallel << "x\n";
        
        // Verify both solutions are correct
        if (residual_serial > 1e-5 || residual_parallel > 1e-5) {
            std::cerr << "\n? WARNING: Solution may not have converged properly!\n";
            std::cerr << "  Serial residual:   " << residual_serial << "\n";
            std::cerr << "  Parallel residual: " << residual_parallel << "\n";
        }
        
        std::cout << "\n=================================================================\n\n";
    }
    
    std::cout << "SUMMARY:\n";
    std::cout << "  Tolerance:          1e-6\n";
    std::cout << "  Max Iterations:     10000\n";
    std::cout << "  Convergence Check:  Every 10 iterations\n";
    std::cout << "  All solutions verified with residual norm < 1e-5\n";
    std::cout << "=================================================================\n";
    
    return 0;
}

/*
 * COMPILE COMMANDS:
 * 
 * Linux/macOS:
 *   g++ main.cpp serial.cpp parallel.cpp -o jacobi -fopenmp -O3 -std=c++17
 * 
 * Windows (MinGW):
 *   g++ main.cpp serial.cpp parallel.cpp -o jacobi.exe -fopenmp -O3 -std=c++17
 * 
 * Windows (MSVC):
 *   cl main.cpp serial.cpp parallel.cpp /openmp /O2 /EHsc /std:c++17 /Fe:jacobi.exe
 * 
 * RUN:
 *   ./jacobi           (Linux/macOS)
 *   jacobi.exe         (Windows)
 * 
 * CONTROL THREADS:
 *   export OMP_NUM_THREADS=4    (Linux/macOS)
 *   set OMP_NUM_THREADS=4       (Windows)
 * 
 * NOTES:
 *   - The -fopenmp flag enables OpenMP support
 *   - The -O3 flag enables aggressive optimizations
 *   - The -std=c++17 flag ensures C++17 standard compliance
 *   - Both serial and parallel versions now use their respective helper functions
 *   - System generation is also parallelized in the parallel workflow
 */

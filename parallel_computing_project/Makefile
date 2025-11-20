# Makefile for Jacobi Solver

# Compiler
CXX = g++

# Compiler flags
# -O3 for optimization
# -Wall for all warnings
# -fopenmp for OpenMP support
CXXFLAGS = -O3 -Wall
OMP_FLAGS = -fopenmp

# Source files
SOURCES = parallel_computing_project.cpp utils.cpp jacobi_serial.cpp jacobi_omp.cpp
HEADERS = solver.h

# Object files for serial version
OBJ_SERIAL = parallel_computing_project.o utils.o jacobi_serial.o

# Object files for OMP version
OBJ_OMP = parallel_computing_project_omp.o utils_omp.o jacobi_omp.o

# Executable names
EXEC_SERIAL = jacobi_serial
EXEC_OMP = jacobi_omp

# Default target
all: serial omp

# Target to build the serial version
serial: $(EXEC_SERIAL)

$(EXEC_SERIAL): $(OBJ_SERIAL)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Target to build the OpenMP version
omp: $(EXEC_OMP)

$(EXEC_OMP): $(OBJ_OMP)
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) -o $@ $^

# Rule to build serial object files
jacobi_serial.o: jacobi_serial.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to build OpenMP object files
# We create separate object files for the OMP version to avoid conflicts
# especially for the main file if it were different.
jacobi_omp.o: jacobi_omp.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) -c $< -o $@

# Common object files
utils.o: utils.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

parallel_computing_project.o: parallel_computing_project.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create separate object files for the OMP build to ensure proper linking
utils_omp.o: utils.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) -c $< -o $@

parallel_computing_project_omp.o: parallel_computing_project.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) -c $< -o $@


# Clean up build files
clean:
	rm -f *.o $(EXEC_SERIAL) $(EXEC_OMP)

# Phony targets
.PHONY: all serial omp clean

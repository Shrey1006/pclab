/*
 * 01_openmp_pi.cpp  -  Pi calculation using OpenMP
 *
 * COMPILE (Linux / macOS / WSL / MinGW):
 *     g++ -fopenmp 01_openmp_pi.cpp -o 01_openmp_pi
 *
 * RUN:
 *     ./01_openmp_pi                      (Linux/macOS/WSL)
 *     01_openmp_pi.exe                    (Windows)
 *
 * Control number of threads (optional):
 *     export OMP_NUM_THREADS=4 && ./01_openmp_pi      (Linux/macOS)
 *     set OMP_NUM_THREADS=4 && 01_openmp_pi.exe       (Windows cmd)
 */
#include <iostream>
#include <omp.h>
using namespace std;

int main() {
    long long num_steps = 100000000;
    double step = 1.0 / num_steps;
    double sum = 0.0;

    double start = omp_get_wtime();

    #pragma omp parallel for reduction(+:sum)
    for (long long i = 0; i < num_steps; i++) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    double pi = sum * step;
    double end = omp_get_wtime();

    cout << "Parallel Pi Calculation using OpenMP\n\n";
    cout << "Pi = " << pi << endl;
    cout << "Execution Time = " << end - start << " seconds\n";
    return 0;
}

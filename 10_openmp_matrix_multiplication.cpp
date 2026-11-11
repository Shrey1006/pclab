/*
 * 10_openmp_matrix_multiplication.cpp  -  Matrix multiplication using OpenMP
 *
 * Computes C = A x B for N x N matrices, serially and in parallel,
 * compares the timings and verifies both results match.
 *
 * COMPILE:
 *     g++ -O2 -fopenmp 10_openmp_matrix_multiplication.cpp -o 10_openmp_matrix_multiplication
 *
 * RUN (optional argument = matrix size N, default 500):
 *     ./10_openmp_matrix_multiplication
 *     ./10_openmp_matrix_multiplication 1000
 *
 * Set threads (optional):  export OMP_NUM_THREADS=4
 */
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <omp.h>
using namespace std;

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 500;

    vector<double> A(N * N), B(N * N), C_serial(N * N, 0.0), C_par(N * N, 0.0);

    for (int i = 0; i < N * N; i++) {
        A[i] = rand() % 10;
        B[i] = rand() % 10;
    }

    // ---------- Serial ----------
    double t1 = omp_get_wtime();
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++)
            for (int j = 0; j < N; j++)
                C_serial[i * N + j] += A[i * N + k] * B[k * N + j];
    double t2 = omp_get_wtime();

    // ---------- Parallel ----------
    double t3 = omp_get_wtime();
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++)
                sum += A[i * N + k] * B[k * N + j];
            C_par[i * N + j] = sum;
        }
    }
    double t4 = omp_get_wtime();

    // ---------- Verify ----------
    bool ok = true;
    for (int i = 0; i < N * N; i++) {
        if (fabs(C_serial[i] - C_par[i]) > 1e-9) { ok = false; break; }
    }

    cout << "Matrix Multiplication using OpenMP (N = " << N << ")\n\n";
    cout << "Threads used   = " << omp_get_max_threads() << endl;
    cout << "Serial time    = " << t2 - t1 << " seconds\n";
    cout << "Parallel time  = " << t4 - t3 << " seconds\n";
    cout << "Speedup        = " << (t2 - t1) / (t4 - t3) << "x\n";
    cout << "Results match  : " << (ok ? "YES" : "NO") << endl;
    return 0;
}

/*
 * 06_mpi_trapezoidal_rule.cpp  -  Trapezoidal rule using MPI, f(x) = x^2
 *
 * Integrates x^2 over [0, 1] (exact answer = 1/3). The subintervals are
 * divided among the processes (remainder spread over the first ranks).
 *
 * COMPILE:
 *     mpic++ 06_mpi_trapezoidal_rule.cpp -o 06_mpi_trapezoidal_rule
 *
 * RUN (4 processes):
 *     mpirun -np 4 ./06_mpi_trapezoidal_rule
 *     mpiexec -n 4 06_mpi_trapezoidal_rule.exe       (Windows, MS-MPI)
 *
 * Fewer cores than processes (OpenMPI):
 *     mpirun --oversubscribe -np 8 ./06_mpi_trapezoidal_rule
 */
#include <mpi.h>
#include <iostream>
#include <cmath>
#include <iomanip>
using namespace std;

double f(double x) {
    return x * x;
}

int main(int argc, char* argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double a = 0.0;
    double b = 1.0;
    int n = 1000000;

    MPI_Bcast(&a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double h = (b - a) / n;
    int local_n = n / size;
    int remainder = n % size;
    int start;

    if (rank < remainder) {
        local_n++;
        start = rank * local_n;
    } else {
        start = rank * (n / size) + remainder;
    }
    int end = start + local_n;

    double local_sum = 0.0;
    for (int i = start; i < end; i++) {
        double x1 = a + i * h;
        double x2 = a + (i + 1) * h;
        local_sum += (f(x1) + f(x2)) * h / 2.0;
    }

    double total_sum = 0.0;
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << fixed << setprecision(10);
        cout << "Number of processes: " << size << endl;
        cout << "Number of subintervals: " << n << endl;
        cout << "Approximate integral: " << total_sum << endl;
        cout << "Exact integral: " << 1.0 / 3.0 << endl;
        cout << "Error: " << fabs(total_sum - 1.0 / 3.0) << endl;
    }

    MPI_Finalize();
    return 0;
}

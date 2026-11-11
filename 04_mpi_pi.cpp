/*
 * 04_mpi_pi.cpp  -  Pi calculation using MPI
 *
 * COMPILE:
 *     mpic++ 04_mpi_pi.cpp -o 04_mpi_pi
 *
 * RUN (4 processes):
 *     mpirun -np 4 ./04_mpi_pi                  (OpenMPI / MPICH, Linux/macOS/WSL)
 *     mpiexec -n 4 04_mpi_pi.exe                (Windows, MS-MPI)
 *
 * If you have fewer cores than processes (OpenMPI):
 *     mpirun --oversubscribe -np 8 ./04_mpi_pi
 */
#include <mpi.h>
#include <iostream>
using namespace std;

int main(int argc, char** argv) {
    int rank, size;
    long long num_steps = 1000000;
    double step, x, sum = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    step = 1.0 / (double)num_steps;

    for (long long i = rank; i < num_steps; i += size) {
        x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    double partial_sum = sum * step;
    double total_pi = 0.0;

    MPI_Reduce(&partial_sum, &total_pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "Approximate value of Pi = " << total_pi << endl;
    }

    MPI_Finalize();
    return 0;
}

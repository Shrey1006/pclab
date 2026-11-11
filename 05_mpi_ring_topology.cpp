/*
 * 05_mpi_ring_topology.cpp  -  Ring topology using MPI
 *
 * A token starts at process 0 and travels 0 -> 1 -> 2 -> ... -> 0.
 * Every process adds its rank to the token before passing it on.
 *
 * COMPILE:
 *     mpic++ 05_mpi_ring_topology.cpp -o 05_mpi_ring_topology
 *
 * RUN (needs at least 2 processes; 4 shown):
 *     mpirun -np 4 ./05_mpi_ring_topology
 *     mpiexec -n 4 05_mpi_ring_topology.exe          (Windows, MS-MPI)
 *
 * Fewer cores than processes (OpenMPI):
 *     mpirun --oversubscribe -np 8 ./05_mpi_ring_topology
 */
#include <mpi.h>
#include <iostream>
using namespace std;

int main(int argc, char **argv)
{
    int rank, size;
    int token;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // A ring needs at least 2 processes (a process cannot Send to itself and Recv)
    if (size < 2)
    {
        cout << "Run with at least 2 processes, e.g. mpirun -np 4 ./05_mpi_ring_topology" << endl;
        MPI_Finalize();
        return 0;
    }

    int next = (rank + 1) % size;
    int prev = (rank - 1 + size) % size;

    if (rank == 0)
    {
        token = 100;
        cout << "Process 0 starting ring with token = " << token << endl;
        MPI_Send(&token, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
        MPI_Recv(&token, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, &status);
        cout << "Process 0 received final token = " << token << " after full ring." << endl;
    }
    else
    {
        MPI_Recv(&token, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, &status);
        cout << "Process " << rank << " received token = " << token
             << " from process " << prev << endl;
        token += rank;
        MPI_Send(&token, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}

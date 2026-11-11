/*
 * 07_mpi_odd_even_sort.cpp  -  Odd-Even Sort using MPI
 *
 * Each process holds ONE element. In every phase, neighbouring processes
 * exchange values (even phase: pairs (0,1),(2,3)..; odd phase: (1,2),(3,4)..)
 * and the lower rank keeps the smaller value, the higher rank the larger.
 * After 'total_processes' phases the elements are sorted by rank.
 *
 * COMPILE:
 *     mpic++ 07_mpi_odd_even_sort.cpp -o 07_mpi_odd_even_sort
 *
 * RUN (8 processes = sorts 8 elements):
 *     mpirun -np 8 ./07_mpi_odd_even_sort
 *     mpiexec -n 8 07_mpi_odd_even_sort.exe          (Windows, MS-MPI)
 *
 * Fewer cores than processes (OpenMPI):
 *     mpirun --oversubscribe -np 8 ./07_mpi_odd_even_sort
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int process_id, total_processes;
    int local_value, neighbor_rank, received_value;
    int *input_array = NULL;
    int *output_array = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &total_processes);

    if (process_id == 0)
    {
        input_array = (int *)malloc(total_processes * sizeof(int));
        output_array = (int *)malloc(total_processes * sizeof(int));

        printf("Input elements: ");
        for (int index = 0; index < total_processes; index++)
        {
            input_array[index] = (total_processes * 7 - index * 3) % 31;
            printf("%d ", input_array[index]);
        }
        printf("\n");
    }

    MPI_Scatter(input_array, 1, MPI_INT,
                &local_value, 1, MPI_INT,
                0, MPI_COMM_WORLD);

    for (int phase = 0; phase < total_processes; phase++)
    {
        if (phase % 2 == 0)
        {
            neighbor_rank = (process_id % 2 == 0) ? process_id + 1 : process_id - 1;
        }
        else
        {
            neighbor_rank = (process_id % 2 == 0) ? process_id - 1 : process_id + 1;
        }

        if (neighbor_rank >= 0 && neighbor_rank < total_processes)
        {
            MPI_Sendrecv(&local_value, 1, MPI_INT, neighbor_rank, 0,
                         &received_value, 1, MPI_INT, neighbor_rank, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (process_id < neighbor_rank)
            {
                if (local_value > received_value)
                    local_value = received_value;
            }
            else
            {
                if (local_value < received_value)
                    local_value = received_value;
            }
        }
    }

    MPI_Gather(&local_value, 1, MPI_INT,
               output_array, 1, MPI_INT,
               0, MPI_COMM_WORLD);

    if (process_id == 0)
    {
        printf("Sorted elements: ");
        for (int index = 0; index < total_processes; index++)
            printf("%d ", output_array[index]);
        printf("\n");

        free(input_array);
        free(output_array);
    }

    MPI_Finalize();
    return 0;
}

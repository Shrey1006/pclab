/*
 * 08_mpi_hyperquicksort.cpp  -  Hyperquicksort using MPI
 *
 * 1,000,000 random numbers are scattered, each process sorts its block, then
 * log2(p) split-and-exchange steps are done inside shrinking sub-hypercubes
 * (pivot = median of the sub-cube root's block).
 * The result is verified against a sequential std::sort.
 *
 * NOTE: number of processes must be a power of 2 (1, 2, 4, 8, ...) and
 *       must divide N = 1,000,000 (so 1, 2, 4, 8 work).
 *
 * COMPILE:
 *     mpic++ -O2 08_mpi_hyperquicksort.cpp -o 08_mpi_hyperquicksort
 *
 * RUN (4 processes):
 *     mpirun -np 4 ./08_mpi_hyperquicksort
 *     mpiexec -n 4 08_mpi_hyperquicksort.exe         (Windows, MS-MPI)
 *
 * Fewer cores than processes (OpenMPI):
 *     mpirun --oversubscribe -np 8 ./08_mpi_hyperquicksort
 */
#include <mpi.h>
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>
using namespace std;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    const int N = 1000000; // 10,00,000 elements

    // Safety check: p must be a power of 2 and divide N
    if ((p & (p - 1)) != 0 || N % p != 0)
    {
        if (rank == 0)
            printf("Error: number of processes must be a power of 2 that divides %d (try 1, 2, 4, 8).\n", N);
        MPI_Finalize();
        return 1;
    }

    const int local_n = N / p;
    vector<int> data;

    // Step 1: root generates N random integers
    if (rank == 0)
    {
        data.resize(N);
        mt19937 gen(12345);
        uniform_int_distribution<int> dist(0, 9999999);
        for (int i = 0; i < N; i++)
            data[i] = dist(gen);
        printf("Generated %d random numbers\n", N);
        printf("First 10 (unsorted): ");
        for (int i = 0; i < 10; i++)
            printf("%d ", data[i]);
        printf("\n");
    }

    // Step 2: divide the data equally among the processes
    vector<int> local(local_n);
    MPI_Scatter(data.data(), local_n, MPI_INT, local.data(), local_n, MPI_INT, 0,
                MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Step 3a: each process sorts its own list with quicksort
    sort(local.begin(), local.end());

    // Step 3b: log p split-and-merge steps
    MPI_Comm comm = MPI_COMM_WORLD;
    int size = p;
    while (size > 1)
    {
        int grank;
        MPI_Comm_rank(comm, &grank);

        // Root of the (sub)hypercube broadcasts the median of its sorted list as pivot
        int pivot = 0;
        if (grank == 0 && !local.empty())
            pivot = local[local.size() / 2];
        MPI_Bcast(&pivot, 1, MPI_INT, 0, comm);

        // Split local list: low = (<= pivot), high = (> pivot)
        int cut = upper_bound(local.begin(), local.end(), pivot) - local.begin();

        int half = size / 2;
        int partner = (grank < half) ? grank + half : grank - half;

        // Lower half keeps low part and sends high part; upper half does the opposite
        int send_cnt, keep_cnt;
        const int *send_ptr, *keep_ptr;
        if (grank < half)
        {
            send_ptr = local.data() + cut;
            send_cnt = (int)local.size() - cut;
            keep_ptr = local.data();
            keep_cnt = cut;
        }
        else
        {
            send_ptr = local.data();
            send_cnt = cut;
            keep_ptr = local.data() + cut;
            keep_cnt = (int)local.size() - cut;
        }

        int recv_cnt;
        MPI_Sendrecv(&send_cnt, 1, MPI_INT, partner, 0, &recv_cnt, 1, MPI_INT, partner, 0,
                     comm, MPI_STATUS_IGNORE);

        vector<int> recv(recv_cnt);
        MPI_Sendrecv(send_ptr, send_cnt, MPI_INT, partner, 1, recv.data(), recv_cnt, MPI_INT,
                     partner, 1, comm, MPI_STATUS_IGNORE);

        // Merge kept list with received list (both sorted)
        vector<int> merged(keep_cnt + recv_cnt);
        merge(keep_ptr, keep_ptr + keep_cnt, recv.begin(), recv.end(), merged.begin());
        local.swap(merged);

        // Move to the sub-hypercube (lower half / upper half)
        MPI_Comm newcomm;
        MPI_Comm_split(comm, grank / half, grank, &newcomm);
        if (comm != MPI_COMM_WORLD)
            MPI_Comm_free(&comm);
        comm = newcomm;
        size = half;
    }
    if (comm != MPI_COMM_WORLD)
        MPI_Comm_free(&comm);

    double t1 = MPI_Wtime();
    double elapsed = t1 - t0, maxtime;
    MPI_Reduce(&elapsed, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Report per-process results (in rank order)
    int cnt = (int)local.size();
    for (int r = 0; r < p; r++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (r == rank)
            printf("Process %d: %d elements, min = %d, max = %d\n", rank, cnt,
                   local.empty() ? -1 : local.front(),
                   local.empty() ? -1 : local.back());
        fflush(stdout);
    }

    // Gather the sorted blocks on root for verification
    vector<int> counts(p), displs(p, 0);
    MPI_Gather(&cnt, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0)
        for (int i = 1; i < p; i++)
            displs[i] = displs[i - 1] + counts[i - 1];

    vector<int> result(rank == 0 ? N : 0);
    MPI_Gatherv(local.data(), cnt, MPI_INT, result.data(), counts.data(), displs.data(),
                MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        bool ok = is_sorted(result.begin(), result.end());
        printf("First 10 (sorted): ");
        for (int i = 0; i < 10; i++)
            printf("%d ", result[i]);
        printf("\nLast 10 (sorted): ");
        for (int i = N - 10; i < N; i++)
            printf("%d ", result[i]);
        printf("\nSorted correctly: %s\n", ok ? "YES" : "NO");
        printf("Parallel time (p=%d): %.4f s\n", p, maxtime);

        vector<int> ref(data);
        double s0 = MPI_Wtime();
        sort(ref.begin(), ref.end());
        printf("Sequential std::sort time: %.4f s\n", MPI_Wtime() - s0);
        printf("Matches sequential result: %s\n", ref == result ? "YES" : "NO");
    }

    MPI_Finalize();
    return 0;
}

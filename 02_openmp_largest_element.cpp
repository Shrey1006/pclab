/*
 * 02_openmp_largest_element.cpp  -  Largest element of an array using OpenMP
 *
 * COMPILE:
 *     g++ -fopenmp 02_openmp_largest_element.cpp -o 02_openmp_largest_element
 *
 * RUN:
 *     ./02_openmp_largest_element
 *
 * Set threads (optional):  export OMP_NUM_THREADS=4
 */
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <omp.h>
using namespace std;

int main() {
    const int N = 10000000;
    int *arr = new int[N];

    srand(time(0));
    for (int i = 0; i < N; i++) {
        arr[i] = rand() % 1000000;
    }

    int largest = arr[0];
    double start = omp_get_wtime();

    #pragma omp parallel for reduction(max:largest)
    for (int i = 0; i < N; i++) {
        if (arr[i] > largest) {
            largest = arr[i];
        }
    }

    double end = omp_get_wtime();

    cout << "Parallel Execution using OpenMP\n\n";
    cout << "Array Size = " << N << endl;
    cout << "Largest Element = " << largest << endl;
    cout << "Execution Time = " << end - start << " seconds\n";

    delete[] arr;
    return 0;
}

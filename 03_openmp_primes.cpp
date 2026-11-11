/*
 * 03_openmp_primes.cpp  -  Prime numbers between 2 and a given number (OpenMP)
 *
 * COMPILE:
 *     g++ -fopenmp 03_openmp_primes.cpp -o 03_openmp_primes
 *
 * RUN:
 *     ./03_openmp_primes
 *     (enter the limit when asked, e.g. 1000000)
 *
 * Set threads (optional):  export OMP_NUM_THREADS=4
 */
#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

bool isPrime(int n) {
    if (n < 2)
        return false;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0)
            return false;
    }
    return true;
}

int main() {
    int limit;
    cout << "Enter the limit: ";
    cin >> limit;

    vector<int> primes;
    double start = omp_get_wtime();

    #pragma omp parallel
    {
        vector<int> localPrimes;

        #pragma omp for
        for (int i = 2; i <= limit; i++) {
            if (isPrime(i)) {
                localPrimes.push_back(i);
            }
        }

        #pragma omp critical
        {
            primes.insert(primes.end(), localPrimes.begin(), localPrimes.end());
        }
    }

    double end = omp_get_wtime();

    cout << "\nTotal Primes = " << primes.size();
    cout << "\nExecution Time = " << end - start << " seconds\n";
    return 0;
}

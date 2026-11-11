/*
 * 09_openmp_producer_consumer.cpp  -  Producer-Consumer problem using OpenMP
 *
 * A bounded circular buffer is shared by 2 producer threads and 2 consumer
 * threads (4 threads in total). Access to the buffer is protected with an
 * OpenMP named critical section. A producer waits when the buffer is full,
 * a consumer waits when it is empty.
 *
 * COMPILE:
 *     g++ -fopenmp 09_openmp_producer_consumer.cpp -o 09_openmp_producer_consumer
 *
 * RUN:
 *     ./09_openmp_producer_consumer
 */
#include <iostream>
#include <omp.h>
using namespace std;

const int BUFFER_SIZE = 5;
const int TOTAL_ITEMS = 20;      // total items to produce (and consume)
const int NUM_PRODUCERS = 2;

int buffer[BUFFER_SIZE];
int in_pos = 0, out_pos = 0, count_items = 0;
int produced_total = 0, consumed_total = 0;

int main() {
    omp_set_num_threads(4);      // threads 0,1 = producers ; threads 2,3 = consumers

    #pragma omp parallel
    {
        int id = omp_get_thread_num();

        if (id < NUM_PRODUCERS) {
            // ---------------- PRODUCER ----------------
            bool finished = false;
            while (!finished) {
                #pragma omp critical(buffer_access)
                {
                    if (produced_total >= TOTAL_ITEMS) {
                        finished = true;
                    } else if (count_items < BUFFER_SIZE) {
                        int item = produced_total++;
                        buffer[in_pos] = item;
                        in_pos = (in_pos + 1) % BUFFER_SIZE;
                        count_items++;
                        cout << "Producer " << id << " produced item " << item
                             << "  (buffer count = " << count_items << ")\n";
                    }
                    // else buffer is full -> retry
                }
            }
        } else {
            // ---------------- CONSUMER ----------------
            bool finished = false;
            while (!finished) {
                #pragma omp critical(buffer_access)
                {
                    if (consumed_total >= TOTAL_ITEMS) {
                        finished = true;
                    } else if (count_items > 0) {
                        int item = buffer[out_pos];
                        out_pos = (out_pos + 1) % BUFFER_SIZE;
                        count_items--;
                        consumed_total++;
                        cout << "Consumer " << id << " consumed item " << item
                             << "  (buffer count = " << count_items << ")\n";
                    }
                    // else buffer is empty -> retry
                }
            }
        }
    }

    cout << "\nDone. Produced = " << produced_total
         << ", Consumed = " << consumed_total << endl;
    return 0;
}

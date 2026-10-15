#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    setbuf(stdout, NULL);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) {
        if (rank == 0) fprintf(stderr, "Need at least 2 MPI ranks\n");
        MPI_Finalize();
        return 2;
    }

    const long long elements = 32LL * 1024 * 1024 / sizeof(int);
    const int warmup = 10;
    const int iterations = 200;
    int *send = malloc(elements * sizeof(int));
    int *recv = malloc(elements * sizeof(int));
    if (!send || !recv) {
        fprintf(stderr, "Allocation failed on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    const int expected_per_element = size * (size + 1) / 2;
    double start = MPI_Wtime();
    for (int iteration = 0; iteration < warmup + iterations; ++iteration) {
        for (long long index = 0; index < elements; ++index) {
            send[index] = rank + 1;
        }
        MPI_Allreduce(send, recv, elements, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (iteration >= warmup) {
            for (long long index = 0; index < elements; ++index) {
                if (recv[index] != expected_per_element) {
                    fprintf(stderr, "Validation failed on rank %d\n", rank);
                    MPI_Abort(MPI_COMM_WORLD, 4);
                }
            }
        }
    }
    double elapsed = MPI_Wtime() - start;

    if (rank == 0) {
        printf("m1_allreduce size=%d elements=%lld iterations=%d warmup=%d elapsed_seconds=%.6f validation=passed\n",
               size, elements, iterations, warmup, elapsed);
    }

    free(send);
    free(recv);
    MPI_Finalize();
    return 0;
}

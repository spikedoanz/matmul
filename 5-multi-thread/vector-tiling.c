#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <sys/time.h>
#include <immintrin.h>

#define MAX_THREADS 24
#define TESTS 5
#define TILE_SIZE 24

typedef struct {
    float *A;
    float *B;
    float *C;
    int M, N, K;
    int start_row;
    int end_row;
} ThreadArgs;

void *matmul_thread(void *arg) {
    ThreadArgs *args = (ThreadArgs *)arg;
    float *A = args->A;
    float *B = args->B;
    float *C = args->C;
    int M = args->M, N = args->N, K = args->K;
    int start_row = args->start_row;
    int end_row = args->end_row;

    for (int i = start_row; i < end_row; i += TILE_SIZE) {
        for (int j = 0; j < N; j += TILE_SIZE) {
            for (int k = 0; k < K; k += TILE_SIZE) {
                for (int ii = i; ii < i + TILE_SIZE && ii < end_row; ii++) {
                    for (int jj = j; jj < j + TILE_SIZE && jj < N; jj += 8) {
                        __m256 sum = _mm256_setzero_ps();
                        for (int kk = k; kk < k + TILE_SIZE && kk < K; kk++) {
                            __m256 a = _mm256_set1_ps(A[ii * K + kk]);
                            __m256 b = _mm256_loadu_ps(&B[kk * N + jj]);
                            sum = _mm256_fmadd_ps(a, b, sum);
                        }
                        __m256 c = _mm256_loadu_ps(&C[ii * N + jj]);
                        c = _mm256_add_ps(c, sum);
                        _mm256_storeu_ps(&C[ii * N + jj], c);
                    }
                }
            }
        }
    }
    return NULL;
}

void matmul(float *A, float *B, float *C, int M, int N, int K, int num_threads) {
    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    ThreadArgs *thread_args = malloc(num_threads * sizeof(ThreadArgs));
    int rows_per_thread = M / num_threads;
    int extra_rows = M % num_threads;

    for (int i = 0; i < M * N; i++) {
        C[i] = 0.0f;
    }

    for (int i = 0; i < num_threads; i++) {
        thread_args[i].A = A;
        thread_args[i].B = B;
        thread_args[i].C = C;
        thread_args[i].M = M;
        thread_args[i].N = N;
        thread_args[i].K = K;
        thread_args[i].start_row = i * rows_per_thread + (i < extra_rows ? i : extra_rows);
        thread_args[i].end_row = (i + 1) * rows_per_thread + (i < extra_rows ? i + 1 : extra_rows);
        pthread_create(&threads[i], NULL, matmul_thread, &thread_args[i]);
    }
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    free(threads);
    free(thread_args);
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    int sizes[][3] = {{128, 128, 128}, {512, 512, 512}, {1024, 1024, 1024}};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    srand(time(NULL));
    printf("threads,m,n,k,time,gflops\n");

    for (int num_threads = 24; num_threads <= MAX_THREADS; num_threads += 4) {
        for (int i = 0; i < num_sizes; i++) {
            int M = sizes[i][0];
            int N = sizes[i][1];
            int K = sizes[i][2];

            float *A = aligned_alloc(32, M * K * sizeof(float));
            float *B = aligned_alloc(32, K * N * sizeof(float));
            float *C = aligned_alloc(32, M * N * sizeof(float));

            if (!A || !B || !C) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }

            for (int j = 0; j < M * K; j++) {
                A[j] = (float)rand() / RAND_MAX;
            }
            for (int j = 0; j < K * N; j++) {
                B[j] = (float)rand() / RAND_MAX;
            }

            double total_time = 0.0;
            double total_gflops = 0.0;

            for (int test = 0; test < TESTS; test++) {
                double start_time = get_time();
                matmul(A, B, C, M, N, K, num_threads);
                double end_time = get_time();

                double elapsed_time = end_time - start_time;
                double flops = 2.0 * M * N * K;
                double gflops = flops / (elapsed_time * 1e9);

                total_time += elapsed_time;
                total_gflops += gflops;
            }

            double avg_time = total_time / TESTS;
            double avg_gflops = total_gflops / TESTS;

            printf("%d,%d,%d,%d,%.6f,%.2f\n", num_threads, M, N, K, avg_time, avg_gflops);

            free(A);
            free(B);
            free(C);
        }
    }

    return 0;
}

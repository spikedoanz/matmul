/*
Tricks used

- tiling
- vectorized fused multiply adds
- prefetching
- microkenrels for loop unrolling
- data packing cache friendliness
- tuned precisely for 5900X

Perf: 625 GFLOPS
- hot zones are still on adds, so will need to be unrolled more
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <sys/time.h>
#include <immintrin.h>
#include <string.h>

#define MAX_THREADS 24
#define CACHE_LINE_SIZE 64
#define L1_CACHE_SIZE (32 * 1024)
#define L2_CACHE_SIZE (512 * 1024)
#define L3_CACHE_SIZE (32 * 1024 * 1024)

// Micro-kernel size
#define MR 8
#define NR 8

// Packing buffer size
#define MC 128
#define KC 256
#define NC 4096

// Align to cache line size
#define ALIGN __attribute__((aligned(CACHE_LINE_SIZE)))

typedef struct {
    float *A;
    float *B;
    float *C;
    int M, N, K;
    int start_row;
    int end_row;
} ThreadArgs;

// Portable way to force inline
#define FORCE_INLINE __attribute__((always_inline)) inline

// Packed buffers
static float ALIGN Ac[MC * KC];
static float ALIGN Bc[KC * NC];

// Function to pack A
FORCE_INLINE void pack_a(int K, const float *A, int lda, float *A_to) {
    for (int i = 0; i < MR; ++i) {
        for (int j = 0; j < K; ++j) {
            A_to[i * K + j] = A[i * lda + j];
        }
    }
}

// Function to pack B
FORCE_INLINE void pack_b(int K, int N, const float *B, int ldb, float *B_to) {
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < NR; ++j) {
            B_to[i * NR + j] = B[i * ldb + j];
        }
    }
}

// Micro-kernel
FORCE_INLINE void micro_kernel(int K, const float *A, const float *B, float *C, int ldc) {
    __m256 c[MR];
    for (int i = 0; i < MR; ++i) {
        c[i] = _mm256_setzero_ps();
    }

    for (int k = 0; k < K; ++k) {
        __m256 b = _mm256_load_ps(&B[k * NR]);
        for (int i = 0; i < MR; ++i) {
            __m256 a = _mm256_set1_ps(A[i * K + k]);
            c[i] = _mm256_fmadd_ps(a, b, c[i]);
        }
    }

    for (int i = 0; i < MR; ++i) {
        _mm256_store_ps(&C[i * ldc], c[i]);
    }
}

// Function to handle edge cases
FORCE_INLINE void edge_case_micro_kernel(int M, int N, int K, const float *A, const float *B, float *C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * ldc + j] += sum;
        }
    }
}

// Main computation kernel
void compute_kernel(int M, int N, int K, const float *A, const float *B, float *C, int ldc) {
    int mb = (M + MR - 1) / MR;
    int nb = (N + NR - 1) / NR;

    for (int i = 0; i < mb; ++i) {
        int m = (i != mb - 1 || M % MR == 0) ? MR : M % MR;

        for (int j = 0; j < nb; ++j) {
            int n = (j != nb - 1 || N % NR == 0) ? NR : N % NR;

            if (m == MR && n == NR) {
                micro_kernel(K, &A[i * MR * K], &B[j * NR], &C[i * MR * ldc + j * NR], ldc);
            } else {
                edge_case_micro_kernel(m, n, K, &A[i * MR * K], &B[j * NR], &C[i * MR * ldc + j * NR], ldc);
            }
        }
    }
}

void *matmul_thread(void *arg) {
    ThreadArgs *args = (ThreadArgs *)arg;
    float *A = args->A;
    float *B = args->B;
    float *C = args->C;
    int M = args->M, N = args->N, K = args->K;
    int start_row = args->start_row;
    int end_row = args->end_row;

    for (int i = start_row; i < end_row; i += MC) {
        int mb = (i + MC <= end_row) ? MC : end_row - i;

        for (int k = 0; k < K; k += KC) {
            int kb = (k + KC <= K) ? KC : K - k;

            // Pack A
            pack_a(kb, &A[i * K + k], K, Ac);

            for (int j = 0; j < N; j += NC) {
                int nb = (j + NC <= N) ? NC : N - j;

                // Pack B
                pack_b(kb, nb, &B[k * N + j], N, Bc);

                // Compute
                compute_kernel(mb, nb, kb, Ac, Bc, &C[i * N + j], N);
            }
        }
    }

    return NULL;
}

void matmul(float *A, float *B, float *C, int M, int N, int K, int num_threads) {
    pthread_t threads[MAX_THREADS];
    ThreadArgs thread_args[MAX_THREADS];

    for (int i = 0; i < num_threads; i++) {
        thread_args[i].A = A;
        thread_args[i].B = B;
        thread_args[i].C = C;
        thread_args[i].M = M;
        thread_args[i].N = N;
        thread_args[i].K = K;
        thread_args[i].start_row = (M * i) / num_threads;
        thread_args[i].end_row = (M * (i + 1)) / num_threads;

        if (pthread_create(&threads[i], NULL, matmul_thread, &thread_args[i]) != 0) {
            fprintf(stderr, "Failed to create thread %d\n", i);
            exit(1);
        }
    }

    for (int i = 0; i < num_threads; i++) {
        if (pthread_join(threads[i], NULL) != 0) {
            fprintf(stderr, "Failed to join thread %d\n", i);
            exit(1);
        }
    }
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    int M = 4096, N = 4096, K = 4096;
    //int M = 512, N = 512, K = 512;
    float *A = (float *)aligned_alloc(32, M * K * sizeof(float));
    float *B = (float *)aligned_alloc(32, K * N * sizeof(float));
    float *C = (float *)aligned_alloc(32, M * N * sizeof(float));
    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize matrices with random values
    srand(time(NULL));
    for (int i = 0; i < M * K; i++) A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) B[i] = (float)rand() / RAND_MAX;
    memset(C, 0, M * N * sizeof(float));

    int num_threads = 24; // Adjust based on your CPU

    double start_time = get_time();
    matmul(A, B, C, M, N, K, num_threads);
    double end_time = get_time();

    double elapsed_time = end_time - start_time;
    double flops = 2.0 * M * N * K;
    double gflops = flops / (elapsed_time * 1e9);

    printf("Time: %.6f seconds\n", elapsed_time);
    printf("Performance: %.2f GFLOPS\n", gflops);

    free(A);
    free(B);
    free(C);
    return 0;
}

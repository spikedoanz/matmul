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

// Macro to generate a single FMA operation
#define FMA_OP(i, j, k) \
    sum##i##j = _mm256_fmadd_ps(_mm256_set1_ps(A[i * K + k]), _mm256_loadu_ps(&B[k * N + j]), sum##i##j);

// Macro to generate unrolled FMA operations for a single tile
#define GENERATE_TILE(ii, jj) \
    { \
        __m256 sum##ii##jj = _mm256_setzero_ps(); \
        for (int kk = k; kk < k + TILE_SIZE && kk < K; kk++) { \
            FMA_OP(ii, jj, kk) \
        } \
        _mm256_storeu_ps(&C[ii * N + jj], _mm256_add_ps(_mm256_loadu_ps(&C[ii * N + jj]), sum##ii##jj)); \
    }

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
                    for (int jj = j; jj < j + TILE_SIZE && jj < N; jj += 32) {
                        GENERATE_TILE(ii, jj)
                        if (jj + 8 < N) { GENERATE_TILE(ii, jj + 8) }
                        if (jj + 16 < N) { GENERATE_TILE(ii, jj + 16) }
                        if (jj + 24 < N) { GENERATE_TILE(ii, jj + 24) }
                    }
                }
            }
        }
    }
    return NULL;
}

// The rest of the code (matmul function, get_time function, and main function) remains unchanged
// ... (include the rest of your original code here)

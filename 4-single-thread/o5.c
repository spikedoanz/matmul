// vectorized + unrolling + prefetch + tiling + transposing b
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>

void transpose(float *src, float *dst, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j += 8) {
            __m256 row = _mm256_loadu_ps(&src[i * cols + j]);
            for (int k = 0; k < 8 && j + k < cols; k++) {
                _mm_store_ss(&dst[(j + k) * rows + i], _mm256_extractf128_ps(row, k / 4));
            }
        }
    }
}

void matmul(float *A, float *B_transposed, float *C, int M, int N, int K, int TILE_SIZE) {
    for (int i = 0; i < M; i += TILE_SIZE) {
        for (int j = 0; j < N; j += TILE_SIZE) {
            for (int k = 0; k < K; k += TILE_SIZE) {
                for (int ii = i; ii < i + TILE_SIZE && ii < M; ii++) {
                    for (int jj = j; jj < j + TILE_SIZE && jj < N; jj += 8) {
                        // vectorized
                        __m256 acc = _mm256_setzero_ps();
                        for (int kk = k; kk < k + TILE_SIZE && kk < K; kk++) {
                            __m256 a = _mm256_broadcast_ss(&A[ii * K + kk]);
                            __m256 b = _mm256_loadu_ps(&B_transposed[jj * K + kk]);
                            acc = _mm256_fmadd_ps(a, b, acc);
                        }
                        __m256 c = _mm256_loadu_ps(&C[ii * N + jj]);
                        c = _mm256_add_ps(c, acc);
                        _mm256_storeu_ps(&C[ii * N + jj], c);
                    }
                }
            }
        }
    }
}

int main() {
    int sizes[][3] = {{128, 128, 128}, {512, 512, 512}, {1024, 1024, 1024}};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int tile_sizes[] = {8, 16, 32, 64};
    int num_tile_sizes = sizeof(tile_sizes) / sizeof(tile_sizes[0]);

    srand(time(NULL));

    printf("m,n,k,tile_size,time,flops\n");

    for (int i = 0; i < num_sizes; i++) {
        int M = sizes[i][0];
        int N = sizes[i][1];
        int K = sizes[i][2];

        float *A = (float *)aligned_alloc(32, M * K * sizeof(float));
        float *B = (float *)aligned_alloc(32, K * N * sizeof(float));
        float *B_transposed = (float *)aligned_alloc(32, K * N * sizeof(float));
        float *C = (float *)aligned_alloc(32, M * N * sizeof(float));

        for (int j = 0; j < M * K; j++) {
            A[j] = (float)rand() / RAND_MAX;
        }
        for (int j = 0; j < K * N; j++) {
            B[j] = (float)rand() / RAND_MAX;
        }

        transpose(B, B_transposed, K, N);

        for (int t = 0; t < num_tile_sizes; t++) {
            int TILE_SIZE = tile_sizes[t];

            for (int j = 0; j < M * N; j++) {
                C[j] = 0.0f;
            }

            clock_t start_time = clock();

            matmul(A, B_transposed, C, M, N, K, TILE_SIZE);

            clock_t end_time = clock();

            double iteration_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
            double flops = 2.0 * M * N * K;
            double flops_per_second = flops / iteration_time / 1e9;

            printf("%d,%d,%d,%d,%.6f,%.2f\n", M, N, K, TILE_SIZE, iteration_time, flops_per_second);
        }

        free(A);
        free(B);
        free(B_transposed);
        free(C);
    }

    return 0;
}

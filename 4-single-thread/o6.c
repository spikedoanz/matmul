// tuned tile size + vectorized + unrolling + prefetch + tiling + transposing b
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>

#define BLOCK_SIZE 24 // tuned for 5900X's 32KB of L1 Cache

void transpose(float *src, float *dst, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j += 32) {
            __m256 row1 = _mm256_loadu_ps(&src[i * cols + j]);
            __m256 row2 = _mm256_loadu_ps(&src[i * cols + j + 8]);
            __m256 row3 = _mm256_loadu_ps(&src[i * cols + j + 16]);
            __m256 row4 = _mm256_loadu_ps(&src[i * cols + j + 24]);

            for (int k = 0; k < 8 && j + k < cols; k++) {
                _mm_store_ss(&dst[(j + k) * rows + i], _mm256_extractf128_ps(row1, k / 4));
                _mm_store_ss(&dst[(j + k + 8) * rows + i], _mm256_extractf128_ps(row2, k / 4));
                _mm_store_ss(&dst[(j + k + 16) * rows + i], _mm256_extractf128_ps(row3, k / 4));
                _mm_store_ss(&dst[(j + k + 24) * rows + i], _mm256_extractf128_ps(row4, k / 4));
            }
        }
    }
}

void matmul_blocked(float *A, float *B_transposed, float *C, int M, int N, int K) {
    for (int i0 = 0; i0 < M; i0 += BLOCK_SIZE) {
        for (int j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
            for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
                int max_i = (i0 + BLOCK_SIZE < M) ? i0 + BLOCK_SIZE : M;
                int max_j = (j0 + BLOCK_SIZE < N) ? j0 + BLOCK_SIZE : N;
                int max_k = (k0 + BLOCK_SIZE < K) ? k0 + BLOCK_SIZE : K;

                for (int i = i0; i < max_i - 3; i += 4) {
                    for (int k = k0; k < max_k; k++) {
                        __m256 a0 = _mm256_broadcast_ss(&A[i * K + k]);
                        __m256 a1 = _mm256_broadcast_ss(&A[(i+1) * K + k]);
                        __m256 a2 = _mm256_broadcast_ss(&A[(i+2) * K + k]);
                        __m256 a3 = _mm256_broadcast_ss(&A[(i+3) * K + k]);

                        for (int j = j0; j < max_j - 15; j += 16) {
                            __m256 b0 = _mm256_loadu_ps(&B_transposed[j * K + k]);
                            __m256 b1 = _mm256_loadu_ps(&B_transposed[(j+8) * K + k]);

                            __m256 c00 = _mm256_loadu_ps(&C[i * N + j]);
                            __m256 c01 = _mm256_loadu_ps(&C[i * N + j + 8]);
                            __m256 c10 = _mm256_loadu_ps(&C[(i+1) * N + j]);
                            __m256 c11 = _mm256_loadu_ps(&C[(i+1) * N + j + 8]);
                            __m256 c20 = _mm256_loadu_ps(&C[(i+2) * N + j]);
                            __m256 c21 = _mm256_loadu_ps(&C[(i+2) * N + j + 8]);
                            __m256 c30 = _mm256_loadu_ps(&C[(i+3) * N + j]);
                            __m256 c31 = _mm256_loadu_ps(&C[(i+3) * N + j + 8]);

                            c00 = _mm256_fmadd_ps(a0, b0, c00);
                            c01 = _mm256_fmadd_ps(a0, b1, c01);
                            c10 = _mm256_fmadd_ps(a1, b0, c10);
                            c11 = _mm256_fmadd_ps(a1, b1, c11);
                            c20 = _mm256_fmadd_ps(a2, b0, c20);
                            c21 = _mm256_fmadd_ps(a2, b1, c21);
                            c30 = _mm256_fmadd_ps(a3, b0, c30);
                            c31 = _mm256_fmadd_ps(a3, b1, c31);

                            _mm256_storeu_ps(&C[i * N + j], c00);
                            _mm256_storeu_ps(&C[i * N + j + 8], c01);
                            _mm256_storeu_ps(&C[(i+1) * N + j], c10);
                            _mm256_storeu_ps(&C[(i+1) * N + j + 8], c11);
                            _mm256_storeu_ps(&C[(i+2) * N + j], c20);
                            _mm256_storeu_ps(&C[(i+2) * N + j + 8], c21);
                            _mm256_storeu_ps(&C[(i+3) * N + j], c30);
                            _mm256_storeu_ps(&C[(i+3) * N + j + 8], c31);
                        }
                    }
                }

                // Handle edge cases
                for (int i = max_i - (max_i % 4); i < max_i; i++) {
                    for (int k = k0; k < max_k; k++) {
                        __m256 a = _mm256_broadcast_ss(&A[i * K + k]);
                        for (int j = j0; j < max_j - 7; j += 8) {
                            __m256 b = _mm256_loadu_ps(&B_transposed[j * K + k]);
                            __m256 c = _mm256_loadu_ps(&C[i * N + j]);
                            c = _mm256_fmadd_ps(a, b, c);
                            _mm256_storeu_ps(&C[i * N + j], c);
                        }
                    }
                }

                for (int i = i0; i < max_i; i++) {
                    for (int k = k0; k < max_k; k++) {
                        float a = A[i * K + k];
                        for (int j = max_j - (max_j % 8); j < max_j; j++) {
                            C[i * N + j] += a * B_transposed[j * K + k];
                        }
                    }
                }
            }
        }
    }
}

int main() {
  #define TESTS 5

  int sizes[][3] = {{128, 128, 128}, {512, 512, 512}, {1024, 1024, 1024}};
  int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

  srand(time(NULL));

  printf("m,n,k,time,flops\n");

  double best_flops = 0.0;
  int best_m = 0, best_n = 0, best_k = 0;

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

      double total_flops = 0.0;

      for (int test = 0; test < TESTS; test++) {
          for (int j = 0; j < M * N; j++) {
              C[j] = 0.0f;
          }

          clock_t start_time = clock();

          matmul_blocked(A, B_transposed, C, M, N, K);

          clock_t end_time = clock();

          double iteration_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
          double flops = 2.0 * M * N * K;
          double flops_per_second = flops / iteration_time / 1e9;

          total_flops += flops_per_second;

          printf("%d,%d,%d,%.6f,%.2f\n", M, N, K, iteration_time, flops_per_second);
      }

      double avg_flops = total_flops / TESTS;

      if (avg_flops > best_flops) {
          best_flops = avg_flops;
          best_m = M;
          best_n = N;
          best_k = K;
      }

      free(A);
      free(B);
      free(B_transposed);
      free(C);
  }

  printf("Best configuration: M=%d, N=%d, K=%d, FLOPS=%.2f\n", best_m, best_n, best_k, best_flops);

  return 0;
}

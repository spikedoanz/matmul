#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void transpose(float *src, float *dst, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

void matmul(float *A, float *B_transposed, float *C, int M, int N, int K, int TILE_SIZE1, int TILE_SIZE2, int TILE_SIZE3) {
    for (int i = 0; i < M; i += TILE_SIZE1) {
        for (int j = 0; j < N; j += TILE_SIZE1) {
            for (int k = 0; k < K; k += TILE_SIZE1) {
                for (int ii = i; ii < i + TILE_SIZE1 && ii < M; ii += TILE_SIZE2) {
                    for (int jj = j; jj < j + TILE_SIZE1 && jj < N; jj += TILE_SIZE2) {
                        for (int kk = k; kk < k + TILE_SIZE1 && kk < K; kk += TILE_SIZE2) {
                            for (int iii = ii; iii < ii + TILE_SIZE2 && iii < M; iii += TILE_SIZE3) {
                                for (int jjj = jj; jjj < jj + TILE_SIZE2 && jjj < N; jjj += TILE_SIZE3) {
                                    for (int kkk = kk; kkk < kk + TILE_SIZE2 && kkk < K; kkk += TILE_SIZE3) {
                                        for (int i4 = iii; i4 < iii + TILE_SIZE3 && i4 < M; i4++) {
                                            for (int j4 = jjj; j4 < jjj + TILE_SIZE3 && j4 < N; j4++) {
                                                float acc = 0.0f;
                                                for (int k4 = kkk; k4 < kkk + TILE_SIZE3 && k4 < K; k4++) {
                                                    acc += A[i4 * K + k4] * B_transposed[j4 * K + k4];
                                                }
                                                C[i4 * N + j4] += acc;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

int main() {
    int sizes[][3] = {{128, 128, 128}, {512, 512, 512}, {1024, 1024, 1024}};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int tile_sizes[][3] = {
        {32, 16, 8},
        {64, 32, 16},
        {128, 64, 32},
        {256, 128, 64}
    };
    int num_tile_sizes = sizeof(tile_sizes) / sizeof(tile_sizes[0]);

    srand(time(NULL));

    printf("m,n,k,tile_size1,tile_size2,tile_size3,time,flops\n");

    for (int i = 0; i < num_sizes; i++) {
        int M = sizes[i][0];
        int N = sizes[i][1];
        int K = sizes[i][2];

        float *A = (float *)malloc(M * K * sizeof(float));
        float *B = (float *)malloc(K * N * sizeof(float));
        float *B_transposed = (float *)malloc(K * N * sizeof(float));
        float *C = (float *)malloc(M * N * sizeof(float));

        for (int j = 0; j < M * K; j++) {
            A[j] = (float)rand() / RAND_MAX;
        }
        for (int j = 0; j < K * N; j++) {
            B[j] = (float)rand() / RAND_MAX;
        }

        transpose(B, B_transposed, K, N);

        for (int t = 0; t < num_tile_sizes; t++) {
            int TILE_SIZE1 = tile_sizes[t][0];
            int TILE_SIZE2 = tile_sizes[t][1];
            int TILE_SIZE3 = tile_sizes[t][2];

            for (int j = 0; j < M * N; j++) {
                C[j] = 0.0f;
            }

            clock_t start_time = clock();

            matmul(A, B_transposed, C, M, N, K, TILE_SIZE1, TILE_SIZE2, TILE_SIZE3);

            clock_t end_time = clock();

            double iteration_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
            double flops = 2.0 * M * N * K;
            double flops_per_second = flops / iteration_time / 1e9;

            printf("%d,%d,%d,%d,%d,%d,%.6f,%.2f\n", M, N, K, TILE_SIZE1, TILE_SIZE2, TILE_SIZE3, iteration_time, flops_per_second);
        }

        free(A);
        free(B);
        free(B_transposed);
        free(C);
    }

    return 0;
}

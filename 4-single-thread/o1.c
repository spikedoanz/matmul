// transposing b
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

void matmul(float *A, float *B_transposed, float *C, int M, int N, int K) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float acc = 0.0f;
            for (int inner = 0; inner < K; inner++) {
                acc += A[row * K + inner] * B_transposed[col * K + inner];
            }
            C[row * N + col] = acc;
        }
    }
}

int main() {
    int sizes[][3] = {{128, 128, 128}, {512, 512, 512}, {1024, 1024, 1024}};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int num_iterations = 100;

    srand(time(NULL));

    printf("m,n,k,time,flops\n");

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

        for (int iter = 0; iter < num_iterations; iter++) {
            clock_t start_time = clock();

            matmul(A, B_transposed, C, M, N, K);

            clock_t end_time = clock();

            double iteration_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
            double flops = 2.0 * M * N * K;
            double flops_per_second = flops / iteration_time / 1e9;

            printf("%d,%d,%d,%.6f,%.2f\n", M, N, K, iteration_time, flops_per_second);
        }

        free(A);
        free(B);
        free(B_transposed);
        free(C);
    }

    return 0;
}

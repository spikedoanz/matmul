#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void benchmark_matmul(int M, int N, int K, int num_iterations) {
    float *A = (float *)malloc(M * K * sizeof(float));
    float *B = (float *)malloc(K * N * sizeof(float));
    float *C = (float *)malloc(M * N * sizeof(float));

    // Initialize matrices with random values
    for (int i = 0; i < M * K; i++) {
        A[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = (float)rand() / RAND_MAX;
    }

    printf("m,n,k,time,flops\n");
    for (int iter = 0; iter < num_iterations; iter++) {
        clock_t start_time = clock();

        for (int row = 0; row < M; row++) {
            for (int col = 0; col < N; col++) {
                float acc = 0.0f;
                for (int inner = 0; inner < K; inner++) {
                    sum += A[row * K + inner] * B[inner * N + col];
                }
                C[row * N + col] = acc;
            }
        }

        clock_t end_time = clock();

        double iteration_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
        double flops = 2.0 * M * N * K;
        double flops_per_second = flops / iteration_time / 1e9;  // Convert to gigaflops

        printf("%d,%d,%d,%.6f,%.2f\n", M, N, K, iteration_time, flops_per_second);
    }

    free(A);
    free(B);
    free(C);
}

int main() {
    int sizes[][3] = {{128, 128, 128}, {512, 512, 512}, {1024, 1024, 1024}};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < num_sizes; i++) {
        int M = sizes[i][0];
        int N = sizes[i][1];
        int K = sizes[i][2];
        printf("\nMatrix size: %dx%d * %dx%d\n", M, K, K, N);
        benchmark_matmul(M, N, K, 100);
    }

    return 0;
}

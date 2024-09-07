#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void add_matrix(float *C, float *A, float *B, int size) {
    for (int i = 0; i < size * size; i++) {
        C[i] = A[i] + B[i];
    }
}

void subtract_matrix(float *C, float *A, float *B, int size) {
    for (int i = 0; i < size * size; i++) {
        C[i] = A[i] - B[i];
    }
}

void strassen(float *A, float *B, float *C, int n) {
    if (n <= 64) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                C[i * n + j] = 0;
                for (int k = 0; k < n; k++) {
                    C[i * n + j] += A[i * n + k] * B[k * n + j];
                }
            }
        }
        return;
    }

    int new_size = n / 2;
    float *A11 = (float *)malloc(new_size * new_size * sizeof(float));
    float *A12 = (float *)malloc(new_size * new_size * sizeof(float));
    float *A21 = (float *)malloc(new_size * new_size * sizeof(float));
    float *A22 = (float *)malloc(new_size * new_size * sizeof(float));
    float *B11 = (float *)malloc(new_size * new_size * sizeof(float));
    float *B12 = (float *)malloc(new_size * new_size * sizeof(float));
    float *B21 = (float *)malloc(new_size * new_size * sizeof(float));
    float *B22 = (float *)malloc(new_size * new_size * sizeof(float));

    float *M1 = (float *)malloc(new_size * new_size * sizeof(float));
    float *M2 = (float *)malloc(new_size * new_size * sizeof(float));
    float *M3 = (float *)malloc(new_size * new_size * sizeof(float));
    float *M4 = (float *)malloc(new_size * new_size * sizeof(float));
    float *M5 = (float *)malloc(new_size * new_size * sizeof(float));
    float *M6 = (float *)malloc(new_size * new_size * sizeof(float));
    float *M7 = (float *)malloc(new_size * new_size * sizeof(float));

    float *temp1 = (float *)malloc(new_size * new_size * sizeof(float));
    float *temp2 = (float *)malloc(new_size * new_size * sizeof(float));

    for (int i = 0; i < new_size; i++) {
        for (int j = 0; j < new_size; j++) {
            A11[i * new_size + j] = A[i * n + j];
            A12[i * new_size + j] = A[i * n + j + new_size];
            A21[i * new_size + j] = A[(i + new_size) * n + j];
            A22[i * new_size + j] = A[(i + new_size) * n + j + new_size];

            B11[i * new_size + j] = B[i * n + j];
            B12[i * new_size + j] = B[i * n + j + new_size];
            B21[i * new_size + j] = B[(i + new_size) * n + j];
            B22[i * new_size + j] = B[(i + new_size) * n + j + new_size];
        }
    }

    add_matrix(temp1, A11, A22, new_size);
    add_matrix(temp2, B11, B22, new_size);
    strassen(temp1, temp2, M1, new_size);

    add_matrix(temp1, A21, A22, new_size);
    strassen(temp1, B11, M2, new_size);

    subtract_matrix(temp1, B12, B22, new_size);
    strassen(A11, temp1, M3, new_size);

    subtract_matrix(temp1, B21, B11, new_size);
    strassen(A22, temp1, M4, new_size);

    add_matrix(temp1, A11, A12, new_size);
    strassen(temp1, B22, M5, new_size);

    subtract_matrix(temp1, A21, A11, new_size);
    add_matrix(temp2, B11, B12, new_size);
    strassen(temp1, temp2, M6, new_size);

    subtract_matrix(temp1, A12, A22, new_size);
    add_matrix(temp2, B21, B22, new_size);
    strassen(temp1, temp2, M7, new_size);

    add_matrix(temp1, M1, M4, new_size);
    subtract_matrix(temp2, temp1, M5, new_size);
    add_matrix(C, temp2, M7, new_size);

    add_matrix(C + new_size, M3, M5, new_size);
    add_matrix(C + new_size * n, M2, M4, new_size);
    add_matrix(temp1, M1, M3, new_size);
    subtract_matrix(temp2, temp1, M2, new_size);
    add_matrix(C + new_size * n + new_size, temp2, M6, new_size);

    free(A11); free(A12); free(A21); free(A22);
    free(B11); free(B12); free(B21); free(B22);
    free(M1); free(M2); free(M3); free(M4); free(M5); free(M6); free(M7);
    free(temp1); free(temp2);
}

int main() {
    int sizes[][3] = {{128, 128, 128}, {512, 512, 512}, {1024, 1024, 1024}};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int num_iterations = 5;

    srand(time(NULL));

    printf("m,n,k,time,flops\n");

    for (int i = 0; i < num_sizes; i++) {
        int M = sizes[i][0];
        int N = sizes[i][1];
        int K = sizes[i][2];

        float *A = (float *)malloc(M * K * sizeof(float));
        float *B = (float *)malloc(K * N * sizeof(float));
        float *C = (float *)malloc(M * N * sizeof(float));

        for (int j = 0; j < M * K; j++) {
            A[j] = (float)rand() / RAND_MAX;
        }
        for (int j = 0; j < K * N; j++) {
            B[j] = (float)rand() / RAND_MAX;
        }

        for (int iter = 0; iter < num_iterations; iter++) {
            clock_t start_time = clock();

            strassen(A, B, C, M);

            clock_t end_time = clock();

            double iteration_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
            double flops = 2.0 * M * N * K;
            double flops_per_second = flops / iteration_time / 1e9;

            printf("%d,%d,%d,%.6f,%.2f\n", M, N, K, iteration_time, flops_per_second);
        }

        free(A);
        free(B);
        free(C);
    }

    return 0;
}

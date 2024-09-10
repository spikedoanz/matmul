import subprocess
import tempfile
import os

def generate_matmul_function(M, N, K, tile_size=24):
    code = f"""
void matmul(float *A, float *B, float *C, int M, int N, int K) {{
    for (int i = 0; i < M; i += {tile_size}) {{
        for (int j = 0; j < N; j += {tile_size}) {{
            for (int k = 0; k < K; k += {tile_size}) {{
                {generate_tiled_matmul(tile_size)}
            }}
        }}
    }}
}}
"""
    return code

def generate_tiled_matmul(tile_size):
    code = f"""
                for (int ii = i; ii < i + {tile_size} && ii < M; ii++) {{
                    for (int jj = j; jj < j + {tile_size} && jj < N; jj += 32) {{
                        {generate_fma_operations()}
                    }}
                }}
"""
    return code

def generate_fma_operations():
    code = """
                        __m256 sum0 = _mm256_setzero_ps();
                        __m256 sum1 = _mm256_setzero_ps();
                        __m256 sum2 = _mm256_setzero_ps();
                        __m256 sum3 = _mm256_setzero_ps();
                        for (int kk = k; kk < k + TILE_SIZE && kk < K; kk++) {
                            __m256 a = _mm256_set1_ps(A[ii * K + kk]);
                            __m256 b0 = _mm256_loadu_ps(&B[kk * N + jj]);
                            __m256 b1 = _mm256_loadu_ps(&B[kk * N + (jj + 8 < N ? jj + 8 : jj)]);
                            __m256 b2 = _mm256_loadu_ps(&B[kk * N + (jj + 16 < N ? jj + 16 : jj)]);
                            __m256 b3 = _mm256_loadu_ps(&B[kk * N + (jj + 24 < N ? jj + 24 : jj)]);
                            sum0 = _mm256_fmadd_ps(a, b0, sum0);
                            sum1 = _mm256_fmadd_ps(a, b1, sum1);
                            sum2 = _mm256_fmadd_ps(a, b2, sum2);
                            sum3 = _mm256_fmadd_ps(a, b3, sum3);
                        }
                        _mm256_storeu_ps(&C[ii * N + jj], _mm256_add_ps(_mm256_loadu_ps(&C[ii * N + jj]), sum0));
                        if (jj + 8 < N) _mm256_storeu_ps(&C[ii * N + jj + 8], _mm256_add_ps(_mm256_loadu_ps(&C[ii * N + jj + 8]), sum1));
                        if (jj + 16 < N) _mm256_storeu_ps(&C[ii * N + jj + 16], _mm256_add_ps(_mm256_loadu_ps(&C[ii * N + jj + 16]), sum2));
                        if (jj + 24 < N) _mm256_storeu_ps(&C[ii * N + jj + 24], _mm256_add_ps(_mm256_loadu_ps(&C[ii * N + jj + 24]), sum3));
"""
    return code

def generate_full_c_code(M, N, K):
    headers = """
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>

#define TILE_SIZE 24

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
"""
    
    matmul_function = generate_matmul_function(M, N, K)
    
    main_function = f"""
int main() {{
    int M = {M}, N = {N}, K = {K};
    float *A = aligned_alloc(32, M * K * sizeof(float));
    float *B = aligned_alloc(32, K * N * sizeof(float));
    float *C = aligned_alloc(32, M * N * sizeof(float));

    // Initialize matrices with random values
    for (int i = 0; i < M * K; i++) A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) B[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < M * N; i++) C[i] = 0.0f;

    double start_time = get_time();
    matmul(A, B, C, M, N, K);
    double end_time = get_time();

    double elapsed_time = end_time - start_time;
    double flops = 2.0 * M * N * K;
    double gflops = flops / (elapsed_time * 1e9);

    printf("%.6f,%.2f\\n", elapsed_time, gflops);

    free(A);
    free(B);
    free(C);

    return 0;
}}
"""
    
    return headers + matmul_function + main_function

def compile_and_run(c_code):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as temp_c_file:
        temp_c_file.write(c_code)
        temp_c_filename = temp_c_file.name

    output_filename = tempfile.mktemp()

    try:
        # Compile the C code
        compile_command = ['gcc', '-O3', '-march=native', '-mavx2', '-mfma', '-o', output_filename, temp_c_filename, '-lm']
        subprocess.run(compile_command, check=True)

        # Run the compiled program
        result = subprocess.run([output_filename], capture_output=True, text=True, check=True)
        
        # Parse the output
        time, gflops = map(float, result.stdout.strip().split(','))
        
        return time, gflops

    finally:
        # Clean up temporary files
        os.unlink(temp_c_filename)
        if os.path.exists(output_filename):
            os.unlink(output_filename)

def run_matmul_benchmark(M, N, K, num_runs=5):
    c_code = generate_full_c_code(M, N, K)
    
    total_time = 0
    total_gflops = 0
    
    for _ in range(num_runs):
        time, gflops = compile_and_run(c_code)
        total_time += time
        total_gflops += gflops
    
    avg_time = total_time / num_runs
    avg_gflops = total_gflops / num_runs
    
    return avg_time, avg_gflops

# Example usage
sizes = [(128, 128, 128), (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]

print("M,N,K,Time (s),GFLOPS")
for M, N, K in sizes:
    avg_time, avg_gflops = run_matmul_benchmark(M, N, K)
    print(f"{M},{N},{K},{avg_time:.6f},{avg_gflops:.2f}")

"""
Tricks used
- compiler flag optimizations
- multithreading
- fused multiply adds
- prefetching
"""
import subprocess
import argparse
import tempfile
import os

def generate_matmul_function(M, N, K, block_size):
    code = f"""
void matmul(float *A, float *B, float *C, int M, int N, int K) {{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i += {block_size}) {{
        for (int j = 0; j < N; j += {block_size}) {{
            for (int k = 0; k < K; k += {block_size}) {{
                {generate_tiled_matmul(block_size)}
            }}
        }}
    }}
}}
"""
    return code

def generate_tiled_matmul(block_size):
    code = f"""
                int max_ii = (i + {block_size} < M) ? i + {block_size} : M;
                int max_jj = (j + {block_size} < N) ? j + {block_size} : N;
                int max_kk = (k + {block_size} < K) ? k + {block_size} : K;
                for (int ii = i; ii < max_ii; ii++) {{
                    {generate_unrolled_j_loop(block_size)}
                }}
"""
    return code

def generate_unrolled_j_loop(block_size):
    unrolled_code = ""
    for jj in range(0, block_size, 8):
        unrolled_code += f"""
                    if (j + {jj} < N) {{
                        {generate_fma_operations(jj)}
                    }}
"""
    return unrolled_code

def generate_fma_operations(jj_offset):
    code = f"""
                        __m256 sum = _mm256_setzero_ps();
                        for (int kk = k; kk < max_kk; kk++) {{
                            __m256 a = _mm256_set1_ps(A[ii * K + kk]);
                            __m256 b = _mm256_loadu_ps(&B[kk * N + j + {jj_offset}]);
                            sum = _mm256_fmadd_ps(a, b, sum);
                        }}
                        _mm256_storeu_ps(&C[ii * N + j + {jj_offset}], _mm256_add_ps(_mm256_loadu_ps(&C[ii * N + j + {jj_offset}]), sum));
"""
    return code

def generate_full_c_code(M, N, K, block_size, num_threads):
    headers = f"""
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#include <omp.h>

#define BLOCK_SIZE {block_size}

double get_time() {{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}}
"""
    
    matmul_function = generate_matmul_function(M, N, K, block_size)
    
    main_function = f"""
int main() {{
    int M = {M}, N = {N}, K = {K};
    float *A = (float *)aligned_alloc(32, M * K * sizeof(float));
    float *B = (float *)aligned_alloc(32, K * N * sizeof(float));
    float *C = (float *)aligned_alloc(32, M * N * sizeof(float));

    if (!A || !B || !C) {{
        fprintf(stderr, "Memory allocation failed\\n");
        return 1;
    }}

    // Initialize matrices with random values
    for (int i = 0; i < M * K; i++) A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) B[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < M * N; i++) C[i] = 0.0f;

    omp_set_num_threads({num_threads});

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
        compile_command = ['gcc', '-O3', '-march=native', '-mavx2', '-mfma', '-fopenmp', '-o', output_filename, temp_c_filename]
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

def run_matmul_benchmark(M, N, K, block_size, num_threads, num_runs=5, save=False):
    c_code = generate_full_c_code(M, N, K, block_size, num_threads)
    
    if save:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as temp_file:
            temp_file.write(c_code)
            temp_file_name = temp_file.name
        print(f"Kernel saved to: {temp_file_name}")
    
    total_time = 0
    total_gflops = 0
    
    for _ in range(num_runs):
        time, gflops = compile_and_run(c_code)
        total_time += time
        total_gflops += gflops
    
    avg_time = total_time / num_runs
    avg_gflops = total_gflops / num_runs
    
    return avg_time, avg_gflops

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run matrix multiplication benchmark')
    parser.add_argument('--save', action='store_true', help='Save the kernel to a temporary C file')
    parser.add_argument('--block-size', type=int, default=24, help='Block size for tiled matrix multiplication')
    parser.add_argument('--num-threads', type=int, default=24, help='Number of threads to use')
    args = parser.parse_args()

    sizes = [(128, 128, 128), (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]

    print("M,N,K,Block Size,Threads,Time (s),GFLOPS")
    for M, N, K in sizes:
        avg_time, avg_gflops = run_matmul_benchmark(M, N, K, args.block_size, args.num_threads, save=args.save)
        print(f"{M},{N},{K},{args.block_size},{args.num_threads},{avg_time:.6f},{avg_gflops:.2f}")

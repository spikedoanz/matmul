import numpy as np
import time

def benchmark_matmul(M, N, K, num_iterations=100):
    A = np.random.rand(M, K)
    B = np.random.rand(K, N)

    print("m,n,k,time,flops")
    for i in range(num_iterations):
        start_time = time.time()
        C = np.matmul(A, B)
        end_time = time.time()
        
        iteration_time = end_time - start_time
        flops = 2 * M * N * K  # FLOPS for matrix multiplication
        flops_per_second = flops / iteration_time
        
        print(f"{M},{N},{K},{iteration_time:.6f},{flops_per_second:.2f}")

def main():
    sizes = [(100, 100, 100), (500, 500, 500), (1000, 1000, 1000)]

    for M, N, K in sizes:
        print(f"\nMatrix size: {M}x{K} * {K}x{N}")
        benchmark_matmul(M, N, K)

if __name__ == "__main__":
    main()

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import time

def benchmark_matmul(M, N, K, num_iterations):
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)

    for _ in range(num_iterations):
        start_time = time.time()

        C = np.matmul(A, B)

        end_time = time.time()

        iteration_time = end_time - start_time
        flops = 2.0 * M * N * K
        flops_per_second = flops / iteration_time / 1e9

        print(f"{M},{N},{K},{iteration_time:.6f},{flops_per_second:.2f}")

    return flops_per_second

def main():
    sizes = [(128, 128, 128), (512, 512, 512), (1024, 1024, 1024)]
    num_iterations = 100

    print("m,n,k,time,flops")

    max_flops = 0
    max_config = None

    for M, N, K in sizes:
        flops = benchmark_matmul(M, N, K, num_iterations)
        if flops > max_flops:
            max_flops = flops
            max_config = (M, N, K)

    print(f"\nConfiguration with highest FLOPS: {max_config}")
    print(f"Highest FLOPS: {max_flops:.2f} GFLOPS")

if __name__ == "__main__":
    main()
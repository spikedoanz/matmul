import torch
import time

def benchmark_matmul(M, N, K, num_iterations=100):
    A = torch.rand(M, K)
    B = torch.rand(K, N)
    C = torch.zeros(M, N)

    print("m,n,k,time,flops")
    for i in range(num_iterations):
        start_time = time.time()
        
        for row in range(M):
            for col in range(N):
                for inner in range(K):
                    C[row, col] += A[row, inner] * B[inner, col]
        
        end_time = time.time()

        iteration_time = end_time - start_time
        flops = 2 * M * N * K
        flops_per_second = flops / iteration_time / 1e9  # Convert to gigaflops

        print(f"{M},{N},{K},{iteration_time:.6f},{flops_per_second:.2f}")

def main():
    sizes = [(128, 128, 128), (512, 512, 512), (1024, 1024, 1024)]

    for M, N, K in sizes:
        print(f"\nMatrix size: {M}x{K} * {K}x{N}")
        benchmark_matmul(M, N, K)

if __name__ == "__main__":
    main()

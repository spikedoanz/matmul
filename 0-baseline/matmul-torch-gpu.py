import torch
import time

def benchmark_matmul(M, N, K, num_iterations=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.rand(M, K, device=device)
    B = torch.rand(K, N, device=device)

    print("m,n,k,time,flops")
    for i in range(num_iterations):
        start_time = time.time()
        C = torch.matmul(A, B)
        torch.cuda.synchronize()
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

# Matmul 

This repo contains a number of matrix multiplication implementations, in order of increasing complexity

1. Naive Implementation -- Python
- Triply nested loops for baseline

2. Naive Implementation -- C

3. Algorithm Optimization
- Implement Strassen's algorithm
- Explore other advanced algorithms (e.g., Coppersmith–Winograd algorithm)
- Focus on lowering theoretical time complexity

4. Code-level Optimizations
- Loop strength reduction
- Block partitioning
- Vectorize

5. Cache Optimization
- Shove outer blocks into L1, shove blocks of blocks into L2, etc

6. Single-threaded Performance Tuning

7. Multi-threaded Implementation

8. GPU Acceleration
- Metal
- Cuda

9. Distributed Computing
- Multi GPU (sharding)
- Multi node with multi gpu

9. Mixed-precision Arithmetic
- f64 vs f16 vs int8

10. Sparse Matrix Optimization

11. Heterogeneous Computing
- 2 different GPUs or CPUxGPU mamuls

12. Autotuning
- Create systems for automatic performance tuning
- Implement runtime adaptation to workload characteristics

## Benchmarking and Profiling
- Develop comprehensive benchmarking suites
- Implement detailed profiling tools for performance analysis

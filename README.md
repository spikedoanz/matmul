# matmul 

This repo contains a number of matrix multiplication implementations, in order of increasing complexity

---

## Agenda

- What is a matrix
- How is it encoded
- What are matmuls, naive impl
- Multithreading discussion
- GPUs are cope, vector instructions are cope

## Implementations 

0. Baselines
- numpy: ~300 gflops
- tinygrad: ~700 gflops
- torch: ~18 tflops

1. Naive Implementation -- Python : 0.0 gflops

2. Naive Implementation -- C : 0.75 gflops

3. Lowered time complexity -- C : 0.92 glfops

4. Single threaded Optimizations -- C

5. Multi-threaded Implementation -- C

6. GPU Acceleration -- C cuda

7. Distributed Computing -- Multi GPU

8. Distributed Computing -- Multi Node

9. Arcane optimizations


---

## Benchmarking

### CPU: [Ryzen 5900X3D](https://www.amd.com/en/products/processors/desktops/ryzen/5000-series/amd-ryzen-9-5900x.html)
- 12 cores
- Base clock 3.7GHz
- Boost clock 4.8GHz
- 24 threads (why is this not infinite?)
- L1 Cache: ???
- L2 Cache: 6MB
- L3 Cache: 64MB
- Memory transfer: 3200 MT/s


- 32 GB GDDR4
- ??? bandwidth

### GPU: [RTX 3090](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/)
- 10496 CUDA cores
- 328 Tensor cores
- Boost Clock 1.40 GHz
- Boost Clock 1.7 GHz
- 24 GB GDDR6X memory
- 936 GB/s memory bandwidth


### Distributed

### Tools

## Theoretical bounds

## Supertheoretical bounds




## Resources

https://github.com/OpenMathLib/OpenBLAS/tree/develop

https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE/tree/master

https://github.com/siboehm/SGEMM_CUDA

http://blog.ezyang.com/2019/05/pytorch-internals/

https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/

https://en.algorithmica.org/hpc/cpu-cache/associativity/

https://arxiv.org/abs/2301.03598


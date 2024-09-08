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

- CPU, singlethreaded (numpy): 120 GFLOPS
- CPU, multithreaded (torch/numpy): 1 TFLOPS
- GPU (torch): 24 TFLOPS

1. Naive Implementation -- Python : 0.0 GFLOPS (not measurable)

2. Naive Implementation -- C : 2.47 GFLOPS

3. Lowered time complexity -- C : 7.13 GFLOPS

4. Single threaded Optimizations -- C : 58 GFLOPS

5. Multi-threaded Implementation -- C : 250 GFLOPS

6. GPU Acceleration -- C cuda : 5 TFLOPS

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


```
NUMANode L#0 (P#0 31GB)
L3 L#0 (32MB)
  L2 L#0 (512KB) + L1d L#0 (32KB) + L1i L#0 (32KB) + Core L#0
    PU L#0 (P#0)
    PU L#1 (P#12)
  L2 L#1 (512KB) + L1d L#1 (32KB) + L1i L#1 (32KB) + Core L#1
    PU L#2 (P#1)
    PU L#3 (P#13)
  L2 L#2 (512KB) + L1d L#2 (32KB) + L1i L#2 (32KB) + Core L#2
    PU L#4 (P#2)
    PU L#5 (P#14)
  L2 L#3 (512KB) + L1d L#3 (32KB) + L1i L#3 (32KB) + Core L#3
    PU L#6 (P#3)
    PU L#7 (P#15)
  L2 L#4 (512KB) + L1d L#4 (32KB) + L1i L#4 (32KB) + Core L#4
    PU L#8 (P#4)
    PU L#9 (P#16)
  L2 L#5 (512KB) + L1d L#5 (32KB) + L1i L#5 (32KB) + Core L#5
    PU L#10 (P#5)
    PU L#11 (P#17)
L3 L#1 (32MB)
  L2 L#6 (512KB) + L1d L#6 (32KB) + L1i L#6 (32KB) + Core L#6
    PU L#12 (P#6)
    PU L#13 (P#18)
  L2 L#7 (512KB) + L1d L#7 (32KB) + L1i L#7 (32KB) + Core L#7
    PU L#14 (P#7)
    PU L#15 (P#19)
  L2 L#8 (512KB) + L1d L#8 (32KB) + L1i L#8 (32KB) + Core L#8
    PU L#16 (P#8)
    PU L#17 (P#20)
  L2 L#9 (512KB) + L1d L#9 (32KB) + L1i L#9 (32KB) + Core L#9
    PU L#18 (P#9)
    PU L#19 (P#21)
  L2 L#10 (512KB) + L1d L#10 (32KB) + L1i L#10 (32KB) + Core L#10
    PU L#20 (P#10)
    PU L#21 (P#22)
  L2 L#11 (512KB) + L1d L#11 (32KB) + L1i L#11 (32KB) + Core L#11
    PU L#22 (P#11)
    PU L#23 (P#23)
```





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


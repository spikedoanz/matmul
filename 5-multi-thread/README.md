# Triple for loop in C

naive parallelization
```
(venv) ~/work/matmul/5-multi-thread § gcc matmul.c -O3 -mavx2 -mfma ; ./a.out
threads,m,n,k,time,gflops
8,128,128,128,0.000607,7.25
8,512,512,512,0.021483,12.50
8,1024,1024,1024,0.451224,4.76
12,128,128,128,0.000602,7.01
12,512,512,512,0.017073,16.27
12,1024,1024,1024,0.295033,7.28
16,128,128,128,0.000547,7.82
16,512,512,512,0.016331,16.56
16,1024,1024,1024,0.240848,8.93
20,128,128,128,0.000719,5.97
20,512,512,512,0.014674,18.33
20,1024,1024,1024,0.202164,10.63
24,128,128,128,0.000779,5.53
24,512,512,512,0.013012,21.22
24,1024,1024,1024,0.180670,11.89
```

tiling
```
(venv) ~/work/matmul/5-multi-thread § gcc tiling.c -O3 -mavx2 -mfma ; ./a.out
threads,m,n,k,time,gflops
24,128,128,128,0.000820,5.61
24,512,512,512,0.011813,22.74
24,1024,1024,1024,0.163601,13.13
```

vectorized + tiling
```
(venv) ~/work/matmul/5-multi-thread § gcc vector-tiling.c -O3 -mavx2 -mfma ; ./a.out
threads,m,n,k,time,gflops
24,128,128,128,0.001064,3.96
24,512,512,512,0.001927,142.16
24,1024,1024,1024,0.016782,128.50
```

```
(venv) ~/work/matmul/5-multi-thread § gcc unroll-prefetch-vector-tiling.c -O3 -mavx2 -mfma ; ./a.out
threads,m,n,k,time,gflops
24,128,128,128,0.000790,5.46
24,512,512,512,0.001587,171.36
24,1024,1024,1024,0.009541,228.00
24,2048,2048,2048,0.090838,189.66
```

> i can totally unroll every single loop with metaprogramming
> pretty sure blas does this statically with inline assembly
> but this is beyond claude's ability soooooo

---






















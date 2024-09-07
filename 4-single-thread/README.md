# Triple for loop in C


- adding compiler optimization flags

```
transposed: 0.75 gflops
-O1: 4 gflops
-O2: 3.8 gflops
-O3: 5 gflops
```

```
transposed + tiling: 0.76 gflops tile size 32
-O1: 5.06 gflops
-O2: 4.06 gflops
-O3: 5 gflops
```

```
transposed + tiling: 0.76 gflops tile size 8
-O1: 6.43 gflops
-O2: 4.06 gflops
-O3: 5 gflops
```

```
transposed + multi + tiling: 0.76 gflops tile size 8
-O1: 5.18 gflops
-O2: 4.53 gflops
-O3: 6.09 gflops
```

```
vectorized : 8.16
-O1 : 8.14
-O2 : 8.14
-O3 : 8.51
```

```
prefetch : 8.07
-O1 : 8.18
-O2 : 8.11
-O3 : 8.11
```

```
unrolling : 8.51
-O1 : 8
ditto
```

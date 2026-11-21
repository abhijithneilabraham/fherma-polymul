# Batched polynomial multiplication

A CUDA implementation of exact multiplication in Z_q[X]/(X^N+1).

The kernel uses a twisted NTT, Montgomery arithmetic, and cooperating lanes
for wide coefficients. Public point constants and device buffers are prepared
once. Operand transforms and output materialization are timed. Pure byte copies
use the standard transfer hooks and are timed separately.

Specification: `polynomial-multiplication/negacyclic-batched@1.0.0`.

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
./build/solution POINT_DIRECTORY
```

The challenge image provides CUDA and cuPQC. The default architecture is the
GPU visible to the build. Use `-DFHERMA_CUDA_ARCH=sm_89` for a CPU-hosted compile
check targeting that architecture.

## Measured result

All three official cases passed at B=128, N=65,536, W=1,747. Median: 1.179581267 s
on `fherma-gpu-rtx6000`. The initial median was 1.645653063 s. Rank remains 20.

Measured source: `1ce7fae0156a3b3737bb2e52f8141f196eb72447`.

The [resident RNS adapter](rns-resident/) uses small-prime transforms, integer
tensor-core reconstruction, and CUDA graph replay. CPU models and CUDA
compilation pass. Its official measurement is pending.

[Challenge entry](https://www.fherma.io/kernels/polynomial-multiplication/implementations/abhijith-batched-cuda-ntt)

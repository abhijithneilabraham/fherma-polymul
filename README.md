# Batched polynomial multiplication

A CUDA implementation of exact multiplication in Z_q[X]/(X^N+1).

The measured kernel uses small-prime NTTs and integer tensor-core reconstruction.
Four product slots share input-copy, compute and output-copy queues. Pinned
staging and parallel host copies keep data moving. All transfers and host output
construction are timed in this version.

Specification: `polynomial-multiplication/negacyclic-batched@1.0.0`.

```sh
cmake -S streaming-rns -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
./build/solution POINT_DIRECTORY
```

The challenge image provides CUDA and cuPQC. The default architecture is the
GPU visible to the build. Use `-DFHERMA_CUDA_ARCH=sm_89` for a CPU-hosted compile
check targeting that architecture.

## Measured result

All three official cases passed at B=128, N=65,536, W=1,747. Median: 89.184818 ms
on `fherma-gpu-rtx6000`. Recorded rank: 17.

Measured source: `7d1d7052a7a3455c50462d3c0da4885a250229ba`, path `streaming-rns`.

The [resident copy candidate](resident-copy/) keeps the graph arithmetic and
uses pure pinned-copy hooks. It constructs the owned host output during download.
This candidate is being qualified separately.

[Challenge entry](https://www.fherma.io/kernels/polynomial-multiplication/implementations/abhijith-streaming-rns)

# Batched polynomial multiplication

A CUDA implementation of exact multiplication in Z_q[X]/(X^N+1).

The measured kernel uses small-prime NTTs, integer tensor-core reconstruction,
and CUDA graph replay. Public point constants and device buffers are prepared
once. Operand transforms and output materialization are timed. Pure byte copies
use the standard transfer hooks and are timed separately.

Specification: `polynomial-multiplication/negacyclic-batched@1.0.0`.

```sh
cmake -S rns-resident -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
./build/solution POINT_DIRECTORY
```

The challenge image provides CUDA and cuPQC. The default architecture is the
GPU visible to the build. Use `-DFHERMA_CUDA_ARCH=sm_89` for a CPU-hosted compile
check targeting that architecture.

## Measured result

All three official cases passed at B=128, N=65,536, W=1,747. Median: 164.554564 ms
on `fherma-gpu-rtx6000`. Recorded rank: 18.

Measured source: `0f36b168b675124d836ae6b362c12454c395f205`, path `rns-resident`.

The [streaming RNS candidate](streaming-rns/) preserves the pinned-copy
pipeline. It passed CPU wrapper checks and CUDA compilation. Its judge run
is queued.

[Challenge entry](https://www.fherma.io/kernels/polynomial-multiplication/implementations/abhijith-resident-rns-graph)

# Batched polynomial multiplication

A CUDA implementation of exact multiplication in Z_q[X]/(X^N+1).

The kernel uses a twisted NTT, Montgomery arithmetic, and cooperating lanes
for wide coefficients. Public point constants and device buffers are prepared
once. Operand transforms, transfers, and output materialization are timed.

Specification: `polynomial-multiplication/negacyclic-batched@1.0.0`.

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
./build/solution POINT_DIRECTORY
```

The challenge image provides CUDA and cuPQC. The default architecture is the
GPU visible to the build. Use `-DFHERMA_CUDA_ARCH=sm_89` for a CPU-hosted compile
check targeting that architecture.

Qualification: host math tests and CUDA compile/link pass. Official GPU
correctness and performance results are pending.

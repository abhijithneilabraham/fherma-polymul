# Batched polynomial multiplication

Exact CUDA multiplication in Z_q[X]/(X^N+1).

The kernel uses small-prime NTTs and integer tensor-core reconstruction. A CUDA
graph computes the resident batch. Pinned copy hooks move raw bytes; all arithmetic
and device completion remain in `run`. The output has independent host ownership.

Specification: `polynomial-multiplication/negacyclic-batched@1.0.0`.

```sh
cmake -S resident-copy -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
./build/solution POINT_DIRECTORY
```

The challenge image provides CUDA and cuPQC.

## Measured result

B=128, N=65,536, W=1,747: **77.362807 ms**, all three official cases passed.
Recorded rank: **17** on `fherma-gpu-rtx6000`.

Measured source: `b271b1ad3a814e58b4069a29c5168be16790c871`, path `resident-copy`.
Copies are measured separately: median H2D 145.836771 ms; D2H 66.188752 ms.
The prior [streaming version](streaming-rns/) scored 89.184818 ms with copies included.

[Challenge entry](https://www.fherma.io/kernels/polynomial-multiplication/implementations/abhijith-resident-copy-rns)

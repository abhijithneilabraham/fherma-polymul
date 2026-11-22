# Batched polynomial multiplication

Exact CUDA multiplication in Z_q[X]/(X^N+1).

The kernel uses small-prime NTTs and integer tensor-core reconstruction. A CUDA
graph computes the resident batch. Pinned copy hooks move raw bytes; all arithmetic
and device completion remain in `run`. The output has independent host ownership.

Specification: `polynomial-multiplication/negacyclic-batched@1.0.0`.

```sh
cmake -S warp-ntt -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
./build/solution POINT_DIRECTORY
```

The challenge image provides CUDA and cuPQC.

## Measured result

B=128, N=65,536, W=1,747: **67.041671 ms**, all three official cases passed.
Recorded rank: **15** on `fherma-gpu-rtx6000`.

Measured source: `6e8eb0ac9862956ebb72cd6e9ae2d10cbea16fa7`, path `warp-ntt`.
Copies are measured separately: median H2D 129.363585 ms; D2H 64.918957 ms.
The prior [streaming version](streaming-rns/) scored 89.184818 ms with copies included.

[Challenge entry](https://www.fherma.io/kernels/polynomial-multiplication/implementations/abhijith-warp-ntt-rns)

Warp transforms keep each small FFT in registers and exchange values through warp shuffles.
The new result reduces ranked time by 11.86% from 76.062912 ms.

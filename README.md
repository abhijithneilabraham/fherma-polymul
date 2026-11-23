# Batched polynomial multiplication

Exact CUDA multiplication in Z_q[X]/(X^N+1).

The kernel uses small-prime NTTs and integer tensor-core reconstruction. A CUDA
graph computes the resident batch. Pinned copy hooks move raw bytes; all arithmetic
and device completion remain in `run`. The output has independent host ownership.

Specification: `polynomial-multiplication/negacyclic-batched@1.0.0`.

```sh
cmake -S odd-roots -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
./build/solution POINT_DIRECTORY
```

The challenge image provides CUDA and cuPQC.

## Measured result

B=128, N=65,536, W=1,747: **57.198401 ms**, all three official cases passed.
Recorded rank: **14** on `fherma-gpu-rtx6000`.

Measured source: `f2c5f26adff384f5c38fd4d47d2a9cea13a53ed0`, path `odd-roots`.
Copies are measured separately: median H2D 131.776358 ms; D2H 64.320415 ms.
The prior [streaming version](streaming-rns/) scored 89.184818 ms with copies included.

[Challenge entry](https://www.fherma.io/kernels/polynomial-multiplication/implementations/abhijith-odd-root-warp-rns)

Warp transforms keep each small FFT in registers and exchange values through warp shuffles.
Shoup multiplication of fixed roots reduces ranked time by 4.70% from 67.041671 ms.

Odd-root butterflies absorb the outer twists. The official batch took 10.47% less time than the 63.889391 ms Shoup control. [Official result](odd-roots/result.json)

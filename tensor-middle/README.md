# Tensor-core middle transform

The middle stage multiplies two length-256 polynomials modulo each prime. Each warp factors a 256-point transform into two 16-point matrix products. Byte matrix instructions form exact dot products. A fixed matrix includes the byte place value of the input modulo the prime. Each output uses four byte sums and an exact reduction. The same forward matrix supplies the inverse by reversing the output indices. The existing outer passes supply the polynomial twist and final normalization.

One transform uses 32 warp matrix instructions per row. The two inputs and inverse require 96 in total, plus six byte-sum reductions per coefficient across those transforms. Padding each shared weight column to 20 words lets all lanes read a matrix operand without a bank collision. The experiment trades scalar butterfly work for tensor instructions, modular reductions and shared memory. Native timing measures that trade.

Radix decomposition for tensor-core NTTs is also studied in [WarpDrive](https://zhengfangyu.github.io/assets/papers/HPCA2025.pdf). This implementation uses the challenge kernel's existing byte reducer and preserves its residue layout.

# Defer modular reductions in the 32-point winner

Port the 30-bit lazy arithmetic to the selected 43.967 ms graph. The fused short forward butterflies carry values below 4p. The long cyclic middle carries values below 2p, including its per-column and per-row rotations. The forward phase multiplies one value below 4p and one canonical value. The middle pointwise product multiplies two values below 2p. Both products are below p times the Montgomery radix and need one final correction.

The ranked shape uses 60 primes instead of 58, which adds 3.45% more residue planes. The signed recovery uses the earlier biased quotient scheme so all output terms still fit the same fifteen CRT tensor steps. The final inverse keeps its canonical butterfly schedule. The input and output shapes and the stream topology follow the selected graph.

The exact tests check the larger Montgomery ranges, packed encoder, actual fused forward, cyclic products, inverse and actual biased decoder at both widths. The Blackwell resource check and ranked batch will determine whether cheaper modular arithmetic repays the extra planes. [Lazy butterfly reference](https://github.com/poulpy-fhe/poulpy/blob/v0.10.0/backends/poulpy-gpu-portable/src/ntt4x30/ntt.rs)

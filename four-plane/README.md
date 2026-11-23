# Four-digit polynomial multiplication

Split the 1,747-bit input at bits 437, 874 and 1,311. For each of 30 primes, evaluate the four digits at the powers of a fourth root of unity. The four-point transform needs eight modular additions or subtractions and one fixed modular multiplication. The input matrix includes each digit's scale and its offset within the first word. The GPU can read aligned words and skip a separate multiplication by that scale. The smaller shape uses 16 primes.

Three kernels run for each product. The first converts both inputs and runs the outer polynomial transform. The second multiplies the cyclic spectra. The last inverts the outer and digit transforms and reconstructs canonical coefficients. Its tensor tiles contain only residue words that contribute to each output warp. Four rounded quotient corrections fit in one word. All input-dependent arithmetic runs on graph replay.

| Ranked batch quantity | Two-digit single-stream control | Four digits |
|---|---:|---:|
| Transform planes per operand | 116 | 120 |
| Warp matrix instructions for both input conversions | 243,269,632 | 125,829,120 |
| Materialized encoded inputs | 2 | 0 |

The four-digit graph still writes and reads 22.5 GiB of intermediate values per batch. It uses 256 matrix-tile pairs per decoder block. The CPU tests check both full polynomial lengths, every constant matrix byte, the fused input and middle blocks, and the reconstructed outputs of a boundary block. Official full-batch timing determines the trade between simpler conversion, the extra planes, and reconstruction.

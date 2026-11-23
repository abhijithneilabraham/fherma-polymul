# Tensor-core reconstruction for eight folded digits

The official folded-eight run passed three cases in 77.003 ms. The sampled first product spent 260 µs in reconstruction. This candidate changes the decoder. It keeps the original scalar encoder and the three polynomial transform kernels.

Each block loads sixteen coefficients and inverts the eight digit frequencies per prime. A 64-by-64 byte matrix holds the cofactor bytes shifted by each input byte's position. An unsigned integer tensor instruction computes 16 digit rows and 8 output bytes at once. Two depth tiles cover all 16 four-byte prime residues. The warp collects four byte sums per output word. Signed integer carries and the CRT product correction recover the exact biased digit. Eight cooperating lanes finish each coefficient.

The centered quotient uses floor(2^48/p) for each prime. All reciprocals together underestimate the real sum by less than sum(p)/2^48. The exact sum is within bound/Q of an integer. Their sum is below one half, so adding a half and truncating returns the exact quotient. The accumulator fits in 52 bits before rounding. The public parameter generator and tests determine the bounds for this point.

The scalar reconstruction uses 256 wide cofactor products per digit. The new path uses 16 warp MMA instructions per tile of 16 digits. Its cost includes the input transpose, warp shuffles, output correction and carry propagation. Benchmark the full batch against folded-carry and folded-eight.

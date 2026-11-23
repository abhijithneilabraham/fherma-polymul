# Reduce the byte dot product with 32-bit arithmetic

The tensor encoder produces four nonnegative byte sums. The old reducer assembles their weighted sum into a 64-bit integer and takes the upper half of a 64-by-64-bit product. This version computes the low 32 bits and the high bits above bit 15 using shifts and additions. One unsigned 32-bit multiply-high estimates the quotient from a constant reciprocal. All three stream dependencies and all later kernels match the selected entry.

For a ranked tile each sum is at most 128 × 255². Let x be the weighted sum, h its integer quotient by 65536, and m the floor of 2^48/p. The estimate is the floor of h × m / 2^32. Its deficit before rounding is below 65536/p + h/2^32, which is below one for every selected prime. The resulting unsigned difference is in [0, 2p), so it fits in one word and at most one subtraction finishes the reduction.

The test checks the bound separately for each prime, random sums, all zero/maximum combinations, and both neighbors of exact multiples. The tensor-fragment test exercises the same reducer on actual accumulator outputs. Official timing measures whether the shorter arithmetic matters in the full batch.

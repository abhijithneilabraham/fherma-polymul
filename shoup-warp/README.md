# Shoup fixed-twiddle warp transforms

This fork preserves the mixed-radix warp schedule. Fixed butterfly roots use
ordinary values and precomputed 32-bit Shoup reciprocals. Variable products,
input conversion, phase factors and normalization retain Montgomery arithmetic.
Multiplying a Montgomery residue by an ordinary root with Shoup arithmetic
keeps the same Montgomery domain.

For a,w<p<2^31, the quotient estimate undershoots by at most one. The unsigned
32-bit difference equals a remainder below 2p, and one subtraction is exact.
Native correctness and performance are pending.

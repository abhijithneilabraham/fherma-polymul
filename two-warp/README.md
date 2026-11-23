# Cooperative two-digit reconstruction

This candidate keeps the selected three-stream schedule. Each warp recovers the two signed residue channels for one output coefficient and accumulates their quotient sums in registers. Each fixed-point reciprocal equals the floor of 2^48 divided by the prime. Across 58 primes the approximation error is below 58 × 2^31 / 2^48, or 0.000443. The centered CRT bound allows an error up to 0.25 before rounding changes.

The tensor stage distributes the final carry across a warp and uses ballots to propagate it. On the ranked shape the decoder executes three whole-block barriers, down from twelve in the selected source. The tensor instruction count stays the same. The independent CPU test runs the decoder's 224 lanes on a boundary block at each size, checks signed CRT values up to one quarter of the modulus product, and compares every output limb with a Python integer.

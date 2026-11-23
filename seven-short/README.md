# Reconstruct each 32-coefficient column in one block

Use a 32-point outer transform. The remaining cyclic product has length 2,048 at the ranked point and 1,024 at the smaller point. A block computes this middle product with row and column transforms of length 32 or 64. Two shared arrays hold its operands.

After the middle product, a single block owns all seven digit channels and all primes for 32 coefficients. It runs the outer inverse, recovers the seven signed digit terms and feeds them directly to tensor reconstruction. The ranked graph removes one temporary array and 7.875 GiB of logical reads and writes per batch. The larger middle transform adds shared-memory traffic and barriers. Whole-batch timing determines the tradeoff against seven-plane.

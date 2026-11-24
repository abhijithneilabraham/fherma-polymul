# Fold one forward butterfly into the pointwise product

Keep the 36.064465 ms encoder and the 31-bit prime basis. The inner transform stores two adjacent values in the same thread at its final butterfly. For the left operand, feed their raw sum and biased difference directly into the existing Montgomery product. The right operand stays in canonical form. This removes one normalization per left spectrum value across 973,078,528 ranked values. All output values are canonical before the inverse transform.

Let R = 2^32. Both unreduced values are below 2p, and the right value is below p. The product z is less than 2p², which is less than Rp because p is below R/2. The Montgomery correction adds t*p with t below R. Thus the wide sum is below 2Rp, which is below R². Its high word is below 2p and the existing final subtraction is sufficient.

The cooperative host test executes the production forward, middle and pointwise instructions with CPU substitutes for warp shuffles and byte matrix products. It checks both inner lengths against direct cyclic convolution. The edge test also covers residue representatives through 2p−1. Inspect the compiled middle and the official batch median.

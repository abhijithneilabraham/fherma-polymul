# Recover four digits and their CRT corrections in one warp

After the length-32 inverse, give one coefficient to one warp and one prime to each active lane. Each lane recovers all four digits, stores them for the matrix multiply, and keeps their four correction terms in registers. All lanes then run four hardware integer sums. The ranked block uses 30 active prime lanes and two zero lanes. This removes one block-wide barrier and the second set of shared residue reads from the previous four-digit decoder.

For each prime p, store floor(2^55/p). The high word of its product with residue t gives a 23-fractional-bit approximation to t/p. The error across 30 terms is below 0.000006. The reconstruction bound leaves at least 0.25 to a rounding boundary. One 32-bit integer sum per digit gives the exact rounded correction. The host fixture checks both entire output blocks against signed arbitrary-precision arithmetic. A second fixture verifies every quantized reciprocal, packed matrix and correction against the production basis.

Compare with the four-digit three-stream pipeline on the same CUDA graph. The two implementations request the same global spectrum bytes. The fused decoder changes lane ownership, shared accesses and synchronization.

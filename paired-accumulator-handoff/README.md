# Combine output contributions before the shared-memory handoff

In the 35.201212 ms reflected-lazy-lane control, each tensor instruction produces four byte-position sums per limb. The decoder writes all four 32-bit sums to shared memory, then one warp reads them and computes their weighted sum. The new decoder combines neighboring lanes in registers and sends one 64-bit sum to the carry warp. Both original block barriers remain. The shared tile shrinks from 7,296 to 3,840 bytes. The number of shared payload bytes written per group drops from 7,168 to 3,584. This trades shared-memory work for shuffle instructions and integer arithmetic. The register count, compiler output and official time decide the benefit.

The coefficient layout, input encoding, transforms, carry propagation and final reduction use the baseline implementation. Both supported widths have a cooperative block test against independent arbitrary-precision reconstruction.

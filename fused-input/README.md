# Fused input experiment

This candidate keeps the verified warp encoder's arithmetic and moves its result
straight into the first NTT stage within the same block. Eight primes share
one input tile. Their canonical residues remain in shared memory until the
unchanged cuPQC stage consumes them. No transposed global residue array is
written or read by this path.

The middle NTT stages, inverse stage, CRT, pure transfer hooks and exactness bound
remain unchanged. The runtime log reports conversion plus first NTT as one phase.
Input reads repeat across prime groups; cache behavior and serial transforms
within a block can offset the saved intermediate traffic. The official run passed all three cases at **87.725010 ms**. This regressed
against the retained control, so it is not selected.
Measured source: `d5f4b0e95639051cbe65c20e9f3423e5e439a333`.

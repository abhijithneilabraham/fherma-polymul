# Reconstruct two residue sums with three smaller matrices

The two input digits have place values 1 and B, with B equal to 2^880 in the ranked ring. Each prime contributes a term C(x+By) to the output. The existing kernel multiplies both residues by full-width weights. Split the fixed cofactor C into a+Bb, including the correction cofactor −P modulo q. Three shorter scalar products suffice across all primes and the correction term.

    L = sum(a*x)
    H = sum(b*y)
    T = sum((a+b)*(x+y))
    output = L + (B² mod q)*H + B*(T-L-H) modulo q

Each x+y fits in one 32-bit word. The individual byte sums fit in 24 bits. The shift by B is exact and B² modulo q is the small gap times a power of two. The largest possible nonnegative result has fewer than W+52 bits at both supported widths. One wide fold and an optional carry fold produce the canonical output.

For each tile of 32 ranked coefficients the tensor work changes from 840 matrix instructions to 720. Across 128 products that saves 31,457,280 matrix instructions (14.3% of the output matrix work). Staging three short products and recombining them adds shared-memory traffic. The full judge median measures that trade against the 35.201212 ms baseline.

The table test reconstructs all three independent byte matrices. The cooperative test runs the actual decoder on a complete tile and compares every limb with arbitrary-precision signed reconstruction. It covers both widths, values near the positive and negative reconstruction limits, and the final output addresses.

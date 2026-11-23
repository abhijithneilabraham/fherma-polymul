# Recover the centered CRT result inside the residue tile

Let p_i be the primes, M their product and c_i = M/p_i. For each digit, reconstruction takes weights r_i in [0,p_i) and subtracts alpha × M, where alpha is the rounded sum of r_i/p_i. The parent puts alpha in an extra tensor-matrix column. Two such columns push 120 ranked residue words beyond a 480-byte tile boundary.

For each digit, add p_i to r_i when i is at least alpha. Each adjusted residue remains below 2p_i. The total gains exactly (count − alpha) × M because p_i × c_i = M. Restore the centered result by adding the fixed bias −count × M × (1 + 2^shift) modulo q during carry propagation. The two digits have independent alpha values and share that one fixed bias.

The 120 ranked words now fit in fifteen tensor steps instead of sixteen. The 64 auxiliary words fit in eight instead of nine. Ranked inverse scratch falls from 32 MiB to 30 MiB and the denominator and transform layouts follow the 30-bit parent. The bias adds one fixed word per output limb and each warp broadcasts its two quotient values before adjusting its residues. The integer reference checks the generated bias, exact CRT identity and packed output matrix. Benchmark the complete ranked batch against the wider reconstruction.

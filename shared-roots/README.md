# Shared roots for the middle transform

Eight warps in each block run two forward transforms and one inverse over the same prime. Stage the 128 roots that each direction needs in 2 KiB of shared memory. Separate the multiplier and reciprocal into 32-bit arrays and store both directions at the same logical index. An XOR address permutation permits broadcasts and avoids shared bank conflicts for every butterfly stride. This adds one block barrier. The large phase tables and all boundary layouts remain those of the selected odd-root kernel.

Compare the official batch time and middle-stage timing with odd-roots. Fewer global root-load instructions can help even when the table already hits cache, but each lookup still needs shared instructions and address arithmetic.

# Reconstruct eight signed digits in one output matrix

The folded-carry control reconstructs each digit as a signed multiword integer, then shifts and adds eight of them. This kernel includes each shift in the constant CRT weights. One byte matrix reconstructs the full coefficient directly modulo the output modulus.

The 128 evaluation terms are ordered by prime and digit. Eight extra columns of input hold the centered CRT quotients, one per digit. Each digit quotient uses the existing 48-bit reciprocal bound. The resulting 544-byte dot products fit in signed 32 bits. Warp-wide carry masks then produce the 55 canonical output words. Input encoding and all polynomial transforms are the folded-carry versions.

The native comparison is against the 72.628 ms folded-carry run. The CPU fixture checks the actual inverse transform and digit quotients, the generated matrix and the warp carry implementation against independent integers.

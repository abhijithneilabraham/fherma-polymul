# Cancel the low word before Montgomery reduction

The two-digit transform uses the same modular product in the forward phase, the spectral product and the inverse phase. Each ranked batch calls it 3,892,314,112 times across the two forward operands, one pointwise product and one inverse phase. The control adds a multiple of the prime to a 64-bit product. This experiment subtracts a multiple with the same low word.

Let R = 2^32 and let x = a × b for canonical residues modulo an odd prime p below R/2. Set t = (x × p⁻¹) mod R. The low words of x and t × p agree. The signed difference of their high words is exactly (x − t × p) / R. It lies strictly between −p and p. The smaller unsigned value of the stored difference and the difference plus p is the canonical answer, including when x is zero.

This changes the word inverse in the per-prime table and the arithmetic of one helper. It adds no lookup, launch or buffer. Both output order and Montgomery scaling follow the combined two-digit parent. The experiment compares the generated Blackwell instructions in all three transform kernels. The official batch is the speed test.

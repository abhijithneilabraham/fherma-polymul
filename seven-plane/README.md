# Seven-digit polynomial multiplication

Split each input coefficient at bit positions ceil(iW/7). For each prime, the encoder reduces the seven digits in one tensor instruction per digit and applies a seven-point transform using eight fixed modular multiplications. The ranked point uses 18 primes and 126 data planes. The two polynomial phase tables together occupy 9 MiB.

Seven warps sharing one prime run the outer inverse polynomial transform. They exchange their results in shared memory, undo the seven-point transform, and put weighted digit terms directly in the order needed by reconstruction. One byte matrix combines the 126 residues and seven centered-quotient corrections into the final 55 limbs. The smaller point uses ten primes and 28 output limbs.

Only constants from the public point parameters enter initialization. The captured graph performs all input-dependent arithmetic. Full-batch official timing determines whether the cheaper conversion offsets the extra data planes.

# Compute both signs inside the three-kernel transform

The fused input block applies the paired two-digit encoder directly to its shared tile, then transforms that tile before writing spectra. Neighboring lanes compute both signs for two primes. The high-digit scale is already in the tensor weights. This removes a separate scale multiplication for each input coefficient and prime.

Four adjacent lanes now publish four different planes at the same row. In a plain plane-major shared array their stores would target different words in the same bank. XOR the row with one of 0, 8, 16 or 24 according to the sign and prime parity. Each warp store then uses all 32 banks once. The reader reverses the permutation while loading a contiguous row group. No extra shared array or block barrier is needed.

Compare the full batch with the three-kernel parent. The independent block test runs both operands and both sizes, checks the end of the last prime group, and compares every spectrum word for the target column with direct evaluation. It also enumerates the production shared offsets for each warp store. The compiler determines the register cost and the official run determines the elapsed time.

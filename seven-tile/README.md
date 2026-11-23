# Fuse seven-digit input conversion with the forward transform

Each ranked block owns one polynomial column and six primes. Sixteen warps load fragments of sixteen coefficients each. They reuse these registers for three prime pairs. For each pair, seven digit evaluations per prime pass through 14 KiB of shared memory into the forward polynomial transform. The block writes its spectrum in contiguous strips. The middle transform gathers eight neighboring rows to read that layout.

The ranked graph removes the encoded operand workspace and 15.75 GiB of logical intermediate writes and reads per batch. Three prime groups traverse each input column. If each group fetches the raw input once, that adds 6.875 GiB before caching. Fragment alignment, cache hit rates, register occupancy and the longer block lifetime all affect the measured result. Compare the complete batch with seven-plane.

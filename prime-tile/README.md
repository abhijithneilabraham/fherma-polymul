# Reuse encoder fragments across three prime pairs

Each warp loads a tile of 32 wide coefficients and multiplies it by six prime-specific sets of byte weights. It keeps 24 integer accumulators per lane. With 114 primes, it visits 19 tiles of six primes. The Shoup control visits 57 tiles of two primes. The number of tensor instructions and mathematical outputs is the same.

At length 65,536 this reduces source-level loads of the shared input fragments by two thirds. The remaining costs include loading weights, executing the byte matrix products, reducing each result and writing the residue planes. The compiler resource report and official timing determine whether the extra accumulators pay for themselves.

A 32-lane CPU emulator runs the same fragment-loading loops as the CUDA encoder. It checks all 5,472 residues in two 32-coefficient tiles against independent integer reductions, including the final partial group at 57 primes.

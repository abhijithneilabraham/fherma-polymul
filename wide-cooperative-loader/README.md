# Ten-warp cooperative encoder

The ranked encoder distributes 29 pairs of primes among ten warps. A block makes three passes over the pairs. The six-warp control makes five. The final pass in each configuration uses all but one warp. The shared matrix and total arithmetic have the same size. Additional threads can reduce the number of concurrent blocks, so use the official batch time to evaluate the tradeoff.

The same width saved 2.261438 ms on the earlier packed encoder. The direct cooperative-load control took 36.482772 ms.

# Combine a butterfly subtraction with its root multiplication

For canonical inputs a and b and a 31-bit modulus p, the quantity a+p−b lies between 1 and 2p−1. Feeding it directly into Shoup multiplication saves the comparison and correction that previously converted the subtraction back into a canonical residue.

For a word x below 2³² and a fixed root w below p, let m=floor(w·2³²/p). The difference between x·w/p and x·m/2³² is less than one. The approximate quotient therefore equals the exact quotient or falls short by one. The remainder fits a word because 2p fits a word. The existing final subtraction produces the canonical output. Butterflies with an identity root that bypass multiplication still reduce their differences directly.

The experiment uses the same primes, spectra, memory layout, graph and input loader as the 36.482772 ms control. Both the middle-stage transforms and the final inverse use the fused operation.

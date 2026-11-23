# Odd-root transforms in both dimensions

The 57.198401 ms control absorbs the input twist into its outer K-point transform. This version also absorbs the cross-stage phase into the 256-point middle transforms. Each middle row evaluates its own geometric sequence of roots. A pointwise Montgomery product and one prime-specific output scale produce the weighted ordinary residue required by CRT.

For physical outer row r, set u = 2*bitreverse(r,K)+1. At butterfly distance d and group g, the forward root is psi raised to d*(u + 2*K*bitreverse(g,128/d)). The mirrored row and mirrored group have exponent N minus this exponent. Since psi^N equals minus one, negating the mirrored forward factor gives the inverse. The inverse butterflies reverse their difference to apply that sign.

The control read one phase factor per coefficient for each input and once for the output. This version removes those reads and two per-element Montgomery products across both inputs and one output. It introduces a 256-entry Shoup table for each row and prime. The larger butterfly working set and its access pattern require a GPU measurement. At both supported sizes, CPU tests check every generated root, its reciprocal, the mirrored inverse and six complete products against independent negacyclic convolution.

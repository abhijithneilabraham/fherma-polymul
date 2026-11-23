# Reuse inverse roots across eight columns

The selected 32-point graph takes 43.967 ms. Its first sampled inverse spans 141.280 µs. The existing inverse launches 7,680 blocks per ranked product. Each 256-thread block processes four columns on eight planes.

Have each warp handle two neighboring groups of four columns. The two groups load the same root at each butterfly and run the same independent arithmetic. This cuts the inverse grid from 7,680 to 3,840 blocks per ranked product. It keeps the per-group four-sector loads and stores, the output offset for each residue, and the original graph. The added values live in registers and may change occupancy.

The direct fixture runs the paired inverse at the first and last block of both sizes and checks every live output and unused address. Compare the full batch with the same 32-point graph, using the stage sample to see whether the smaller block count helps the inverse.

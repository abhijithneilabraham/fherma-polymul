# Balance reconstruction work across eight warps

The fused inverse transform launches eight warps per block. The four-digit CRT step originally used seven of them. The packed output matrix has 28 tile lists at the ranked size. Some lists require twelve matrix pairs and others require eight. The heaviest warp processed 48 pairs before the block could reconstruct and store the output.

Each ranked warp now receives exactly 32 pairs. Four warps process two long lists and one short list. Four warps process four short lists. At the smaller size, the longest assignment falls from fourteen pairs to twelve. An immutable schedule maps each list back to its original output columns. All 256 ranked pairs remain in the product.

The schedule test reads the packed production matrix, verifies every output tile has one writer, and counts every warp's work. The decoder test runs all 256 lanes on boundary cases and checks all 2,656 output words across the two sizes. Actual batch savings depend on GPU scheduling and the fraction of time spent in this section.

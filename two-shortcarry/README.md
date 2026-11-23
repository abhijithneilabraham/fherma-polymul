# Finish the short-transform result with warp-wide carries

Port the warp-based CRT to the selected 43.967 ms graph. A block reconstructs 32 coefficients from one short-transform column. Its warp recovers both signed channels and sums their scaled reciprocals together. The 48-bit fixed-point table also serves the input reduction. The worst ranked quotient error from truncation is below `58 × 2^31 / 2^48`, and the centered CRT margin to a rounding boundary is at least one quarter.

The tensor stage stores two words per limb. Warp ballots propagate carries across 32 limbs at once. All four groups of ranked tensor outputs can finish before the same block barrier. This reduces the ranked reconstruction from twelve block-wide barriers to three. The tensor instruction count, memory layout at the inverse output and number of shared bytes stay the same.

The source fixture runs the complete 224-lane decoder. It compares signed values near the CRT range limit, all output limbs and all untouched addresses at both widths. Time it against the existing reconstruction on the same 32-point graph.

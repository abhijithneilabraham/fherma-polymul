# Absorb two butterfly levels into the integer conversion weights

Start from the 35.201212 ms widened reflected encoder. Divide the 32 transform rows into four groups of eight. Let v be the eighth root and I=v² be the fourth root. Multiply the conversion coefficients for the four groups by 1, v, I and vI. These are fixed constants for each prime and operand.

At the first butterfly, the two upper inputs already carry I. The second level needs v for its lower branch and v³ for its upper branch. Its inputs already carry v, so the lower branch needs no product and the upper branch needs only the ratio v³/v=I. Both outputs then have exactly the original scale. A later level consumes the widened sums and differences.

A tensor matrix tile needs sixteen rows with common weights. Two adjacent columns supply eight rows each. Each warp still executes eight tile accumulations per prime pair. The conversion tables grow from 116 to 464 KiB at the ranked size. The first two levels remove three of four fixed multiplications for every four input values, or 1,459,617,792 products across the ranked batch. Extra table traffic and shared-memory access patterns determine the actual speedup. Compare the full official batch.

The table oracle rebuilds all eight operand and quarter layouts from the prime roots. The cooperative host fixture checks every emitted word from both operands on both supported widths, and the full polynomial contract covers wraparound.

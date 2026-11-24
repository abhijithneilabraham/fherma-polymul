# Round the centered CRT correction with one integer warp reduction

Each of 58 prime residues contributes a fraction t/p to the correction. Store c=floor(2^55/p) per prime. A 32-bit high multiply gives floor(t*c/2^32), an integer approximation to 2^23 times the fraction. Each lane sums at most two values. One warp-wide integer reduction collects them. Add 2^22 and shift by 23 to round to the nearest integer. This replaces two floating conversions or multiplies per term and the five-level shuffle-and-add tree for each correction.

The approximation always rounds down. For any 0 ≤ t < p < 2^31 its deficit in ordinary units is less than 2^-23 + 2^-24. Even 114 terms miss the true sum by less than 0.000021. Each exact signed reconstruction lies within one quarter of its CRT representative, so the margin to a rounding boundary is at least 0.25. The integer sum and rounding bias fit 32 bits. Every corrected digit and output word therefore remains exact.

The ranked batch has 8,388,608 coefficient rows and two corrections per row. Each correction now uses one 32-bit hardware sum. Compare the full batch with the 36.064465 ms lane-owned encoder control.

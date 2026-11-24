# Combine row rotation with both reflected encoder weight tables

Keep the 35.442676 ms reflected-phase control. It has separate byte weights for its inputs because B carries the common normalization. Split each table by the half of the input row tile. For rows 16 through 31, include the fourth root used by the first outer butterfly in the constant byte weights. That butterfly then takes just a sum and a difference.

The ranked byte weights grow from 116 to 232 KiB. Both inputs still use the same signed residues, the same matrix instruction count, the same four-column loads, and the same final phase table. Across the ranked batch this removes 973,078,528 fixed modular products. The separate weight audit reconstructs all four tables from the primes, normalization and fourth roots. The cooperative tile test checks the complete forward output, and six full products exercise the reflected domain boundary.

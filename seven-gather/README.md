# Pack the live CRT terms for each output warp

Each output warp selects the residue words with a nonzero coefficient in any of its eight byte columns. The host packs up to eight such words per tensor tile and records their shared-memory word indices. Each warp visits a contiguous range of its own packed tiles. Final-tile padding carries zero weights.

A rounded quotient fits in one byte. Each residue is less than its prime, so its contribution to the fixed-point quotient is less than one. The rounded sum is at most the prime count, which is 18 on the ranked shape. The seven quotient bytes occupy two shared words. The host weights select the low byte from each original quotient term.

| Width | Original dense tiles | Whole-zero-tile skipping | Per-warp packing |
|---|---:|---:|---:|
| 868 bits | 140 | 86 | 59 |
| 1,747 bits | 476 | 225 | 168 |

The ranked decoder issues 64.7% fewer matrix instructions than the dense version and 25.3% fewer than skipping whole zero tiles. The extra shared-memory address indirection can reduce the timing gain. The CPU reference reconstructs the full compact matrix from the packed tiles and compares every byte with the independent dense coefficients. The native decoder test also covers signed quotient recovery, canonical subtraction and boundary writes.

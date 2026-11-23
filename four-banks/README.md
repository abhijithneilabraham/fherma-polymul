# Shared-memory bank layout for the four-digit forward transform

The fused forward block converts a tile, transforms it, then gathers four columns for a contiguous global write. In the parent layout one gather instruction addresses four distinct words in the same shared-memory bank. Matrix fragment loads and digit stores each address two. The new layout uses a column-dependent row permutation for the transform tile and a four-word exchange in every other group of four ranked input rows. The smaller input uses a compact 28-word stride.

| Maximum distinct words per bank per warp instruction | Parent | Swizzled |
|---|---:|---:|
| Input store | 1 | 1 |
| Matrix fragment load | 2 | 1 |
| Digit store | 2 | 1 |
| Transform tile access | 1 | 1 |
| Output gather | 4 | 1 |

These counts come from enumerating the addresses at both problem sizes. The complete fused-block CPU test checks both operands and every digit plane against direct evaluation. The official comparison with the four-digit parent measures the cost of computing the new addresses and its effect on batch time.

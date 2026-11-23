# Remove shared-memory bank conflicts when staging CRT residues

In the selected 32-point graph, a warp reads one residue for each of 32 coefficients. Its global load is coalesced. The following shared store uses eight banks, four lanes per bank, because consecutive coefficient rows are 124 words apart. The auxiliary width has the same conflict at its 68-word stride.

XOR the two low bits of the logical word index with the coefficient row divided by eight. A staging warp then addresses each bank once. Tensor fragment loads and quotient reduction still address distinct banks. Preserve the logical order for sign recovery, quotient summation and reconstruction. The new layout adds address arithmetic to some instructions, so the native batch will determine its effect.

This touches 1,006,632,960 staging stores per ranked batch at the algorithm level, including the zero padding words. Shared allocation, tensor instruction count and graph structure follow the selected entry. The bank test enumerates every address and both widths. The full decoder test compares the actual words against Python integers. [Selected control](../two-shortouter/README.md)

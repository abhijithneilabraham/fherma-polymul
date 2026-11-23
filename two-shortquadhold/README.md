# Keep four-column coalescing with half the shared memory

Load all four columns through full sectors and perform each short inverse once. Each lane writes the first two of its four reconstructed rows to shared memory. It holds the signed residues for the last two rows across CRT of the first half. The ranked shape holds up to 36 words per lane before compiler allocation. When the first half releases shared scratch, the warp publishes the remaining residues and the block reconstructs the second half.

The ranked block uses 46,336 shared bytes, half the original four-column allocation. Two blocks can fit by shared-memory capacity. Register residency and spills still need Blackwell qualification. The 16-row tensor tiles reload fixed weights for each half and introduce additional barriers. The boundary keeps its 7.25 GiB request count and the inverse still executes once. Native timing will show whether the extra residency repays the additional output work. [Four-column control](../two-shortjointquad/README.md)

The host fixture runs all 224 lanes, checks both halves of all four columns and compares every result word with the independent wide-integer reference. The address check covers every middle store and inverse load at both widths.

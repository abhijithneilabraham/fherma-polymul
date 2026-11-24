# Pack reconstruction weights in warp order

Each reconstruction warp needs two 32-bit weight words per lane. The control reads from eight columns, with a 480-byte stride at the ranked size. Setup now stores both words together in the order the 32 lanes consume them. A full warp can read the fragment with one aligned 256-byte access across the lanes.

One ranked batch executes 128 × 2,048 blocks × 4 groups × 7 warps × 15 depth tiles, or 110,100,480 warp iterations. The old loads of the two halves touch the same sectors, so the reduction in load instructions can exceed the reduction in device memory traffic. Padding the last four columns grows the ranked table from 105,600 to 107,520 bytes.

The CPU decoder test packs the production table, emulates the matrix instructions and compares every output limb of a boundary block with independent integer arithmetic at both sizes. Compare the linked Blackwell reconstruction kernel and full-batch median with the 36.482772 ms cooperative-load control.

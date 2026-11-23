# Coalesced reads for the final transform

The three-kernel parent writes four columns side by side. Each final-transform warp consumes one column, so 32 consecutive lanes touch 16 memory sectors. Here a middle block retains eight completed rows in shared memory and writes one full 32-byte sector per column. The final warp reads 32 consecutive words in four sectors. The number of transform operations and the intermediate byte count stay the same.

At 128 products and 126 planes, the last stage requests 3.9375 GiB of words. The sector count for its input loads falls from 15.75 to 3.9375 GiB before cache reuse and compiler effects. Middle blocks process eight rows in four pairs. The ranked middle requires 99,328 shared bytes per block and issues 504 blocks per product. Compare the middle-time cost, the decoder-time saving, and the complete batch with the three-kernel parent.

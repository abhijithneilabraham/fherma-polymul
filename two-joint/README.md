# Finish the inverse transform inside reconstruction

The middle kernel owns 32 neighboring frequency rows. After the cyclic product and phase correction, it performs the first three inverse butterfly stages at the ranked size, or two at the smaller size. The remaining work consists of independent 32-point inverse transforms. Each reconstruction block reads two columns, finishes those transforms and converts 64 coefficients to canonical wide integers with the shorter carry reducer.

One 32-byte sector holds four coarse frequencies from each of two adjacent columns. Each middle-store warp fills four sectors, and each reconstruction-load warp reads four sectors. The address test enumerates every intermediate word at both sizes. The middle-block simulator checks dense cyclic products and the first inverse stages at a boundary block. The 224-lane reconstruction simulator checks both output columns against independent large-integer results.

The ranked batch avoids writing and rereading 116 planes of weighted residues, or 7.25 GiB before cache reuse. Two equal-sized middle buffers allow consecutive products to overlap. The larger 1,024-thread middle block and the extra shared memory in reconstruction may offset the saving. Compare the full batch with the combined two-digit control.

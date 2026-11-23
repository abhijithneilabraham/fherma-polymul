# Three-kernel two-digit polynomial product

The first kernel converts a tile of wide coefficients and immediately applies its outer transform. The second reads both spectra in complete sectors, performs the cyclic product and starts the inverse. The third finishes independent 32-point inverses and reconstructs the wide output in the same block. Alternating forward and partial-inverse buffers let all three stages overlap across products.

At 128 products the input fusion removes 14.5 GiB of logical intermediate traffic and the output fusion removes 7.25 GiB. The remaining two spectrum boundaries move 21.75 GiB. Reading each wide input once and writing the final result once account for another 5.15625 GiB. Actual device-memory traffic depends on cache reuse. The middle kernel transposes its operands one at a time through a 32 by 257 shared tile to preserve contiguous reads without holding both tiles at once.

Independent CPU tests verify both operands of the fused input stage, every spectrum address, the cyclic product and partial inverse on a boundary block, all reconstructed words on paired boundary columns, and the event dependencies at batch sizes up to 128. Individual input and output fusion candidates isolate each change against the same two-digit control.

# Input-tile reuse across all primes

Each block owns one input operand and one coefficient column. It loads the wide coefficients into shared memory once. Four warps per 32 rows convert eight primes at a time. Up to eight warps apply the odd-root transform before the block reuses its residue buffer. The middle stage reads eight adjacent frequency rows at a time and transposes them for its warp transforms.

The ranked batch uses 512 blocks per product, each with 1,024 threads and 68 KiB of dynamic shared memory. Fusing the two stages removes one write and one read of both encoded operands, or 14.25 GiB of logical transfers per batch. GPU time also depends on how efficiently those blocks use the machine and how much of the original traffic hit cache.

The reference is the odd-roots implementation. Only a native timed run establishes the effect on the complete batch.

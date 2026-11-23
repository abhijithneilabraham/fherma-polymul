# Tensor-core encoding of eight folded digits

Each block splits 32 coefficients into 256 digit rows and pads each row to
32 bytes. Eight warps cover 16 primes. Each warp uses one unsigned m16n8k32
matrix operation for each 16-row tile, reduces the four byte-dot columns
exactly, and transforms eight twisted digits with warp shuffles. One block
writes 128 frequency planes for 32 coefficients. The inverse path follows
folded-eight. Compare the full batch and the measured encoding time.

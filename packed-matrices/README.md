# Packed constant matrices

The encoder and reconstruction kernels load each lane’s two MMA weight words
as one aligned 8-byte pair. Setup packs the constant matrices in warp order
and zero-fills inactive columns. A warp reads one contiguous 256-byte tile.
Arithmetic, coefficient layout and the NTT schedule follow the measured warp
baseline. Benchmark the complete batch to determine whether fewer load requests
outweigh the larger padded tables.

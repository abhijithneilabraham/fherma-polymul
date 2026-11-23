# Transform one group of primes with 64 input rows at a time

The instrumented two-digit graph spends 48.17 of 49.18 ms on its conversion and forward stream. The signed three-kernel candidate fuses those stages, but each ranked block uses 1,024 threads and 96 KiB of shared storage. This variant tests whether shorter blocks can deliver that fusion more efficiently.

One block owns one column, one operand and up to twelve primes. Its eight warps load 64 rows, split them into two digits and save the paired evaluations. After all chunks are evaluated the same warps run the complete transform from the saved residues. Consecutive block IDs cover the same raw column with different primes. The final short group has ten ranked primes and six on the smaller shape.

| Ranked forward block | Fused signed parent | Chunked input |
|---|---:|---:|
| Threads | 1,024 | 256 |
| Shared memory | 96 KiB | 42 KiB |
| Blocks per product across both operands | 512 | 2,560 |
| Times each raw coefficient is requested | 1 | 5 |

The new arrangement requests an extra 13.75 GiB of raw inputs per ranked batch before caching. Both variants write the same spectra and execute the same number of tensor products. It trades input reuse for smaller blocks and more independently schedulable work. Each block uses eight whole-block barriers on the ranked shape. The block test runs every prime group at both sizes and compares the assembled spectrum with independent direct evaluations. Measure the full batch against the fused signed parent.

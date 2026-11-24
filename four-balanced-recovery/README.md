# Balance the four-digit output stage after fused recovery

The measured four-digit pipeline takes 39.251932 ms. Its first-product output sample is 226.592 µs while other products can overlap it. A full-batch win over 36.064465 ms needs 3.187467 ms, or 24.902 µs per product at unchanged contention.

Use the earlier verified sparse matrix schedule on the new pipeline and fused integer correction. It redistributes the 256 matrix pairs over all eight warps. Each ranked warp runs 32 pairs. The former longest warp ran 48 and one warp did no matrix work. At the auxiliary size the longest assignment drops from 14 to 12. Each output tile still has one writer. Global read and write layouts stay the same.

The schedule test reads the actual packed matrices, counts every pair and verifies unique output ownership. The cooperative decoder test runs all lanes on both widths. Compare the official result with the pipeline and the fused-correction arm. A 33% shorter longest assignment applies only to the matrix section and is not a forecast for the batch.

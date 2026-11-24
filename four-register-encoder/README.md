# Keep each four-digit evaluation in its owning warp

The measured four-digit parent took 57.253501 ms. Its first-product forward sample was 242.112 µs. It used a 768-thread ranked block and exchanged the encoder output and transformed values through shared memory in five prime waves. Every wave required three block barriers.

Here six warps split the raw coefficients once into four packed digit tiles and meet at one initial barrier. A warp then owns two primes and both 16-row halves of four columns. Each neighboring pair of lanes completes a different row of the byte dot. The 48-bit reciprocal reducer returns each digit residue. The warp computes all four evaluation channels and their outer butterflies in registers. Four consecutive output columns share one aligned store per lane.

The ranked input still needs 125,829,120 matrix instructions across the batch. The packed input uses 32 KiB per block. The shared bank map permutes groups of four words so that all eight matrix rows have distinct 16-byte bank groups. The CPU block emulator checks both complete operand tiles against direct point evaluations at every prime, row and digit.

The graph still runs all three stages on one stream. This isolates the encoder before the separate pipelined comparison.

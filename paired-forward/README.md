# Paired forward transforms

Both operands use the same prime, root and cross-stage phase for a given column. A warp now owns both columns and loads each root and phase once. It computes the butterflies for both operands, then transposes each into the original spectrum layout.

For the ranked batch this removes 3.5625 GiB of requested phase loads before cache effects and halves the forward grid. Each block holds two output tiles in shared memory and each thread carries both sets of coefficients. Compare the compiled register count and the native forward and batch time with odd-roots. This change affects the transform after input conversion.

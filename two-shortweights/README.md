# Reuse the encoder's fixed tensor weights across its input tile

A warp in the packed 32-point encoder handles four columns of 32 coefficients. It evaluates eight 16-coefficient tiles for each prime pair. The fixed right-hand matrix is the same for all eight tiles.

Pack the four right-hand words for each lane into one aligned 16-byte group. Load one group per tensor inner step before the column and half-row loops. The same registers then supply every tile for that prime pair. There are at most sixteen weight words per lane live at once on the ranked shape, and eight on the smaller shape. The source requests one eighth as many weight bytes per prime pair before caching. That does not tell us how much code motion the compiler already performed or how many bytes reached device memory.

The tile loads the same coefficients and performs the same tensor operations as the packed parent. The two tensor fixtures check the generated weight table and evaluate the full forward output at both sizes. Inspect the compiled global loads, register count and spills before timing the complete ranked batch.

# Keep the forward spectra in a 24 MiB window

The chunked input design starts one block per column, operand and prime group. Its middle kernel waits for the entire product's forward spectrum. At the ranked size two full forward slots use 116 MiB, before inverse data, raw inputs, and root tables compete for the same cache.

This graph starts the middle transform whenever one group of at most twelve primes is ready. The forward stream can fill the next slot while the middle stream reads the current slot. Two group slots use **24 MiB**, a reduction of 92 MiB. Each raw input still gets requested five times per ranked product, and both kernels perform the same arithmetic. Full partial-inverse storage remains so reconstruction can consume every prime at once.

This trades a smaller live cache footprint for eight additional kernel launches per ranked product. The complete batch adds 1,024 graph launches. The forward and middle tests check the layout at both sizes and include the short final group. The graph test checks both slot lifetimes across group and product boundaries. Compare with the chunked parent on the official GPU to learn whether cache reuse repays the additional launches.

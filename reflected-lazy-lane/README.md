# Delay forward reductions on the shared phase table

The official control passed at 35.442676 ms. Keep both reflected input weight tables, the aligned four-column phase loads and stores, and the exact inverse phase lookup. Carry 31-bit forward sums and differences in the interval [0,2p) until the next fixed-root multiplication needs a canonical value. The final phase multiply accepts the same widened interval. This changes only the butterflies in the input encoder.

The separate lazy encoder on the previous layout kept 73 registers and reduced its static instruction count from 1,960 to 1,672. Use the full ranked batch for the interaction between that change and the reflected layout. CPU tests check the actual tile, 2,523,136 arithmetic cases, and six complete dense-by-sparse polynomial products.

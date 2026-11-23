# Reuse each prime's transform tables on the selected graph

The 45.501 ms control stores duplicate rotations and polynomial roots for its positive and negative evaluation planes. These tables depend on the prime, which both signs share. Store one copy and dispatch neighboring blocks on the signs and operands that use the same constants. Keep the scalar butterflies, register transpose, tensor conversion, tensor reconstruction and three-stream dependencies of the selected graph.

The two ranked phase arrays shrink from 58 to 29 MiB. Both scratch allocations and both signed spectra keep their original size. The same number of blocks and warps run, and every output word has the original address. Table reads still occur in each warp. Any traffic savings depend on cache hits.

The tests check the tables and both sign and operand mappings, all prime matrices, both support widths, and the original graph dependencies. Compare the full ranked batch with the selected graph to measure the effect of reusing the constant working set. This ports the earlier table-sharing idea from its serial graph to the measured three-stream layout.

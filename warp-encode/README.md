# Warp encoder experiment

This variant starts from verified resident-copy source `b271b1ad` (77.362807 ms).
Only the two input encoders use a new kernel. The NTT, CRT, transfer hooks,
RNS bounds and graph completion remain intact.

Each four-digit residue uses accumulator values in a neighboring lane pair.
Warp shuffles replace the shared result tile and its two block barriers per
prime group. A warp skips a final group with no valid prime pair. Shared input
bytes remain immutable after the initial block barrier. All 32 lanes execute
both shuffles before any lane-specific output guard.

The arithmetic is exact unsigned-byte MMA with the original Barrett reduction.
All three official cases passed at **77.119967 ms**, rank **17**.
Measured source: `34bdb4d071286d04a5076615601074689fd61b34`.
Pure transfers are timed separately. This is 0.3139% below the prior recorded
77.362807 ms. The small change has not had a paired noise study.

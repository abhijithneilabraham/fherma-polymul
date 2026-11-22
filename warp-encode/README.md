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
GPU correctness and performance are unmeasured until an official result lands.

# Overlap reconstruction with the next product

The 57.198401 ms odd-root graph finishes one product before starting the next. In this version, a second stream reconstructs each product after its inverse transform. The primary stream can start encoding and transforming the next input pair immediately. It waits for the previous reconstruction immediately before the inverse stage reuses the weighted residue buffer. The final join includes the last output in the timed graph.

All kernels, operand bytes, output offsets and arithmetic are those of the verified parent. Both streams share one workspace. The dependency-order test checks the actual graph-building source for five batch sizes. The optional timing metric measures the intersection of the first CRT stream interval and the second product's encoding interval. Device event intervals can include scheduling delays. The official batch time decides whether overlap improves throughput.

At the recorded 13th-place threshold this experiment needs to recover 1.143 ms per batch, about 9.0 µs per adjacent product pair. The parent measured 106.528 µs for its first reconstruction. The amount that can overlap across the batch depends on register, cache and execution-unit contention.

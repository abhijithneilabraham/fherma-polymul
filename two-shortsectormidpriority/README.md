# Give the middle transform first access to available blocks

The full-batch diagnostic on the fused control found 5.36 ms with only
the middle transform active. The input stream still had the longest
cumulative span. This test gives newly runnable middle blocks the highest
available scheduling priority to see whether they release the two forward
slots sooner and finish with less exposed work.

CUDA captures each kernel's priority from its stream. Instantiate the
graph with the node-priority flag so replay uses the captured values.
The scheduler can favor ready work as resources become available. The
controller retains the sector-layout winner's kernels, launch sizes,
buffer ownership, graph dependencies and exact arithmetic. Compare the
complete batch with 40.136425 ms to detect any benefit or extra producer
delay. [Sector-layout control](../two-shortsector/README.md) · [CUDA graph flags](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/group__CUDART__GRAPH.html)

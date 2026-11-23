# Keep the encoded inputs in the GPU's L2 cache

The encoder writes two 29 MiB residue arrays per product on the ranked point. The forward transform reads them and writes a different spectrum buffer. Across the batch the first boundary moves 14.5 GiB at the kernel interface. Its actual device-memory traffic depends on cache eviction.

The CUDA graph gives accesses inside the encoded-input buffer a persisting cache hint. Its window covers exactly the 58 MiB allocation. No raw inputs, transform outputs or model data receive the hint. The GPU caps the requested reservation. If the reservation is smaller than the window, the hint applies to the corresponding fraction of accesses. A device without this capability uses the ordinary graph.

This isolates a data-movement change on the combined two-digit control. All final outputs and graph dependencies follow that control. The cache policy runs during setup and the input address never changes between graph replays. Every replay computes the full operation on fresh inputs. The policy can reduce eviction at this boundary, but reserving cache can increase eviction elsewhere. The official batch will measure the balance. [CUDA cache policy](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html)

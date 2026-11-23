# Keep the ranked transform roots in L2

The forward, middle and inverse stages reuse the same 979,968 bytes of NTT root pairs across 128 independent products. Two forward workspaces alone hold 116 MiB. Give graph kernel accesses to the fixed roots a persisting L2 hint. Keep cache policy for other allocations at the default. The auxiliary graph uses the default policy.

Configure the window once after graph capture. If the device reports no capacity, use the original graph. Restore any reservation changed by this instance during cleanup. The input, scratch and output layouts and the arithmetic follow the selected 43.967 ms graph. Any gain requires root misses on that graph. The retained data can also evict scratch, so the full native batch will determine the effect. [CUDA L2 cache controls](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html)

The public batched CUDA source uses a root cache window on a different transform. This version applies the idea to our existing short-outer layout and reuses our earlier graph policy approach. [Public implementation](https://github.com/Wiredshark/polymul-batched-cuda/tree/02b924174e9cf3d0b13b5a68e6bd90639b002a7c) · [Earlier cache experiment](../two-cache/README.md)

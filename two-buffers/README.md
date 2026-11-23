# Give alternate inverse transforms separate output buffers

The selected three-stream graph waits for the preceding wide reconstruction before its inverse transform can reuse the shared output buffer. This variant alternates two buffers. An inverse may proceed while the previous product's reconstruction still reads the other slot. It waits for the reconstruction two products back before reusing its own slot. The single-product case allocates only one buffer. A final stream join covers the last reconstruction.

The ranked batch uses 30 MiB of extra device storage. The arithmetic and kernel geometry match the 48.835 ms control. More concurrency and a larger cache footprint can have competing effects. The dependency test checks both slots, all other workspaces and the final join at batch sizes 1, 2, 3, 57 and 128.

# Profile the complete 32-point batch

Time the fused input, cyclic middle, short inverse and CRT for every product. Record each start after its stream waits for reusable storage, and record the end before it publishes that storage. Use the original kernels, workspace sizes and stream dependencies.

The report sums each stage's spans and records how much time each stage runs alone. It also accounts for time with two or three active streams and time with no active stage. These are CUDA event intervals. An active interval can include scheduling delays or contention. Compare the profiled batch with the 43.967 ms selected result to quantify the instrumentation cost.

The source graph check proves both spectrum slot lifetimes and the final join. The timeline check covers stages overlapping across three streams and rejects a premature overwrite. Use this result to decide which stage has enough exposed time to target next.

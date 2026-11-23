# Use integer tensor cores for the short middle transforms

Keep the measured 39.602632 ms graph, its exact arithmetic and all
harness interface and graph scheduling. Replace the three scalar length-256 transforms
within the middle block with six exact radix-16 tensor passes. Reuse
the operand transpose workspace as warp scratch once all lanes have
loaded their operand rows.

The ranked block uses 37 KiB of shared memory, up 5 KiB for the shared
tensor weights. It adds two block barriers around scratch reuse. There
are 364,904,448 warp matrix instructions across the ranked batch.
The change removes the corresponding scalar inner butterflies and
adds shared accesses and exact byte-dot reductions. The surrounding
input and output stages also use tensor cores, so full-batch timing
must establish whether the new middle makes the pipeline faster.
[Winning control](../evidence/optimization-two-shortjointmin/outcome.json) · [Tensor transform](../two-tensor-lanes/README.md)

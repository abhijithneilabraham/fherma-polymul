# Trace the measured four-digit pipeline

Time the input, middle and output for each of the 128 products. The arithmetic kernels, launch shapes, buffers and wait dependencies match the 39.251932 ms official revision. Six external CUDA events per product record stage start and completion. The report measures overlap, exclusive stage time and the last dependency before each stage starts.

This trace can identify whether reconstruction, the middle transform, or a slot reuse wait holds up the four-digit batch. Compare the instrumented median with the clean revision to quantify event overhead. Read exclusive times alongside inclusive times because concurrent stages share the same GPU.

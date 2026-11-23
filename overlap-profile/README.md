# Full-batch overlap profile

This diagnostic uses the selected two-stream kernels. Eight timestamps bracket input conversion, the three transform stages and output reconstruction for each product. The inverse start follows the wait for the preceding reconstruction. The reconstruction start follows its own input dependency.

Durations are event spans on the last invocation. Intersecting each reconstruction with the main-stream spans gives the overlap for each stage. Subtract these intersections from the sum of all stages to obtain covered wall time. The remainder includes gaps between the markers. CUDA event spans include scheduling delays. They do not measure active warps or establish how fast the graph would run with one stage removed. Compare the instrumented median with the 54.709003 ms control to assess the cost of the markers.

# Combine the shorter carry with the revised input and middle transform

The control is the official 47.728620 ms three-stream graph. Use its output reducer with the same early carry termination. Read each wide input word once, compensate digit padding in its evaluation roots, and use a bounded 32-bit reciprocal for the byte-matrix dot products. Inside the length-256 cyclic product, keep the three innermost butterfly stages in registers. The surrounding stream dependencies and intermediate array layouts follow the control.

The separately compiled encoder dropped from 488 to 264 static instructions and uses 38 ranked registers. The separately compiled middle kernel dropped from 2,296 to 1,880 static instructions, from 120 to 84 shuffles and from 48 to 40 registers. Both ranked kernels are spill-free in those builds. The combined official run measures their effect on the full pipeline.

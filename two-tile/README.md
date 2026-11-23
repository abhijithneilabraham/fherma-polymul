# Fuse two-digit input conversion and the first transform

A block reads one input column, splits every coefficient into two digits and keeps both digits in shared memory. Groups of twelve primes produce up to 24 transform planes at a time. The block transforms each plane before it writes the spectrum. The middle kernel gathers eight adjacent rows and consumes those spectra with the register-transposed cyclic transform.

At the ranked point this removes one write and one read of both encoded operands, or 14.5 GiB per batch before cache reuse. The block uses 96 KiB of shared memory and 1,024 threads. The two spectrum buffers occupy the space previously allocated for one encoded pair and one spectrum pair. The next product can start on the second slot while the middle kernel reads the first. Event dependencies guard every later slot reuse.

The comparison source is the combined two-digit graph with the shorter carry. The CPU test runs both sides of the fused kernel on synchronized virtual lanes. It checks each output against direct evaluation from the original wide coefficients at every prime, both digit signs and every local frequency. It checks all 98,304 middle-input gather addresses across both problem sizes. The full ranked batch measures whether the saved transfers offset the large block and smaller number of independent blocks.

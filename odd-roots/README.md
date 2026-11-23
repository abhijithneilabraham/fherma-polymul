# Odd-root warp transforms

The first warp transform evaluates at odd powers of a primitive root. Its inverse uses the reciprocal roots. This absorbs the coefficient twist into the butterflies. The two cross-stage phase factors also apply the remaining column twist, normalization and CRT weight.

Relative to the Shoup control, the ranked graph removes two forward table reads and one inverse table read per polynomial coefficient per prime for each product. It also removes six Montgomery multiplications across the two inputs and one output. For the batch of 128, this removes 10.6875 GiB of logical table loads. Actual device traffic depends on cache reuse. The shared serial initializer still allocates the original tables. The ranked kernels never read them.

The native batch time and correctness require an official run.

# Eight-digit polynomial multiplication

The ranked point splits each 1,747-bit coefficient into eight digits of 218 or
219 bits. Sixteen auxiliary primes implement cyclic convolution across digits
and negacyclic convolution across polynomial coefficients. Twists encode the
identity 2^1747 = 9699327 modulo q. Each batch graph encodes and transforms
both operands, multiplies their spectra and reconstructs canonical coefficients.
Other points use the wide arithmetic fallback.

The CPU convolution reference matches GMP on a dense full-length product.
The compiled digit and CRT helpers pass boundary and random cases against
Python integers. The official GPU verifier measures the complete CUDA path.

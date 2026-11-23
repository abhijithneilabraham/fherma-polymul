# Transform operand A during conversion of B

The 52.334 ms control converts both inputs before transforming either. The new schedule launches the forward transform for A when A is ready. B conversion continues on the original stream. The two forward launches write disjoint halves of the original workspace. The middle transform waits for both. The next product waits for the corresponding input and spectrum buffers to become available.

The graph uses the same CUDA kernels and the same polynomial buffers. It adds one forward launch per product. The first-product event intersection reports how much the A forward span overlaps the B conversion span. The official median measures whether earlier execution offsets the launch overhead and contention.

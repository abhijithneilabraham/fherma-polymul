# Load the input matrix fragments with one warp instruction

The signed fused encoder loads the four words of each tensor operand separately from shared memory. Blackwell supports a cooperative load that returns four fragments to every lane. The input rows are aligned to 16 bytes. Groups of eight rows span 32 banks without repeating a bank. The load returns the same four 32-bit words in the order required by the integer MMA.

This replaces eight scalar shared-memory loads per inner-loop iteration with two warp matrix loads. Address arithmetic and the hardware's shared-memory transactions still matter. The number of tensor products, the packed prime weights and all subsequent computations come from the fused signed-input parent. [CUDA PTX matrix-load semantics](https://docs.nvidia.com/cuda/archive/11.3.0/parallel-thread-execution/index.html#warp-level-matrix-load-instruction-ldmatrix)

The CPU test uses the documented lane-to-fragment mapping to interpret the actual row addresses, checks alignment and bank usage, and compares complete spectra for both shapes. The Blackwell qualification confirms that the compiled forward kernel contains the cooperative load. Compare the full batch with the fused signed-input parent.

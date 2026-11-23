# Overlap phase loads with the forward butterflies

The register-local forward transpose scored 46.311 ms, while the selected graph took 45.603 ms. The first forward stage was 2.080 µs slower in the sampled product. The register prefetch moves phase reads ahead of the block barrier, but grows the ranked forward kernel from 40 to 47 registers and from 856 to 1,040 static instructions.

This version starts 16-byte asynchronous copies into a separate shared tile. The base column is aligned to eight words and the two vectors within each row start at word zero or four. Every thread issues two copies at the ranked size and one at the auxiliary size. The threads execute their register butterflies while the copies progress, then wait at the existing block barrier before reading phases. The output loop and its address order are the same as the register-local parent.

The shared phase tile adds 8 KiB per ranked block. The source-level oracle checks the same address permutation with synchronous copies. Check the Blackwell resource report and the compiled copy and wait instructions before scheduling a timed run. [Asynchronous copy and wait semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async)

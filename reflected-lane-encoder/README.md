# Keep four phase factors together in the encoder

Each lane keeps one modulus and one sign during the 32-point transform. Its four columns stay together until the result is ready. A single aligned 16-byte load reads the four phase factors, and an aligned 16-byte store writes the four products. The table starts on a CUDA allocation boundary. Every prime stride, tile stride and frequency row begins at a multiple of four words.

The reflected table supplies both inputs and the middle stage. The second input carries the normalization in its byte weights. This preserves the exact polynomial product and halves the large phase allocation. The cooperative host test runs both operands and both sizes. The convolution test checks that the three phase factors agree with direct negacyclic multiplication.

Compare against the separate lane-owned encoder, the reflected table, and the 36.482772 ms control. The CUDA compiler can raise register use when it holds four transformed columns at once. Only the official batch measures the combined effect.

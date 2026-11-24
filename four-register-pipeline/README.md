# Overlap four-digit conversion, cyclic products and reconstruction

Use the register-resident four-digit encoder, then run the middle transform and reconstruction on separate CUDA streams. Two forward scratch pairs and two partial scratch slots alternate. Forward waits before reusing a pair, and middle waits before reusing a partial slot. A final event joins the last reconstruction to the capture stream. Each graph replay reads the current input buffers.

The serial four-digit parent measured 57.253501 ms and had first-product spans of 242.112 µs, 112.640 µs and 122.880 µs for its three stages. The new encoder shortens the first stage at source level. The three-stream schedule can overlap adjacent products. Overlap depends on shared memory, register occupancy and execution-unit contention. Moving from one set of scratch buffers to two increases the ranked temporary allocation from 90 MiB to 180 MiB.

The graph test reads the actual CUDA enqueue sequence. It checks all three producer-consumer edges, both reuse dependencies and the final join for five batch sizes up to 128. The encoder and middle tests emulate both polynomial lengths, and the decoder checks arbitrary-precision output limbs.

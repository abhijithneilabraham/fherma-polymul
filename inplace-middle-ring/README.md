# Reuse the first operand spectrum for the middle result

Forward writes four-column groups in the order reconstruction reads. Each middle block owns one prime plane and two outer frequencies. It loads both operands, exchanges values through shared memory with a block barrier, and overwrites the first operand at the addresses owned by the block. Other blocks read disjoint addresses.

Three pairs of spectra rotate through the three streams. Forward waits for reconstruction of product i−3 before reusing its pair. Middle waits for forward, and reconstruction waits for middle. The ranked graph allocates 174 MiB of temporary spectra, the same total as the control. It reads each raw operand once. The change gives middle a chance to overwrite resident cache lines and removes the separate partial-buffer handoff events. Logical intermediate loads and stores still total 21.75 GiB across the ranked batch.

The layout check covers every address in a plane at both supported sizes. The middle-kernel emulator uses actual aliased pointers and checks all untouched words against a snapshot. The graph test checks dependencies through 128 products. Compare full-batch time against the 36.482772 ms cooperative-load control.

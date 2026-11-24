# Replace the inner cyclic transform with NVIDIA cuPQC

One 256-thread block owns one outer frequency and one residue plane. It gathers both operands into shared memory, calls the SDK forward transform twice, multiplies matching frequencies, then calls the normalized inverse. The ranked block uses 16 KiB of shared operand storage. Its input reads cover four contiguous words in each eight-word group. The neighboring block reads the other four.

The existing encoder phase now converts each input to Montgomery form. The output phase removes the radix, compensates for the SDK inverse normalization, and applies the CRT weight in one multiply. The input and output layouts, three-stream graph, and number of global intermediate writes match the retained control. Setup builds forward and inverse twiddle tables once per prime outside graph capture.

The host tests check all phase factors and six dense-by-sparse products. A second test runs the submitted addressing and barriers with a host substitute for the documented forward and inverse operations. The binary links the real SDK in Blackwell CI. The official judge checks execution with the real SDK and measures the full batch. [SDK device operations](https://docs.nvidia.com/cuda/cupqc/libraries/cupqc_ntt/api/device.html)

# Load packed input tensor fragments cooperatively

The measured register-transpose encoder keeps 28 live words per ranked
digit row. Fetch the existing byte matrix operands with warp matrix
loads. Both shapes have 16-byte aligned rows and eight-row address
groups that cover all 32 shared banks. The ranked tail uses two
matrices and zeros the absent half in registers. This preserves the
28 KiB shared-memory footprint and the exact padding.

The ranked batch uses 121,634,816 warp inner steps across the two
operands. Each full step uses two cooperative loads instead of eight
scalar shared loads. The shorter ranked step uses two loads instead
of four. All tensor operations, weight reads and global layouts stay
the same. Inspect the linked instruction mix and resource report
before timing against 37.465438 ms.
[Measured control](../evidence/optimization-two-shortjointshuffle/outcome.json) · [PTX matrix loads](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#warp-level-matrix-load-instruction-ldmatrix)

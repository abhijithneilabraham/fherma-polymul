# Give each ranked encoder tile ten warps

The packed 32-point encoder has 29 prime pairs to process per block. Six warps need up to five rounds. Ten warps finish in three rounds and leave one warp slot idle in the last round. The block loads the same 128 wide coefficients, evaluates the same prime pairs and stores each result to the same address. The auxiliary shape uses its existing six-warps launch.

The ranked block grows from 40 to 48 KiB of static shared memory. Two blocks fit within the SM's 128 KiB shared memory capacity. At the parent's compiled 80 registers per thread, the register file could also hold both blocks. That changes the resident warp limit from 12 to 20 when shared memory limits the parent to two blocks. Check the actual compiler allocation and judge timing. Shorter per-warp work does not establish a batch speedup. [Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)

Compare against the measured packed encoder once it has a native result. The cooperative CPU fixture runs every lane and checks the complete boundary tile for both operands and both widths. The stream dependency test checks the same graph across five batch sizes.

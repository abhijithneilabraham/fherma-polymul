# Load fused CRT fragments cooperatively

The fused control loads eight 32-bit shared words per lane in each tensor
inner step. Use two warp matrix loads to fetch the same eight fragments.
Each `ldmatrix.x4` reads the left and right halves of 16 rows and returns the
words in INT8 MMA order. Ranked rows are 496 bytes apart. Auxiliary rows
are 272 bytes apart. Both strides keep row starts 16-byte aligned and map
each group of eight half rows onto 32 distinct shared banks.

The ranked batch executes 110,100,480 warp inner steps. At source level
each now issues two cooperative shared loads in place of eight scalar
loads. The shared-memory byte count, tensor operations, weight layout,
carry computation and global traffic follow the 40.148 ms control.
Inspect the emitted loads and compare the complete batch to determine the
runtime effect. [Fused control](../two-shortjoint/README.md) · [PTX matrix loads](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-load-instruction-ldmatrix)

The decoder fixture emulates the documented source lane and fragment
order, then checks every output word against independent wide integers
for both sizes. The build audit inspects the ranked and auxiliary resource
reports and checks that the Blackwell compiler emits matrix loads.

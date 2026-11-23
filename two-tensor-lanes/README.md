# Compute one tensor dot per lane

An adjacent pair of threads holds four byte sums for each of two rows. The tensor-middle control lets the even lane reconstruct both rows. It exchanges two sums per row and runs two predicated reductions. Here the even lane handles the first row and the odd lane handles the second. They exchange the two words needed by their partner and each lane reduces one complete dot.

The 256-point middle performs six tensor passes, each with eight pairs of matrix products. This cuts exchanges from 192 to 96 per warp row. Across the ranked batch that is 364,904,448 fewer requested warp exchanges. It also replaces two half-warp reductions with one full-warp reduction in each pair. The byte arithmetic, output addresses, tensor instructions and all other graph stages follow the tensor-middle control. The full-warp result stores request up to two words per shared bank. The control stores the two rows in separate instructions.

The same 32-lane emulation checks the exchanged sums against the exact dot and verifies six complete cyclic products and phased inverse rows. Compare the compiled instruction count and register use before timing this revision against tensor-middle.

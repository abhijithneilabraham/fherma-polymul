# Pack and scale seven input digits before their transform

The seven-digit source extracts each digit from adjacent input words again for every prime pair. This version lets one warp load an entire coefficient, extract its seven digits once and place the packed words in shared memory. At 1,747 bits this takes 56 words per coefficient, the same space as the original padded row. The warp uses temporary space that the transform can reclaim after the block barrier.

The input matrix contains the product of each byte weight and its digit scale modulo the prime. This removes a fixed modular multiplication for each digit, coefficient, prime and operand. At the ranked size that is 2,113,929,216 products per batch. Constant matrix storage grows from 2,304 to 16,128 bytes on the ranked shape. The instruction count for byte matrix multiplication, the 126 transform planes, and the 168 packed CRT tile pairs per output block come from the parent.

The CPU tests compare the packed matrix with independent modular powers, the production warp with direct integer digit evaluation, and a complete fused block with independent forward and middle transforms. The official experiment compares against the packed seven-digit decoder to measure the extra packing pass and the larger matrix.

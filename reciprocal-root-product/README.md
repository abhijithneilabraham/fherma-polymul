# Multiply by a fixed transform root using its fractional ratio

Let R = 2^32, let p < R/2, and let y be a fixed root below p. Store the two words of C = ceil(y R²/p). For any input word a, take h as the upper word of the low 64 bits of a*C. The upper word of (h+1)*p is the exact residue a*y mod p. The kernel uses two multiply-high operations and a low multiply-add. It adds one as the accumulator of the first high product.

Write a*y = k*p+r. If C = y R²/p+e, then 0 ≤ e < 1. The fraction of a*C left after removing k R² is r R²/p+a*e. It lies below R² since a<R and p<R/2. Thus h=floor(r R/p+a*e/R), and r R/p < h+1 < r R/p+2. Multiplication by p/R puts the result strictly between r and r+1. Even when r=p−1, h+1 fits in a word. The delayed subtraction a+p−b used by the parent also fits.

The direct-remainder literature uses the fractional part of a reciprocal product. Here the stored fraction includes the fixed root, and the word-size bound permits the final high-word shortcut. Both roots and inputs can vary across lanes. [Lemire, Kaser and Kurz](https://arxiv.org/abs/1902.01961)

The CPU tests compare 6,488,064 products with exact integer remainders. They include both prime sets and the largest 32-bit input. Separate tests cover the generated tables, actual forward, middle and inverse blocks, and centered reconstruction. Compare the complete native batch against the fused-difference parent and the retained cooperative-load control.

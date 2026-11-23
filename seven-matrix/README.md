# Combine the inverse digit transform and recovery

The packed seven-digit decoder transforms seven residues per prime and scales each output digit. Both steps are linear modulo that prime. The host combines their coefficients into four byte matrices. Each warp applies them to sixteen coefficients at once. After this step the decoder uses the parent's packed CRT reconstruction and carry code.

For prime p, inverse digit root u, digit d and input frequency s, the weight of input byte b is recovery[d] × u^(s×d) × 256^b modulo p. The four bytes of that weight occupy four output columns. The dot product has 28 live terms and four zero padding bytes. Recombining the columns with powers of 256 and reducing modulo p produces the same residue as the scalar transform.

Each byte sum is at most 28 × 255². The combined integer is less than 2^45. Dropping its low 16 bits and rounding the reciprocal down at 48 bits can underestimate the quotient by at most one for the supported primes. One conditional subtraction gives the canonical residue. Each warp reads all its input residues before publishing its first recovered digit.

The table test checks every packed byte, including padding, against independent modular powers on both shapes. A second test emulates the production decoder lanes and checks all output limbs for signed boundary and random inputs. Compare the official complete batch with the packed seven-digit decoder to determine whether tensor operations offset shuffle, reduction and memory costs.

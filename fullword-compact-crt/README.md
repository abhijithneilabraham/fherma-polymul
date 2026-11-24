# Center the ranked reconstruction in the carry loop

The 56-prime transform produces 112 residue words per coefficient. These occupy exactly fourteen 32-byte tensor steps. The two centering quotients previously required a fifteenth step. Here the row warp places each quotient in shared padding. During carry propagation it multiplies each quotient by the corresponding 32-bit limb of its fixed correction and adds both products. The correction vectors represent −M and −M × 2^880 modulo q, where M is the prime product.

The output matrix shrinks by 7,040 bytes. The two limb vectors use 440 bytes. Each ranked block drops eight tensor operations per warp across the four output groups, or 14,680,064 warp matrix instructions across B=128. The added work is two wide multiply-adds per output limb. Their operands are uniform across the 32 coefficients in the carry warp. The 868-bit shape still fits its corrections in the existing tile.

The independent Python reference checks every coefficient of both correction vectors and all matrix bytes. The native block emulator checks the final limbs against arbitrary-precision reconstruction for both shapes.

# Use 30-bit lazy butterflies on the selected scalar layout

The selected 45.501 ms implementation uses canonical corrections and the original odd-root forward layout. The separate 30-bit experiment used a different register-local forward kernel. This branch ports the smaller-prime arithmetic to the selected forward layout, keeping its phase application before the shared transpose.

The odd-root forward butterflies carry residues below 4p. The middle and inverse stay below 2p. For the forward phase, one Montgomery operand is below 4p and the other is canonical. In the middle product, both operands stay below 2p. Each product is below p × 2^32, so one final correction still produces the canonical result. The new prime set increases the ranked plane count from 116 to 120. Biased CRT reconstruction fits the two centered quotient corrections into those same 120 terms, preserving the fifteen tensor steps of the selected layout.

The host contracts compare actual forward blocks with the odd-frequency oracle and check the wide Montgomery inputs, reconstruction, prime bounds and full-size products. After Blackwell qualification, compare the whole ranked batch with the current selected scalar entry. [30-bit butterfly reference](https://github.com/poulpy-fhe/poulpy/blob/v0.10.0/backends/poulpy-gpu-portable/src/ntt4x30/ntt.rs)

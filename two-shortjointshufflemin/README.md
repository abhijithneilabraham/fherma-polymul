# Combine register transposition and unsigned-minimum arithmetic

The register-transpose graph reached 37.465438 ms on three official
seeds. The separate arithmetic branch reached 39.602632 ms. Both are
changes to the same 40.147894 ms fused control. The transpose changes
where each input lane holds values. The arithmetic helper changes
how the same residues are reduced.

Place the exact helper from the arithmetic branch on the new
register-transpose winner. Compare the paired change against 37.465438
ms. Neither parent result establishes the combined result.
[Register-transpose result](../evidence/optimization-two-shortjointshuffle/outcome.json) · [Arithmetic result](../evidence/optimization-two-shortjointmin/outcome.json)

# Fold the first 32-point rotation into byte encoding

The retained encoder processes rows 0–15 and 16–31 in separate matrix instructions. For each prime, multiply the second group's fixed byte weights by the fourth root ψ^(N/2) during initialization. Both sign residues in those rows then already carry the factor that the first forward butterfly needs. Its two outputs require only a sum and a difference. The lower sixteen rows use the original weights.

This transfers the earlier long-transform preweighting approach to the measured 32-point cooperative loader. It removes 973,078,528 fixed modular products per ranked batch. The input table grows from 59,392 to 118,784 bytes. Each warp issues the same number of matrix instructions and still reads each input coefficient once.

The independent host audit reconstructs both byte tables and checks that the rotation squares to minus one. The cooperative block emulator checks the complete transformed output against a direct evaluation for both operand streams and both supported sizes. Measure the full batch against 36.482772 ms.

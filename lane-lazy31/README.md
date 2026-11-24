# Reduce only the left operand of each lane-local butterfly

The official lane-owned encoder reached 36.064465 ms. Preserve its four-column output and warp ownership. In its 32-point forward transform, carry sums and differences in the interval [0, 2p). Each fixed-root product already returns a canonical value. Reduce the left operand once before using it in both outputs. The final phase product consumes an unreduced input and returns the original canonical output.

Compare the official batch median against the lane-owned control and the separate change on its previous control.

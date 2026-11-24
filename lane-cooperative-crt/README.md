# Use cooperative loads in the input and output stages

Keep the official 36.064465 ms lane-owned encoder and its input fragment loader. In reconstruction, each warp loads both matrix fragments cooperatively. Two warp-wide matrix loads replace eight shared-word loads per inner step. The resident data, coefficients, reconstruction, and three-stream schedule retain their existing layout.

Compare the official batch median against the lane-owned control and the separate change on its previous control.

# Short carry propagation on the two-stream graph

This combines the verified 54.709 ms schedule with the short-fold output reducer. The two-stream graph and the transforms come directly from overlap-crt. The CRT source and its full-width test are identical to short-fold. The queue runs the isolated reducer first. Compare both the isolated and combined native medians to see whether the shorter CRT helps when the next encoder is already active.

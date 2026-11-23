# Shared encoder weights for both operands

A block stages 32 left coefficients and the corresponding 32 right coefficients.
Each warp loads a fixed weight fragment once and applies it to four 16-row
tiles. The previous encoder applied each load to two tiles in a separate
launch per operand. This halves the issued weight-fragment loads and removes
128 kernel launches in the ranked batch. It doubles the encoder’s accumulator
count and shared input storage. The official comparison is against the Shoup
kernel at 63.889391 ms.

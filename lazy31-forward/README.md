# Postpone correction in the outer forward transform

Keep the 58 ranked primes and use the interval [0,2p) between the five butterfly levels. At the first level the left input is already canonical. At later levels reduce only the left input below p. Shoup multiplication of the right input by a fixed root returns a value below p even when its input reaches 2p−1. The two outputs a+b and a+p−b both fit in one unsigned word, since p is below 2^31.

The final phase multiplies a value below 2p by a canonical constant. Their product is below p × 2^32. The Montgomery sum fits in 64 bits since it is below 2p × 2^32. The phase writes canonical words for the existing middle transform.

Each 32-point evaluation replaces 160 logical sum and difference corrections with 64 left-input corrections. The original encoder, tensor dot, phase table, prime count and graph structure supply a direct full-batch comparison. Other lazy implementations used a smaller modulus and more planes or applied lazy arithmetic to additional stages.

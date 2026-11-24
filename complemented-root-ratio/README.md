# Remove the bias from the fractional root product

The parent stores C = ceil(y R²/p) for the Montgomery word size R = 2^32 and a fixed root y. Store that same integer as H R − L with 0 ≤ L < R. The low word of a H − high32(a L) is ceil(a C/R) modulo R. This expression permits a single multiply-add with a negative accumulator between the two high products.

For r = a*y modulo p, the high approximation lies between r R/p and r R/p + 2. Because p < R/2, taking the high word after multiplying by p returns exactly r. The ratio for y=0 is represented by two zeros. The full unsigned input range and both supported prime sets are covered by the same independent 6,488,064-product test as the parent.

The parent's ranked middle kernel has 2,328 static instructions and uses 64 registers per thread. Inspect generated multiply-adds and registers before native admission.

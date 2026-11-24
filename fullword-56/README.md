# Use 56 full-width residues for the ranked transform

A 1,747-bit input coefficient splits at bit 880. Its two evaluation signs share each prime. Using 56 primes close to 2^32 cuts the ranked transform from 116 planes to 112. The product of the primes still exceeds four times the maximum signed convolution magnitude. The host checks this bound at startup. The 868-bit configuration uses 30 primes.

For a product x = a*b, take t = low32(x) * inverse(p) modulo 2^32. The low words of x and t*p match. Subtract their high words and add p if the difference is negative. For a fixed multiplier y, store w = y*2^32 modulo p and f = w*inverse(p) modulo 2^32. Two high products and one low product give a*y modulo p. The wide byte-dot reducer keeps the full provisional remainder through the final comparison.

The ranked transform removes 3.45% of the planes. Each plane can cost at most 3.57% more before the plane savings disappear. The decoder still traverses a padded 480-byte row. The official control took 36.482772 ms. Only the judge measures the full-batch change.

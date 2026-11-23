# Lazy reduction for 31-bit odd-root butterflies

The 57.198401 ms control returns each fixed-root product in the interval from zero to p−1. The transform can instead use any congruent word below 2p. A 31-bit prime leaves room for that interval in an unsigned 32-bit register.

The Shoup reciprocal works for any input below 2^32. Its raw remainder is below 2p, so the butterfly skips one correction. Addition compares its first input with 2p minus the second to handle unsigned overflow. Subtraction uses the borrow to add 2p. At a variable Montgomery product, canonicalizing one of the two inputs ensures their product is below p times 2^32. The last inverse store returns a canonical residue for reconstruction.

The prime count, tables, memory layouts and graph schedule come from odd-roots. A CPU shuffle model checks every intermediate range and compares the results with independent transforms and convolution. A separate Blackwell compile compares the static instruction mix of the three ranked stages with the control. The complete GPU batch decides whether the saved corrections repay the cost of the wider addition.

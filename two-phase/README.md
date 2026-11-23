# One-word phase multiplication

The measured two-digit kernel applies a fixed rotation to each forward spectrum and an inverse rotation after the pointwise product. The ranked batch executes 2,919,235,584 such products. This version keeps one 32-bit entry per rotation and replaces its Montgomery multiply with ordinary word arithmetic.

Let R = 2^32. For a fixed residue v modulo p, store c = ceil(Rv/p). The high word of c × p recovers v exactly because Rv ≤ cp < Rv + p. The high word of a × c gives the true quotient of av/p or one more. Consequently av − p × high(a × c) lies strictly between −p and p. Since p < 2^31, the sign bit identifies the one case that needs an addition of p. The two high products can issue independently. The variable pointwise product still uses Montgomery arithmetic, so the inverse table contains the factor R that restores its scale.

The full-polynomial test checks forward and inverse phase maps at every prime and coefficient address. Boundary and random comparisons exercise the packed arithmetic at both widths. The build also disassembles the ranked parent and candidate for comparison. The official batch timing determines the value of the change.

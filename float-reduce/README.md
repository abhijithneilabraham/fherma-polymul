# Exact correction after a float quotient estimate

This variant retains the paired-lane input encoder and replaces its 64-bit
Barrett quotient estimate. A point-only FP32 reciprocal table gives a quotient
estimate, followed by exact signed integer remainder corrections. Canonical
residues, primes, NTT, CRT, transfers and ownership remain unchanged.

The maximum quotient is below 56,100. The conservative accumulated float error
bound is below 0.013376, so the estimated floor differs by at most one. One
negative and one positive integer correction return the exact canonical residue.
The tests cover all quotient boundaries and random remainders across both
prime plans. All three official cases passed at **76.955058 ms**, rank **17**.
Measured source: `23ca4c6882612887efacd91a41110a96c37a49a4`. Pure transfers are timed
separately. The 0.2138% lower recorded time has no paired noise qualification.

# Short carries in CRT output

The selected odd-root source folds a wide result modulo a number of the form 2^W − gap. Its final stage makes two full-width carry passes. This variant propagates each carry only as far as it reaches. If the first fold crosses 2^W, the remaining low value is below 2^70. Adding the gap then requires at most three words. A final comparison handles every residue in the interval from the modulus to 2^W − 1.

The speed and divergence of this data-dependent reducer need a native test. It is intended for this throughput benchmark and has data-dependent execution time.

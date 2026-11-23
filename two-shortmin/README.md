# Apply unsigned-minimum corrections to the 32-point kernel

The earlier transform graph recorded 45.501437 ms with unsigned-minimum corrections and 45.602867 ms with canonical comparisons. The current 43.967171 ms graph uses a different 32-point factorization and the original comparisons. Port the exact correction helper while preserving the selected graph, primes, buffers and tensor arithmetic.

For a value below twice the prime, subtracting the prime wraps to a large unsigned word exactly when the original value is already canonical. The unsigned minimum selects the canonical result. The subtraction butterfly similarly picks the smaller of a wrapped difference and the difference with the prime added. All participating residues are canonical and each prime is below 2^31.

Compare the actual ranked input, middle and inverse instructions with the selected build. Run the complete batch to measure any gain from combining the two changes. [Earlier native result](../evidence/optimization-two-min-direct/outcome.json) · [Selected layout](../two-shortouter/README.md)

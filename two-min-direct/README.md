# Test unsigned minimum corrections on the measured layout

The register-local forward candidate used fewer Blackwell instructions and ran 0.709 ms slower than the selected 45.603 ms entry. The original unsigned-minimum experiment inherited its forward layout. This variant moves only the exact residue corrections to the selected layout and stream graph. Addition, subtraction, Montgomery correction and fixed-root correction use an unsigned minimum of the two candidate words.

The host tests compare each operation against exact modular arithmetic and check the production transform and stream order. Compare the compiled stages and complete batch with the selected two-digit control. This measures the arithmetic change independently of the register transpose.

# Reduce canonical residues with an unsigned minimum

Each NTT butterfly adds or subtracts two canonical residues modulo p. For the sum x, take min(x, x − p) using unsigned 32-bit arithmetic. If x < p, subtracting p wraps to a larger value. Otherwise it gives the lower canonical residue. For the difference x = a − b, take min(x, x + p). A negative difference wraps to a larger value, and adding p returns it to the canonical range. The same sum correction works for the bounded remainders from Montgomery and Shoup multiplication.

This fork changes the common modular arithmetic and keeps the forward, middle and inverse layouts of the register-local control. It tests whether Blackwell can use integer minimum instead of predicate-controlled corrections. Each prime passed exact reference tests at boundary and random residues. The compiler report and a complete official batch decide whether to keep the change.

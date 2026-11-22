# CRT split at the integer boundary

The tensor-core kernel produces all L+2 words of each unreduced integer. A
separate 32-thread kernel applies the original exact two-fold reduction. The
carry words are retained, not truncated. Both launches finish inside `run`.

This separates the register-heavy reducer from the 224-thread MMA block. It
adds a launch and a temporary write/read, so speed must be measured. Encoding
uses the retained exact float-quotient path. All CRT arithmetic is unchanged.

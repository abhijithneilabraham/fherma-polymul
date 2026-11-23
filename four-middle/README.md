# Factor the four-digit middle transform into short columns and 256-point rows

The official four-digit run took 65.568 ms. Its middle product took 184.320 µs in the first product of the final batch. The parent uses a 32-by-64 middle transform at the ranked size. This candidate preserves the input conversion, outer transform, reconstruction, and all buffer boundaries.

Each thread reads one column of eight coefficients for each operand and transforms it in registers. Each warp then owns a 256-point row. It transforms both operands, multiplies corresponding values, and runs the inverse row transform. The original thread finishes its inverse column and writes the result. The smaller shape uses four coefficients per column. Only two block barriers separate these steps. The parent uses four. The register transpose for the 256-point transform comes from the tested two-digit middle design.

The shared layout exchanges two 16-word groups on the second outer row. This gives every active lane a distinct bank for both directions of the shared exchange. Every input load warp uses four sectors. The tests compare six full cyclic products per shape with independent dense convolution at the first, middle and final block positions. Runtime depends on register pressure and the throughput of the longer warp transform.

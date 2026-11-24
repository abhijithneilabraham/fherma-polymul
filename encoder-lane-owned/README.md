# Keep each encoder lane's prime and sign through the outer transform

After the matrix multiply, each lane owns one prime and sign for four columns. The control exchanges prime and sign with the register column index. This experiment keeps that original ownership. The outer transform exchanges rows using lane offsets divisible by four, so all exchanges stay within the same prime and sign. Each lane then writes its four columns with one aligned 16-byte store per frequency.

At the ranked shape the control's two bit exchanges use 16 warp shuffle instructions per prime-pair iteration. Across 128 products, two operands, 512 four-column groups and 29 prime pairs, this mapping removes 60,817,408 source-level warp shuffles. Each pair still writes the same 2,048 bytes. The new mapping reads the same phase values for both signs and can change register pressure.

The direct comparison is the 36.482772 ms cooperative-load graph. Inspect the generated stores, shuffle count and resource report before the full-batch measurement.

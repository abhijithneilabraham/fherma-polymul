# Use one phase table in both directions

The encoder and the middle kernel use the same 14.5 MiB phase table at the ranked size. Select an offset s such that 3s + J − 1 is zero modulo 2N, where J = N/32. Both inputs at column i read T(i) = ψ^((i+s)f) for odd outer frequency f. The inverse at column k reads T(J−1−k). For k = (i+j) mod J the three exponents sum to (i+j−k)f modulo 2N. Thus a product that wraps receives exactly ψ^(Jf), and a product that stays in range receives one.

The B encoder's fixed byte weights include the per-prime normalization and CRT weight. They also restore the Montgomery radix lost in the variable pointwise product. This costs one additional 58 KiB encoder table and zero additional tensor steps. The auxiliary shape uses the same identity.

Store four neighboring columns for each frequency together. The encoder accesses consecutive words across its four-column warp. The middle kernel looks up columns in reverse order, and each pair of outer frequencies still occupies complete 32-byte sectors. All three stages request phase data, so device traffic and timing depend on cache reuse.

The CPU contract checks every phase address at both sizes, for three partner indices each. It then composes the tables, normalization and pointwise product through six full dense-by-sparse polynomial multiplications. All output coefficients match independent signed convolution. Separate tests run the production cooperative blocks and check every byte of both encoder tables. Compare the official batch with the 36.482772 ms cooperative-load control.

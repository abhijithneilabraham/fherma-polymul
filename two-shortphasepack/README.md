# Factor the forward phase table into two exponent ranges

The 40.148 ms fused graph reads a separate phase word for every frequency and prime. Store 256 low powers and 512 high powers per ranked prime. Split the exponent at eight bits and multiply the two stored factors in Montgomery form. Both signed planes use the resulting phase. The inverse phase table and the middle and CRT kernels follow the selected graph.

The ranked forward table shrinks from 14.5 MiB to 174 KiB. This adds 973,078,528 modular products across the two ranked operands before accounting for overlap. The smaller table can remain cached, although the kernel requests two words instead of one. Check emitted code, resource use and the complete batch before promoting the change. [Cheddar's exponent factorization](https://arxiv.org/html/2407.13055v2#S4.SS2)

The host contract checks every forward phase for every prime against independently generated powers. The cooperative block test also verifies both complete operands and both supported widths.

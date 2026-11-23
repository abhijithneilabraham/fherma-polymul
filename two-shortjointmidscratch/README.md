# Reuse the middle transpose for both operands

The middle transform transposes both operands before its 256-point warp
transforms. The control keeps their tiles side by side. This version
transposes the first operand into registers,
and reuses the same shared tile for the second operand. Each warp then
multiplies its two spectra and returns its own inverse values to the same
tile. The long-row factorization and all global layouts stay the same.

Shared memory falls from 32 to 16 KiB per ranked block and from 16 to
8 KiB at the auxiliary width. Two additional block barriers protect reuse.
The second operand is published before the first warp transform to shorten
its register lifetime.
At ranked size the encoder, middle and reconstruction blocks together
request 93.75 KiB of shared memory. All three can fit within the SM's
100 KiB limit if the register allocation fits and the scheduler places
them together. Compare actual Blackwell registers, spills and native
batch time with the fused control. [Fused control](../two-shortjoint/README.md)

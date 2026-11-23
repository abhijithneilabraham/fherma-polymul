# Propagate CRT carries across each warp

Keep the selected short inverse, shared CRT inputs, centered quotients and
integer tensor arithmetic. After each tensor group, pack the four byte sums
into two words for each limb. Keep the packed groups in shared memory so
a warp can finish a coefficient without a barrier between groups. The
warp uses generate and propagate masks for limb addition and for the final
reduction modulo the public coefficient prime.

The ranked block goes from eleven block barriers to three. The new packed
sums occupy the same 30,464 shared bytes per block as the previous combined
allocation, including shared CRT inputs. The tensor operation count and
partial-spectrum layout stay the same. Each finishing warp writes the
coefficient in two limb groups. The official comparison will determine how
much the shorter synchronization chain helps. [Fused control](../two-shortjoint/README.md) · [Prior carry implementation](../two-shortcarry/README.md)

Both full-width decoder fixtures simulate the exact ballot and shuffle
instructions and compare all reconstructed words, including boundary inputs
and untouched output addresses.

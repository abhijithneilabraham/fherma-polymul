# Use one forward spectrum and overlap reconstruction

Run input conversion and the middle transform on the same stream.
The middle transform consumes the forward spectrum before the next
product can overwrite it. Fused reconstruction keeps its two partial
slots and can overlap both phases of the next product. The graph keeps
the same three kernels per product and eliminates the forward handoff
events. It reduces ranked forward scratch from 116 to 58 MiB and
leaves the raw input read count at one.

The trade is less forward-to-middle overlap in exchange for a smaller
cache footprint and less resource contention. Applying the previous
first-product spans of 140.736, 101.952 and 213.824 µs as constant
durations gives 31.278 ms for the 128-product graph. Those samples
cannot establish runtime after the schedule changes. Compare the full
batch with the 37.465438 ms register-transpose control.
[Measured control](../evidence/optimization-two-shortjointshuffle/outcome.json)

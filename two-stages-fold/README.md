# Two-digit multiplication on three streams

Input conversion and forward transforms run on the first stream, middle and inverse transforms on the second, and wide reconstruction on the third. Four events protect the two reusable buffers shared between streams. The graph joins the final reconstruction before it returns. Arithmetic and every kernel launch geometry come from the official 51.427 ms two-digit source. The dependency test checks the actual enqueue code at five batch sizes.

The combined candidate uses the shorter exact carry reducer after the two signed digit channels have been untwisted and accumulated. Compare it with the otherwise identical three-stream candidate.

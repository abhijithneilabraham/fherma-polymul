# Combine aligned input loads with word-sized modular reduction

This graph uses the selected three-stream schedule. Each input word is loaded once. The forward evaluation compensates for the padding in the upper digit. A bounded 32-bit quotient reduces the tensor accumulators. The polynomial transforms and wide reconstruction match the 48.835 ms control.

Compile comparisons of the individual encoders found 488 static instructions in the ranked control, 312 with aligned loads, and 440 with word reduction. Each contains eight integer tensor instruction sites. The independent changes act on different parts of the encoder. Official timing determines their combined effect.

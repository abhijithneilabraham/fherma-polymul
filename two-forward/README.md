# Keep the last forward butterflies in registers

The full-batch profiler measured 9.560 ms with only the forward transform active. The combined control already groups the last three cyclic middle butterflies in registers. This experiment applies that technique to the odd-root forward transform, which uses different roots and must write a transposed spectrum for the next stage.

The warp exchanges its lowest lane and register bits before the last three ranked stages. Each lane then owns both values in each short butterfly. The result goes directly to the shared transpose tile. XORing the row with its high bits lets both the producers and consumers use all 32 banks once. The consumer applies the phase as it writes the final spectrum. Host initialization arranges phases in that same order.

A ranked warp uses 28 exchanges instead of 40. A smaller-shape warp uses 16 instead of 20. Across the ranked batch this removes 91,226,112 warp exchanges. The phase load and spectrum store request the same four aligned memory sectors per complete warp. One cooperative test executes the actual block and compares both operands and the digit planes at boundary columns with direct modular evaluation. It also checks every transpose access and counts the executed exchanges. Compare the native batch against the combined input and middle control.

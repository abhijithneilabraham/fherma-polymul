# Compute the two digit evaluations in neighboring lanes

The two-digit encoder needs the low digit plus and minus a fixed multiple of the high digit for each prime. Its parent reduces both digit sums in an even lane, multiplies the high digit by the fixed factor, and issues two spectrum stores.

Here the high-digit byte weights include that factor and its padding correction. Both weight matrices are packed in the order required by the tensor instruction. A pair of lanes exchanges two partial byte sums. The even lane reduces the low digit while the odd lane reduces the scaled high digit. One final exchange gives both lanes the inputs for their respective signs. Each lane writes one evaluation.

Per output pair, this removes the separate fixed multiplication and uses three warp exchanges instead of four. It doubles the ranked input table from 29,696 to 59,392 bytes. The tensor instruction count, spectrum layout and downstream transforms come from the combined two-digit control. The tests reproduce every weight byte and check both lanes against integer arithmetic, including values on either side of the digit boundary.

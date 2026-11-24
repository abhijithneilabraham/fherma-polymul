# Keep inverse butterfly outputs below twice the prime

The 35.442676 ms control makes both outputs canonical after each inverse butterfly. Here each inverse butterfly starts with two canonical summands, then emits their sum and their biased difference. Both are below 2p. The next stage reduces the first input once. A fixed-root multiply consumes a full unsigned word and produces a canonical second input. An identity root that skips the multiply needs its own reduction.

The first inverse stage consumes canonical pointwise products and skips both input checks. The two outputs remain below 2p through the register transpose, warp shuffles and shared-memory exchange. The final phase multiply takes a value below 2p and a canonical phase. Their product is below 2p², which is below 2³²p. The standard single-subtraction Montgomery finish applies.

At the ranked length, the 256-point inverse has eight levels and the remaining column transform has three. For each spectrum word, the source removes one correction in the first level, 1/4 and 3/8 on average in the next two, and 1/2 at each of the next five. The three column levels save another 0, 1/4 and 3/8 on average. That totals 4.75 potential corrections across 973,078,528 ranked spectrum words. Predication, compiler scheduling and overlapping kernels determine the time saved. Compare the full official batch with the reflected control.

The exact production kernels run on host fixtures for both supported widths. The edge contract checks the range and congruence of both butterfly outputs and the final Montgomery multiply across every public prime.

# Alternate two carry buffers

Alternate the shared accumulator buffer across output groups. The publication barrier in group g+1 follows all reads of g and precedes every write of g+2 to the same bank. Keep the final output publication barrier.

Baseline: paired-accumulator-handoff at 35.156465 ms. Ranked carry-loop block waits change from eight to four, with 3,840 extra shared bytes per block.

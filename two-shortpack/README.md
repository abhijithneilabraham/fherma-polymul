# Omit the zero tail of the ranked shared input tile

The ranked byte tensor encoder uses 128-byte rows for two packed digits. A digit occupies at most 112 bytes, including the alignment offset. Its last four 32-bit words are zero. The smaller benchmark uses the existing shared layout.

The 32-point outer transform now stores only 28 words per ranked digit. The last tensor instruction gets zeros in registers for the part it does not load. A 28-word row still puts the eight active rows in distinct groups of four shared banks. The ranked input block uses 40 KiB of shared memory instead of 48 KiB. The change also removes 0.5 GiB of logical shared stores when loading the ranked batch. The count and output order of the matrix products follow the 32-point parent.

The CPU block fixture executes the active fragments and verifies both signs for every prime at both shapes. The Blackwell build captures register and shared memory use. Native timing will show whether the smaller working set improves the batch.

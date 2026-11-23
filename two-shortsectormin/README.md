# Combine unsigned-minimum arithmetic and the sector layout

The arithmetic branch took 39.602632 ms across three official seeds.
The sector branch took 40.136425 ms against their common 40.147894 ms
control. Their production changes are independent. The arithmetic branch
changes only the modular helper. The sector branch changes only the
layout of the middle output and fused decoder input.

Use the unchanged helper from the arithmetic winner on the exact sector
layout. Compare with the current 39.602632 ms leaderboard entry and
retain its revision. The small standalone sector result is within
observed seed variation, so measure the combined batch before attributing
any further gain. [Arithmetic result](../evidence/optimization-two-shortjointmin/outcome.json) · [Sector result](../evidence/optimization-two-shortsector/outcome.json)

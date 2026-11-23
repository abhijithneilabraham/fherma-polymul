# Process the 32-point spectra in two compact windows

The selected 43.967 ms graph uses 116 MiB for its two forward workspaces. Split the 58 ranked primes into 30 and 28. Each window still holds both operands and both signs. Two 30-prime capacity slots use 60 MiB in total. The smaller workload splits 30 primes into 16 and 14.

The middle stream starts as soon as the first window is ready. The next forward window can run alongside it. Before a slot is reused, the input stream waits for its previous middle reader. Both middle groups write the original full-plane layout for the unchanged inverse and CRT. The new schedule adds one forward and one middle launch per product. It reads and packs the raw input twice, which may offset the smaller working set.

The source fixtures check both compact producer ranges, including the shorter second group, and compare the middle destinations and inverse with independent reference arithmetic. The graph test verifies both slot dependencies and the final join. The standard middle sample starts at the end of forward. An extra diagnostic records the full middle span, which can overlap the forward window. Use the complete ranked batch to decide whether the smaller workspace repays its cost.

# Transform both signs of one prime together

A middle block now operates on the positive and negative channels for a single prime and outer frequency. Both channels use the same inverse phase, and paired lanes can read the same phase address. The encoder emits the pair together and the reconstruction warp loads both residues with one 64-bit load per lane.

Each output position groups two consecutive columns within an outer frequency. Per 32 words the middle store requests eight sectors and the reconstruction load requests eight. The cooperative-load control requests four and sixteen respectively. Across the ranked batch the difference is 3.625 GiB of requested sectors. Cache hits and write combining determine the effect on device-memory traffic.

Both independent 2,048-point channels still fit the same 512-thread block. The direct control completed the ranked batch in 36.482772 ms. Compare with the older two-column sector layout as a secondary control for the change in sign ownership.

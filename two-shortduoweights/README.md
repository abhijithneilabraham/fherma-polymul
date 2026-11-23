# Reuse CRT tensor weights across two columns

The paired inverse gives one block both neighboring columns. Its CRT reconstructs one column at a time and reads the same byte weights for each. Interleave the two columns inside the tensor loop so one weight fragment supplies both. The ranked graph requests 12.891 GiB of fixed weight bytes across the batch, down from 25.781 GiB before caching. The weight table is about 103 KiB. Any reduction in device-memory traffic depends on cache behavior.

Keep both columns' partial digits in shared memory and normalize carries with warp ballots. The ranked output block now executes three block barriers, down from twenty-one. It performs the same number of tensor products and reads the same coalesced residue sectors as the paired inverse parent.

The ranked block uses 60,928 bytes of dynamic shared memory. The launch explicitly requests this capacity. Two blocks fit within the SM's published 128 KiB capacity if register use permits. The larger live accumulator and shared allocation can reduce occupancy. Check the exact Blackwell build and compare native timing against the original paired-column reconstruction. [Parent](../two-shortduo/README.md) · [Carry arithmetic](../two-shortcarry/README.md)

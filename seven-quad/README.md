# Reconstruct four adjacent columns together

The middle kernel writes four neighboring columns side by side. In the packed decoder, four independent warps each read one word from the same 16 sectors. A warp now uses a 16-byte load per lane to read all four columns from those 16 sectors. It performs four independent outer transforms and keeps the four sets of residues in shared memory.

A 512-thread block contains two teams. Each team reconstructs one column at a time with the packed decoder's arithmetic. The two teams reuse their carry buffers for the second pair of columns. The ranked block uses 96,768 bytes of shared memory. The grid falls from 2,048 blocks to 512 per product. Reduced parallelism and the larger shared allocation are the main costs to measure.

At the ranked batch size, decoder input requests cover 3.9375 GiB of sectors versus 15.75 GiB in the parent, assuming the vector load issues as one instruction. The useful input bytes stay the same. Cache hits can already serve repeated requests in the parent. The source-level decoder test checks every limb of all four columns, both widths and all untouched outputs. The address test checks the complete intermediate plane.

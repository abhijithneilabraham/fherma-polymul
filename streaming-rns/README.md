# Streaming RNS batch

Four product slots share three queues: input copy, compute, and output copy.
Pinned staging and parallel host copies overlap DMA with exact NTT and CRT work.
The result vector uses normal ownership and range insertion. All copies,
conversions, arithmetic and output construction stay in the timed call.

The fast pipeline is retained from ByteCRT. Only its outer entry names change.
A thin wrapper keeps the original general wide-integer path for other shapes.
Official result: 3/3 passed at B=128, N=65,536, W=1,747. Median: 89.184818 ms.
Recorded rank: 17. Source: `7d1d7052a7a3455c50462d3c0da4885a250229ba`.
The earlier upstream result was 89.648935 ms; that small difference does not
establish a new kernel-level gain.

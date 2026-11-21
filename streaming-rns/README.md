# Streaming RNS batch

Four product slots share three queues: input copy, compute, and output copy.
Pinned staging and parallel host copies overlap DMA with exact NTT and CRT work.
The result vector uses normal ownership and range insertion. All copies,
conversions, arithmetic and output construction stay in the timed call.

The fast pipeline is retained from ByteCRT. Only its outer entry names change.
A thin wrapper keeps the original general wide-integer path for other shapes.
The published 89.648935 ms result belongs to the upstream adapter. This packaged
candidate has not yet run on the official judge.

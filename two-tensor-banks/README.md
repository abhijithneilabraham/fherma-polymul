# Remove the shared-bank collisions in the paired tensor kernel

Place the sixteen words of each tensor row together. XOR three row bits into the column. The resulting physical layout maps each warp input, A fragment and paired result store to all 32 banks once. The former paired stores could address one bank twice. The warp tile keeps four extra words for the final cross-warp transpose.

The tensor inverse uses a forward matrix and then reverses the output. Store the fixed inverse phases in the forward order so the warp can load the shared values and the phases in the same order. Reverse the shared destination index after the warp reads, and keep the original global output layout. Both the linear destination and the transposed consumer cover every shared bank once.

The radix scratch falls from 10,368 to 8,320 bytes per block. Including matrix weights, the middle uses 13,440 shared bytes. The dot arithmetic and thread assignments follow the paired tensor kernel. The exact fixtures check the physical bank addresses, the changed phase table, and both cyclic products and weighted inverse values. Run against the paired tensor control after its timing.

# Balance sector usage across the middle and reconstruction kernels

The 32-point joint reconstruction loads one word from each frequency row for one column. The parent places four columns together, so one reconstruction warp requests 16 sectors to read 32 words. The middle kernel that writes those words requests four sectors per warp store.

Put two columns in each sector instead. Each warp load in reconstruction now requests eight sectors. Each middle warp store also requests eight. Counting one write and one read for every partial spectrum word, these instructions request 20% fewer sectors. The layout is a permutation of the same allocation. Neither the arithmetic, stream dependencies nor the compiled data type changes. Cache and write combining can change the physical traffic.

The address fixture checks every word at both sizes and enumerates every affected warp instruction. The existing stand-alone forward, middle and reconstruction oracles check the results against the new layout. Compare with the joint reconstruction before choosing a layout.

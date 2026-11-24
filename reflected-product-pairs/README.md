# Process two polynomial products per launch

The batch has 128 independent products at the ranked size. Each transform tile uses the same read-only phase factors, prime data and byte weights for both products. Neighboring blocks handle the same tile from two products. This gives the GPU an opportunity to reuse those constants in cache. Block placement and cache hits are hardware decisions.

Each pair runs one forward transform, one middle transform and one reconstruction kernel. The full graph uses 192 kernel launches instead of 384. Three streams let consecutive groups overlap. Each stream waits for the producer before reading its result and waits for the previous consumer before reusing its buffer. Two pairs can be in flight. Scratch space doubles from 174 to 348 MiB at the ranked size. An odd final product uses the single-product layout.

The host checks cover product offsets in the production forward, middle and reconstruction bodies. The graph test checks the dependencies for batches of 1, 2, 3, 57 and 128 products. Compare the full official batch with the 35.442676 ms reflected-table control. Stage timestamps measure the first group.

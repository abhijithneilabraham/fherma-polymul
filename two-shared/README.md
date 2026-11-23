# Share the transform tables between digit evaluations

Each input produces positive and negative evaluations under the same prime. Both evaluations use identical polynomial roots, normalization and CRT weights. Store one set per prime. Neighboring forward blocks process the four input evaluations for one prime and tile. Neighboring middle blocks process its two signs. The inverse retains the output layout consumed by reconstruction.

At the ranked width the two dense phase arrays fall from 58 MiB together to 29 MiB. There are still 116 data planes per operand. The smaller tables reduce the constant working set. Actual device-memory traffic and batch time depend on cache reuse. Compare the official median with folded-two.

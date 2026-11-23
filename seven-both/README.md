# Three kernels per polynomial product

One launch converts and transforms both inputs. Its blocks own four columns each and every prime for an operand. A ranked block stages 128 wide coefficients once, then processes three prime pairs at a time. Four columns from two adjacent frequency rows occupy one 32-byte sector. The middle kernel consumes both rows together. The final kernel inverts 32-coefficient columns and reconstructs the wide results in the same block.

The complete graph keeps one pair of forward spectra and one middle output. Across the ranked batch this removes 23.625 GiB of logical intermediate traffic relative to seven-plane. The remaining stage boundaries total 23.625 GiB. Adding the original inputs and final output gives 28.78125 GiB before constants, cache effects or partially used memory transactions. The official time will show how these savings compare with the larger blocks and extra synchronization.

# Reduce cache pollution from one-pass coefficients

Each ranked product reads 27.5 MiB of raw coefficients and writes
13.75 MiB of final coefficients. The intermediate spectra and fixed
weights are reused during the same batch. Mark only the raw reads and
the final stores as streaming to let those cache lines be evicted sooner.
This leaves the intermediate spectra, phases, roots and weights on the
ordinary cache policy.

The same addresses carry the same values at the same points in the
graph. The change marks 5.15625 GiB of logical ranked traffic as
streaming. Each operand is loaded once by the fused encoder and each
result is written once by the fused decoder. Compare complete batch time
and the separately checked transfer limits with the 40.136425 ms sector
control. The build audit checks for the emitted eviction hint.
[Sector control](../two-shortsector/README.md) · [PTX cache operators](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#cache-operators)

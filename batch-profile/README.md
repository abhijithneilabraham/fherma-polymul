# Whole-batch stage timing

Each product records six event timestamps in its compute stream. Adjacent
timestamps bracket the two encoding launches together, forward transform,
middle transform, inverse transform and reconstruction. Diagnostics sum each
interval across all products after completion. The span from the first to
the last event also includes gaps between products. Compare that span and the
official median against warp-ntt to quantify instrumentation overhead.

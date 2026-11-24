# Give middle transform blocks higher scheduling priority

The control completed 128 products in 36.482772 ms using cooperative matrix loads. This experiment raises the priority of ready middle transform blocks. CUDA captures the middle stream's priority in its kernel nodes and applies it when the graph is instantiated with per-node priorities. Each event dependency still controls when a block can become ready.

On the earlier sector-layout graph the same policy saved 0.441866 ms. Repeating that absolute saving would give 36.040906 ms. The judge measures whether the effect transfers.

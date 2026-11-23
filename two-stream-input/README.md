# Convert the next input while the preceding forward transform runs

On the measured three-stream graph, conversion and forward transform occupied one stream for 48.170 of 49.179 ms. The graph required the forward transform of product i to finish before converting product i+1. Forward ran for 9.560 ms without another timed stage active.

This experiment gives the forward transform its own stream and alternates two encoded-input slots. The forward transform waits for both operands of its product and for the middle stage to release the spectrum. Conversion waits before reusing the slot from two products back. The last reconstruction joins the original stream before return. Every polynomial kernel and launch count comes from the combined control.

A second input slot costs 58 MiB at the ranked size. A one-product request allocates one slot. The earlier operand-forward design split A and B into separate forward launches. Here a single launch still transforms both operands of each product. The graph test checks task dependencies, buffer reuse and the final join for batches from one to 128. A first-product event measures actual overlap with the second conversion. The official batch time captures any cost of competing for the cache or compute units.

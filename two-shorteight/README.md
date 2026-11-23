# Run eight short inverse transforms per warp

The selected 32-point graph takes 43.967 ms. Its sampled inverse took 141 µs. The inverse currently assigns four columns to a warp and exchanges data across lanes in three of its five stages. Processing eight columns this way needs two sets of twelve exchanges.

Give each warp eight columns and use the lane to select row bits 1 and 2. Registers hold rows that differ in bits 0, 3 or 4, which makes three stages local. Eight columns now need sixteen exchanges instead of twenty-four. The wider grid has 3,840 ranked blocks per product. Each lane writes two adjacent rows together, so one 64-bit warp store fills eight sectors. The middle writes an eight-column tile so inverse loads still fill every sector.

The exact host fixtures exercise the new middle layout, the full inverse at both boundary blocks, the unchanged CRT addresses and both supported widths. They also enumerate the producer and consumer sectors and check every store's alignment. The Blackwell build checks the vector store and the register cost. Compare the official batch with the 43.967 ms control and the simpler paired inverse.

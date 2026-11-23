# Keep four inverse stages in registers

Give each warp sixteen 32-point transforms. Lane bit 4 selects row bit 2 of each transform. The other four row bits stay in registers. The inverse uses sixteen warp exchanges per sixteen columns, compared with 32 in the eight-column variant and 48 in the selected four-column layout. There are 1,920 ranked inverse blocks per product.

The middle writes sixteen columns per tile. Inverse loads still fill four sectors per warp instruction. Each lane stores four consecutive 32-bit words in the CRT layout. Two lanes fill each column's sector, so a 128-bit store covers sixteen full sectors. The wider assignment increases live registers and reduces the inverse grid's independent work.

The host fixture exercises the middle and inverse at both widths, counts the exact exchanges and checks every sector and vector alignment. Blackwell qualification checks that the compiler emits the expected vector stores. Compare the full batch against the selected layout and the eight-column alternative.

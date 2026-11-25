# Direct-limb polynomial multiplication on the CPU

TonRac's lifetime planner highlights three transient byte arrays in the packing and unpacking graph. The prior entry reused their storage. This variant instead writes each operand into GMP's own limbs and reads the product's limbs directly. The slot width is the smallest whole number of limbs that holds N(q−1)². High zero limbs may be absent in a normalized GMP product, so the decoder clips each read to the allocated product size. Inputs, outputs, modular reduction and all timings follow the same FHERMA host interface as the original CPU entry.

The native CPU backend uses exact GMP integers throughout. Build offline with CMake and the included GMP 6.3.0 archive. The archive contains GNU GMP's original licenses. [TonRac PR](https://github.com/farizrahman4u/carnot/pull/1) · [Arena planner](https://github.com/farizrahman4u/carnot/blob/2fac07d9679561a98b506bebc6a9ae5a658867c8/tonrac/graph.py)

GNU M4 1.4.19 is built offline for GMP assembly generation. Its source archive includes its GPL license.

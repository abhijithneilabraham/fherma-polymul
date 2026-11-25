# Exact polynomial multiplication on the CPU

Each coefficient occupies the smallest whole number of bytes that can hold N(q−1)². Multiply the two packed positive integers with GMP, subtract the upper half from the lower half coefficient by coefficient, and reduce each difference modulo q. The run includes packing, arithmetic, output allocation and reduction. Independent products share a pool of up to eight host threads.

Carnot's slice 10 search compared byte-aligned and machine-word-aligned packing on the development CPU. At the ranked width these occupy 439 and 440 bytes per coefficient. The entry uses the byte-aligned choice. The program builds offline from the included GNU GMP 6.3.0 source archive and has no CUDA dependency. The repository license applies to the entry code. GMP retains its own licenses in the archive. [Carnot search](https://github.com/farizrahman4u/carnot/blob/5bdff39edafa3b8b43659804be4081ba5abb647b/slice10/between.py) · [GNU GMP](https://gmplib.org/)

GNU M4 1.4.19 is built offline for GMP assembly generation. Its source archive includes its GPL license.

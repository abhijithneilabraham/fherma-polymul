# Read each input limb once for two-digit conversion

The 880-bit split crosses a word boundary. The preceding encoder reads adjacent words to assemble the unaligned upper digit. Here one warp reads each of the 55 words of a ranked input coefficient once. A shuffle places the upper digit in a word-aligned tensor fragment with 16 leading zero bits. On the 868-bit shape a half warp reads 28 words and leaves 24 leading zero bits.

Multiplying the forward twist by the modular inverse of the padding factor restores the original evaluation at each prime. Reconstruction keeps the original inverse twist. Both inputs still use the original arithmetic split, primes and polynomial transforms. The host test checks the number of input reads, both digit arrays and every byte of the encoding and reconstruction matrices. Compare the official batch against the single-stream two-digit baseline.

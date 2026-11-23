# Skip zero reconstruction tiles

Each reconstruction warp consumes eight output bytes and both halves of a 32-byte tensor tile. The immutable coefficients for many of those tiles are zero. Initialization scans each tile, writes a bit mask per warp, and packs both halves of every live tile in lane order. Reconstruction visits just the set bits and loads contiguous words. Only modulus-dependent constants are packed during initialization.

| Shape | Dense tiles | Live tiles | Warp matrix instructions removed |
|---|---:|---:|---:|
| 868 bits | 140 | 86 | 38.6% |
| 1,747 bits | 476 | 225 | 52.7% |

Each tile issues two matrix instructions per 32 coefficients. The ranked batch removes 131,596,288 warp matrix instructions. The CPU reference checks every byte, including the omitted tiles, against independently computed CRT coefficients. The cooperative decoder checks every reconstructed word on each boundary block. Compare the official full batch with the three-kernel parent. Shorter loops, different work per warp, mask iteration and weight locality can all affect elapsed time.

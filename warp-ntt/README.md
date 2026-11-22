# Warp RNS transforms

This candidate starts from the retained 76.064133 ms entry. Encoding, the
cooperative CRT kernel, its register budget and pure transfer hooks stay intact.
Three new exact Montgomery kernels replace the library transform stages.

Each warp owns a 128/256-point transform in four/eight coefficient registers per
lane. DIF forward and DIT inverse share bit-reversed frequency order. Point-only
phase tables preserve the full mixed-radix transform. Eight neighboring warps
share a padded transpose tile for aligned global stores. Four padding words
make that transpose conflict-free under the 32-bank mapping.

The CPU contract checks actual butterfly source at all 171 prime plans and the
complete mixed-radix frequency/output map at both full ring sizes. These are
mathematical and emulation results. Native GPU correctness and speed are pending.

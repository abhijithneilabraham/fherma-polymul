# Stream compact spectra into the fused winner

Split the 58 ranked primes into 30 and 28. Give each group one producer
workspace that is reused after the middle transform reads it. Two workspaces
use 60 MiB instead of 116 MiB. The middle stream consumes the first group
before the second group finishes forward conversion. Both write separate
planes into one of the winning graph’s two partial-spectrum slots. Fused
inverse and CRT wait for both groups and use the original output layout.

Shorter live spectra can help the cache and create overlap. The ranked batch
also gets 256 more launches and requests a second copy of both inputs, or
3.4375 GiB of logical raw reads before cache reuse. The same tensor products
and transforms still run. Measure this trade against the 40.148 ms fused
parent. Keep the original prime-window experiment as a control for separating
the effect of the fused output. [Window control](../two-shortwindow/README.md) · [Fused control](../two-shortjoint/README.md)

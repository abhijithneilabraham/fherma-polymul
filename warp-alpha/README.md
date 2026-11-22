# Cooperative exact CRT correction

This variant changes only the CRT alpha sum. Each warp owns one row at a time,
loads consecutive prime terms and reduces them with warp shuffles. Seven warps
share the 32 coefficient rows. The unsigned MMA, carries, output reduction and
all earlier transforms remain unchanged.

For valid centered CRT inputs, the exact sum is at least 0.25 from a rounding
boundary. The parallel sum has absolute error below 0.000109 at P=114, so the
rounded integer correction remains exact. Native GPU speed is unmeasured.

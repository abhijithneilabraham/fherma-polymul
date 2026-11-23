# Profile the fused 40.148 ms graph across the full batch

The official result times the entire product batch and reports stage samples for its first product. This version times the encoder and forward transform, middle transform, and inverse plus CRT for every product. Each stage records its start after the waits that protect its scratch. Its end records before it releases scratch or makes the result available. The input, output and execution order follow the source-bound winner.

The report shows the sum of each stage's spans, its time running alone, and time with zero, two or three stages active. An event span can include resource contention or scheduling delays. Compare the profiled batch with the winning uninstrumented run to measure the overhead. Use the exclusive spans to choose where to spend the next experiment. [Official control](../evidence/optimization-two-shortjoint/outcome.json)

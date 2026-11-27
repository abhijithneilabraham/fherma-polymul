# Skip the final carry-buffer reuse barrier

Keep each shared accumulator reuse barrier except the last. The final group has no later writer to its buffer. Each carry lane folds only its own digits, then the final block barrier publishes every reduced row.

Baseline: paired-accumulator-handoff at 35.156465 ms. Ranked carry-loop block waits change from eight to seven, with unchanged shared storage.

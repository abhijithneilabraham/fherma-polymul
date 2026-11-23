# Warp-parallel CRT carry handling

The tensor stage generates four byte sums for each output word. Each pair of lanes combines them, then publishes the low and high 32-bit halves. All groups can write to distinct word positions before a single block barrier.

One warp completes a coefficient. Each lane adds its low half to the preceding word’s high half. Two ballot masks encode where carries start and propagate. A 32-bit addition yields all incoming carries in parallel, with one carry between the two warp tiles. The same operation folds high bits modulo the challenge modulus. A final ballot selects the canonical answer. The outputs of each warp occupy consecutive memory words.

The selected kernel uses 80 registers and spills in the ranked CRT. The earlier eight-digit design gained from parallel carry handling. A native comparison with odd-roots will measure its effect here.

# Group 114 transform primes in two passes

The first pass processes primes 0 through 55 and the second processes 56 through
113. Each pass encodes both operands and completes all three transforms. Both
passes write the original reconstruction layout. One CRT launch consumes the
full set. The smaller benchmark retains one pass. The design trades a second
read of each input for a smaller immediate working set. It requires an official
comparison against the full-prime warp transform.

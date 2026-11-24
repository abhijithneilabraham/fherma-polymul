# Share a phase table across conjugate outer frequencies

The 35.442676 ms control shares each frequency's phase table across both input transforms and the inverse. This version also shares one table between frequencies f and 64−f. A cyclic carry changes the exponent by J=N/32. The two frequencies 64−f and −f therefore give the same carry factor because 64J=2N.

Let D be the smallest multiple of four with 3D at least J−1. Store T(t)=ψ^((t+s)f) for t from 0 through J+D−1, with 3s+2(J−1)=0 modulo 2N. For positive f, each input at i reads T(i+D), and the output at k reads T(2(J−1)−2D−k). Their three exponents sum to (i+j−k)f. For the paired negative frequency the inputs read T(J−1−i) and the output reads T(k). Their exponents sum to −(i+j−k)f. The independent full product test includes both wrapped and unwrapped terms.

Both input orientations keep a four-column vector in one aligned 16-byte load. Negative rows reverse its four words in registers. The ranked phase allocation uses 16×2,732 words per prime instead of 32×2,048. This reduces it from 14.50 MiB to 9.67 MiB without adding a modular product. Additional address calculations and lane selections can affect GPU time. Measure the full batch.

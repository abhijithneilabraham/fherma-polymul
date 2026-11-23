# Keep active forward spectra in a smaller cache window

The four-digit graph evaluates 30 ranked primes in groups of 12, 12 and 6. Each group runs input conversion, the forward transform and the middle product before the next group can overwrite its scratch buffer. The final kernel reconstructs each wide coefficient from all 30 primes. The smaller challenge shape uses two groups of eight.

On the ranked shape the allocated forward workspace drops from 60 to 24 MiB per product. The two raw inputs together occupy 27.5 MiB. Processing three groups reads them three times at the kernel interface, adding 6.875 GiB of logical input traffic over the batch. Reuse in cache determines how much reaches device memory. The intermediate reads and writes remain 22.5 GiB. Compare the grouped forward and middle totals and full official batch with the ungrouped four-digit source.

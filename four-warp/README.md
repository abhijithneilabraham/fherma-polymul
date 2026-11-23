# Give each warp one prime pair through input conversion and the outer transform

The four-digit control spent 235.424 µs on input conversion and the forward transform in one sampled product. It divided each input tile among all warps and used block barriers to exchange ownership. This candidate keeps the register-based middle transform from four-middle and changes the forward stage.

Each block loads four columns of wide inputs once. Each warp converts one pair of primes and owns the corresponding residues through the final spectrum write. Its tile holds all four columns and both primes. One block barrier completes the input load. One warp barrier makes the converted digits available for the same warp to read. The ranked parent executes 16 block barriers in this kernel.

Each lane carries four rows and one column during the 32-point transform. The two largest butterfly distances use registers and the remaining three exchange values inside the warp. The output goes directly into four complete memory sectors per warp store. The four digit planes share one phase load per coefficient and prime. At the ranked batch size the source requests 5.625 GiB fewer phase bytes before caching and compiler effects.

The ranked kernel launches 480 threads and requests 88 KiB of shared memory, versus 768 threads and about 40 KiB in the parent. Less parallel work per block can offset the saved synchronization. The full-block tests compare both operands and every prime with direct polynomial evaluation. A separate address check verifies every tile word, shared bank, and output sector. Official timing measures the complete batch.

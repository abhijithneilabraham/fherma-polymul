# Reconstruct the coefficients as each 32-point inverse finishes

The 32-point outer design runs its final inverse and CRT as separate kernels. Here one output block owns a column of 32 coefficients. A warp processes both signs for one prime. It runs both short inverses, recovers the signed digit residues immediately and publishes them to the reconstruction tile. This removes the weighted output buffer and the separate pass that reads it.

The ranked batch avoids **7.25 GiB** of logical inverse-buffer traffic. Two 29 MiB partial-spectrum slots keep the next middle transform independent of the previous reconstruction. The fused inverse warp gathers every fourth word. Neighboring column blocks may share those sectors in cache. Separate inverse warps use complete sectors and share work across four columns, so the fused version trades coalescing and extra warp exchanges for less staging.

The block test constructs its inverse input with a direct forward evaluation and checks the entire inverse and CRT output against Python integers at both sizes. The graph test verifies both slot lifetimes and the final stream join. Compare the full ranked batch with the same 32-point graph that keeps inverse and CRT separate.

# Resident RNS batch

Convert wide coefficients to small-prime residues. Run staged NTTs. Use integer
tensor cores to reconstruct each coefficient exactly modulo q.

Both complete input batches and the complete result have device buffers. One
scratch workspace serves successive products. A CUDA graph records six kernels
per product and replays them on fresh inputs. Forward transforms for both operands
share one launch. Each product uses one complete CRT launch.

Pure byte copies use the standard transfer hooks. All conversion, transforms,
reconstruction and device completion stay inside `fherma_run`. CPU output
allocation overlaps the GPU graph. The general exact kernel handles other shapes.

The fast path supports N=65536/W=1747 and N=32768/W=868, B=1..128, and checks the
near-power-of-two modulus. A runtime prime-product guard stays active in Release.
The copied math is unchanged. See NOTICE and THIRD_PARTY_LICENSE for ByteCRT.

Build with CUDA 12.8, cuPQC 0.6.0 and OpenMP in the challenge image:

```sh
cmake -B build -DFHERMA_CUDA_ARCH=native
cmake --build build --parallel 2
```

This adapter has no measured GPU result yet. The published ByteCRT result applies
to its own streaming adapter, not this graph or transfer boundary.

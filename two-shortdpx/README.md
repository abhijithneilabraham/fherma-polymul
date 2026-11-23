# Fuse the modular correction with CUDA add-and-min

The 32-point unsigned-minimum build contains 432 VIMNMX instruction sites in the ranked middle kernel. For each fixed-prime correction, the add that produces the candidate value can execute with the unsigned minimum. Use CUDA's DPX add-and-min intrinsic on supported GPU targets. It applies the same 32-bit wrapping addition and unsigned comparison to canonical inputs. The existing scalar expression supplies the CPU and older GPU path. [CUDA integer intrinsics](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html)

Compare the three actual ranked kernel specializations with the unsigned-minimum parent. A fused instruction can remove correction dependency steps, while the negative prime constant or instruction pipeline can offset the gain. Check the Blackwell disassembly for emitted operations and any register changes before allocating native runtime. [Unsigned-minimum parent](../two-shortmin/README.md)

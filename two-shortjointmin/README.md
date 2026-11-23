# Apply unsigned-minimum arithmetic to the fused winner

An earlier graph recorded a 0.22% improvement from unsigned-minimum modular corrections. On the separate 32-point graph, the port reduced static middle-kernel instruction sites by 10.3% and had zero register spills. This version transfers the same helper to the 40.148 ms graph that fuses inverse and CRT. Each transform and centered residue remains exact.

The fused result and both scratch slot lifetimes follow the verified control. The host fixture compares all four modular operations for every prime and executes the complete fused inverse and CRT at both widths. Check the fused decoder's register allocation and native batch time before retaining the combined change. [Selected fused result](../evidence/optimization-two-shortjoint/outcome.json) · [Earlier arithmetic comparison](../two-shortmin/README.md)

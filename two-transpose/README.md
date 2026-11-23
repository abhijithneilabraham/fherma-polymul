# Register butterflies for the middle transform

Each warp computes a 256-point transform. A lane holds eight coefficients in registers. After the first five forward stages, the warp exchanges the three low bits of its lane number with the three bits of the register number. Each lane now owns both coefficients of the final three butterflies. The pointwise product uses this layout directly. The inverse exchanges the bits back before writing its output.

For the affected stages of two forward transforms and one inverse, the code issues 36 warp shuffles per product of two rows of 256 coefficients. The original stages issue 72. The transpose adds register selects. Register use, instruction scheduling and the amount of middle-stage time hidden by input conversion determine the effect on batch time.

The host test executes the same butterflies and synchronized lane exchanges as the CUDA source. It compares all 256 forward evaluations with direct evaluation and all 256 output coefficients with a dense cyclic product at each supported auxiliary prime. The enclosing graph uses the selected two-digit schedule.

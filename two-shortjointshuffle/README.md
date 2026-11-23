# Keep the fused forward transpose in registers

The fused control loads each input coefficient once and reuses it across all
prime pairs. After tensor conversion each lane owns one prime and sign at four
columns. The outer transform needs the opposite ownership. The control stores
these values to shared memory and synchronizes the warp before reading them in
their new order.

Exchange the two low lane bits with the column index in registers. The
cooperative stage still loads the input tile once per operand block. It keeps
only the 28 live 32-bit words in each ranked digit row. Total ranked shared
memory falls from 49,152 to 28,672 bytes. The smaller shape keeps its existing
digit row layout. The same integer weights, tensor instruction count, NTT
roots and global spectrum layout apply.

Each lane replaces 16 shared stores and 16 shared loads per prime pair with
16 warp shuffles. Across the 128 ranked products this removes 7.25 GiB of shared stores and 7.25 GiB of shared loads for the transpose. The extra live values can increase register pressure or
increase the cost of instruction issue. Three 192-thread blocks need at most
112 registers per thread with allocation rounding, and at least 86,016 bytes
of shared memory per SM. Check the actual Blackwell build and compare the
whole batch with the fused control and the packed-input source separately.

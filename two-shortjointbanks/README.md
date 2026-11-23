# Remove four-way shared bank conflicts from the fused inverse output

The selected 40.148 ms kernel writes each recovered residue into a shared CRT matrix. All warp lanes write different coefficient rows for the same prime. With the original stride, four lanes share each of the eight active banks. XOR the two low word-index bits with the coefficient row divided by eight so all 32 bank accesses can be served together.

The global partial spectrum, short inverse, tensor arithmetic and graph follow the winning fused version. The transformed shared addresses also preserve conflict-free quotient reads and tensor gathers. The added address arithmetic can offset the benefit, so compare actual kernel resources and full-batch runtime.

The exhaustive address test checks every residue word and tensor group at both widths. The block fixture runs the complete fused inverse and CRT against the reference and checks every untouched address. [Official fused control](../evidence/optimization-two-shortjoint/outcome.json)

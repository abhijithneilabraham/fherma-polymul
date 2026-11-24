# Group intermediate residues by output column

Four neighboring reconstruction blocks consume the same 116 residue planes. The control spaces consecutive planes 256 KiB apart at the ranked size. This layout places each 128-word segment next to the corresponding segment of the next plane. The four blocks now share a contiguous 58 KiB region. Input size, arithmetic, kernel launches, and bytes requested per warp stay the same.

The experiment tests whether locality for the reconstruction blocks offsets the greater distance between successive middle-stage stores. Compare the full batch against the 36.482772 ms cooperative-load control.

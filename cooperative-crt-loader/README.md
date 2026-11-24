# Fetch reconstruction matrix fragments cooperatively

Keep the 36.482772 ms encoder and graph. Port the earlier cooperative decoder to this exact revision. Each reconstruction warp uses two matrix loads for the eight left-hand fragments in one tensor step. The two loads cover all 32 rows and preserve the layout of each byte dot. The ranked and auxiliary row strides remain 16-byte aligned.

The ranked graph executes 110,100,480 warp tensor steps. At source level this replaces eight scalar shared loads with two matrix loads in each step. The hardware transactions and full-batch time require measurement. The decoder emulator checks all wide output limbs and untouched addresses at both widths. [Earlier CRT implementation](../two-shortjointldmatrix/README.md)

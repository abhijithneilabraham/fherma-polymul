# Eight-lane folded reconstruction

Each subgroup reconstructs one coefficient. Its eight lanes each own eight
consecutive 32-bit columns. They exchange carry information, fold high words
twice using the challenge modulus and select the canonical output. The
subgroup uses 64 shared words per coefficient. This version isolates final
assembly from the other stages of the folded-eight baseline.

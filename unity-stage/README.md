# Adjacent-pair transform stage

The root exponent is zero everywhere in the distance-one stage. Each lane
exchanges a word with its neighbor and computes the modular sum or difference.
This specialization skips the fixed-root fetch and multiplication in that
stage of each forward and inverse transform. The measured Shoup kernel is
the control.

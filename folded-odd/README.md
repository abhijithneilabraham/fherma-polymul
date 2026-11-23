# Odd-root transforms for eight folded digits

The input encoder splits each coefficient into eight digits. The polynomial transform processes 128 planes, with eight planes sharing the same prime and phase tables. Odd-root butterflies incorporate the input and output rotations. The middle stage multiplies the spectra. Its Montgomery factor is canceled by the inverse phase and the existing digit decoder.

This changes the polynomial transform on the 72.628 ms folded-carry implementation. The eight-lane output reconstruction and kernel launch order are the same in both versions. The full-width ring test checks every output coefficient against an independent integer convolution with signed wraparound.

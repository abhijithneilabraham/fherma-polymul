# Isolate the packed CRT on the earlier seven-digit graph

The fused seven-digit candidate passed at 65.174 ms. Its sampled inverse plus CRT fell to 98 µs from 197 µs across the earlier two stages. Its sampled middle rose to 186 µs from 68 µs. The complete earlier graph took 57.179 ms.

Keep the earlier graph's input, middle and separate inverse. Change its decoder alone. The decoder receives the same weighted words, packs the seven centered quotients into seven bytes and uses the sparse tensor tiles. The ranked count is 168 tile pairs instead of 476. The separate inverse still writes its intermediate buffer. The measured fused result includes other changes, so it cannot give the isolated speed of this decoder.

The tests independently rebuild the packed table and compare the complete decoder on signed boundary and random cases at both widths. Compare the complete ranked batch against 57.179 ms and the selected 43.967 ms graph.

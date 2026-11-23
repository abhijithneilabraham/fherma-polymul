# Pipeline encoding, transforms and reconstruction

The 54.709 ms control overlaps output reconstruction with the next product. This graph also lets the next input conversion start as soon as the previous forward transform releases its input buffer. The middle and inverse transforms run on a third stream.

Each handoff protects a specific allocation. The next forward waits for the middle to finish reading its spectrum. The inverse waits for the preceding CRT to finish reading weighted residues. The middle and inverse share a stream, protecting their intermediate product. The last CRT joins the origin stream before timing ends. No extra polynomial workspaces are needed.

The graph test follows the actual enqueue calls for batch sizes 1, 2, 3, 57 and 128. It verifies all four dependencies and identifies which stages may run together. The runner reports the measured intersection of the next encoder with the first middle and CRT spans. An event span can include scheduling delay.

# Load the forward phase while butterflies can run

The register-local forward transform ran at 46.311 ms. Its parent scored 45.603 ms. The register-local transform writes a shared transpose, then its consumers load the phases they need before writing to global memory. The load can extend the delay after the barrier.

Load each consumer's phase into its own registers before the butterflies. The load addresses and tensor arithmetic stay the same. The consumers then have a phase ready when they read the shared transpose. Holding eight ranked phase words can raise register pressure. The machine-code comparison checks placement and resources, and the official batch determines whether the earlier loads help.

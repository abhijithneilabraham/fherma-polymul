# Resident compute with pure transfer hooks

The complete input batches and output have device buffers. A CUDA graph computes
every product. All conversion, NTT, CRT and device completion remain in `run`.
Its arithmetic and launch sequence match the earlier resident implementation.

The input hook copies raw input bytes through pinned staging. The output hook
copies the finished raw device result to pinned memory, then into a normal owned
host vector. This avoids a separate zero-fill before overwriting the vector.
Both hooks preserve the required layout and values. Neither launches a kernel.

The host copies use the ByteCRT parallel memcpy/memmove wrapper, with its original
license and notices. Point-only staging buffers are allocated at setup. A general
fallback remains for other points. The runtime log retains last-case phase times.

Three official cases passed at **77.362807 ms**, rank **17**.
Measured source: `b271b1ad3a814e58b4069a29c5168be16790c871`.
Pure copies are measured separately: H2D median 145.836771 ms and D2H median
66.188752 ms. Every case stayed within the transfer limits.

The separate streaming result is 89.184818 ms with copies included.

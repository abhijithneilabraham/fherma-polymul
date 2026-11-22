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

This candidate is unmeasured. The separate retained streaming result is 89.184818 ms.

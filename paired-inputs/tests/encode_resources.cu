// Isolated kernel resource report; never executes a CUDA kernel.
#ifdef CONTROL_ENCODE
#include "core/rns-encode-mma.cuh"
#else
#include "warp-encode.cuh"
#endif

void probe_encoder(unsigned N, unsigned L, unsigned P, const uint32_t* input,
                   const uint8_t* b, const uint32_t* primes, const uint64_t* mu,
                   uint32_t* output, cudaStream_t stream) {
#ifdef CONTROL_ENCODE
    bytecrt::launch_encode_mma32_transposed(N, L, P, input, b, primes, mu, output, stream);
#else
    bytecrt::launch_encode_mma32_warp_transposed(N, L, P, input, input + size_t(N)*L, b, primes, mu, output, stream);
#endif
}

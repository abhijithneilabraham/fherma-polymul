// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "warp-tables.hpp"
#include "sdk-descriptor.cuh"

namespace bytecrt::warp_ntt {
template<unsigned J>
__global__ void make_sdk_table(uint32_t* twiddles, const SdkPrime* primes) {
    const SdkPrime field = primes[blockIdx.y];
    auto* destination = twiddles + (2 * blockIdx.y + blockIdx.x) * J;
    InnerNtt<J,cupqc::nttDirection::FORWARD>().make_twiddles(
        destination, field.p, blockIdx.x ? field.inverse_root : field.root);
}

template<unsigned J>
__global__ void convert_sdk_table(uint32_t* twiddles, const SdkPrime* primes) {
    auto* destination = twiddles + (2 * blockIdx.y + blockIdx.x) * J;
    InnerNtt<J,cupqc::nttDirection::FORWARD>().transform_twiddles_to_mont(
        destination, primes[blockIdx.y].p);
}

template<unsigned N, unsigned P>
__global__ __launch_bounds__(256) void middle(const uint32_t* input, uint32_t* output,
        const uint32_t* phases, const uint32_t* twiddles, const SdkPrime* fields, const Mod* mods) {
    constexpr unsigned K = 32, J = N/K;
    const unsigned row = blockIdx.x, plane = blockIdx.y, prime = plane%(P/2);
    const auto mod = mods[prime];
    const uint32_t* forward = twiddles + 2*prime*J;
    __shared__ uint32_t a[J], b[J];
    for (unsigned id=threadIdx.x; id<J; id+=256) {
        const unsigned tiled = (row/2)*(2*J) + (id/4)*8 + (row%2)*4 + id%4;
        a[id] = input[size_t(plane)*N+tiled];
        b[id] = input[size_t(plane+P)*N+tiled];
    }
    __syncthreads();
    InnerNtt<J,cupqc::nttDirection::FORWARD> transform;
    transform.execute(a,forward,mod.p);
    __syncthreads();
    transform.execute(b,forward,mod.p);
    __syncthreads();
    for (unsigned i=threadIdx.x; i<J; i+=256) a[i]=multiply(a[i],b[i],mod);
    __syncthreads();
    InnerNtt<J,cupqc::nttDirection::INVERSE>().execute(a,forward+J,mod.p,fields[prime].inverse_length);
    __syncthreads();
    for (unsigned id=threadIdx.x; id<J; id+=256) {
        const unsigned tiled = (id/4)*K*4 + row*4 + id%4;
        output[size_t(plane)*N+tiled] = multiply(a[id],phases[(prime*K+row)*J+id],mod);
    }
}
} // namespace bytecrt::warp_ntt

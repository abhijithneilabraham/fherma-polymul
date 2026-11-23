// SPDX-License-Identifier: MIT
#pragma once
#include "folded-math.hpp"
#include "warp-mod.hpp"
namespace folded {
using bytecrt::warp_ntt::add;
using bytecrt::warp_ntt::sub;
using bytecrt::warp_ntt::multiply;
using bytecrt::warp_ntt::Mod;

// DIF produces physical bit-reversed frequencies; DIT consumes that order.
template<bool inverse, int distance>
__device__ __forceinline__ uint32_t digit_step(uint32_t x, unsigned lane, const Prime& p, const Mod& mod) {
    const unsigned index = (lane % distance) * (4 / distance);
    const bool upper = (lane & distance) != 0;
    if constexpr(inverse) {
        if (upper) x = multiply(x, p.roots[(8-index)%8], mod);
    }
    uint32_t peer = __shfl_xor_sync(0xffffffff, x, distance, 8);
    if (upper) {
        x = sub(peer, x, mod.p);
        if constexpr(!inverse) x = multiply(x, p.roots[index], mod);
    } else x = add(x,peer,mod.p);
    return x;
}

__global__ void encode(const uint32_t* input, uint32_t* planes, const Plan* plan, const Mod* mods) {
    constexpr unsigned rows = 32;
    __shared__ uint32_t raw[rows][L], transpose[8][rows+1];
    unsigned id = blockIdx.x % 256, base = blockIdx.x / 256 * rows;
    unsigned row = threadIdx.x / 8, digit = threadIdx.x % 8;
    for (unsigned word = threadIdx.x; word < rows*L; word += blockDim.x) {
        unsigned r = word / L, limb = word % L;
        raw[r][limb] = input[(id+(base+r)*256)*L+limb];
    }
    __syncthreads();
    uint32_t chunks[14];
    #pragma unroll
    for (unsigned i=0; i<14; ++i) chunks[i] = chunk(raw[row], digit, i);
    for (unsigned p=0; p<16; ++p) {
        const Prime& prime = plan->primes[p];
        Mod mod = mods[p];
        uint32_t x = encode_digit(chunks,digit,prime);
        x = digit_step<false,4>(x,digit,prime,mod);
        x = digit_step<false,2>(x,digit,prime,mod);
        x = digit_step<false,1>(x,digit,prime,mod);
        transpose[digit][row] = x;
        __syncthreads();
        unsigned frequency = threadIdx.x / rows, offset = threadIdx.x % rows;
        planes[(p*8+frequency)*N+id*256+base+offset] = transpose[frequency][offset];
        __syncthreads();
    }
}

__global__ void decode(const uint32_t* weighted, uint32_t* output, const Plan* plan, const Mod* mods) {
    __shared__ uint32_t transpose[8][33], digits[32][8][16];
    unsigned id = blockIdx.x % 256, base = blockIdx.x / 256 * 32;
    unsigned row = threadIdx.x/8, digit = threadIdx.x%8;
    uint32_t values[16];
    for (unsigned p=0; p<16; ++p) {
        const Prime& prime = plan->primes[p];
        Mod mod = mods[p];
        unsigned freq = threadIdx.x/32, offset = threadIdx.x%32;
        transpose[freq][offset] = weighted[((id*128+p*8+freq)*256)+base+offset];
        __syncthreads();
        uint32_t x = transpose[digit][row];
        x = digit_step<true,1>(x,digit,prime,mod);
        x = digit_step<true,2>(x,digit,prime,mod);
        x = digit_step<true,4>(x,digit,prime,mod);
        values[p] = multiply(x, prime.recover[digit], mod);
        __syncthreads();
    }
    recover_digit(values, digits[row][digit], *plan);
    __syncthreads();
    if (threadIdx.x < 32) {
        unsigned coefficient = id+(base+threadIdx.x)*256;
        assemble(digits[threadIdx.x], output+coefficient*L, *plan);
    }
}
}

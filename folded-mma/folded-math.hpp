// SPDX-License-Identifier: MIT
#pragma once
#include "folded-plan.hpp"
#ifdef __CUDACC__
#define FOLD_HD __host__ __device__ __forceinline__
#else
#define FOLD_HD inline
#endif
namespace folded {
constexpr unsigned N = 65536, L = 55, W = 1747, gap = 9699327;
FOLD_HD unsigned position(unsigned digit) { return (digit * W + 7) / 8; }

// Two consecutive words suffice for an unaligned 16-bit chunk.
FOLD_HD uint32_t chunk(const uint32_t* coefficient, unsigned digit, unsigned part) {
    unsigned start = position(digit) + part * 16;
    unsigned end = position(digit + 1);
    if (start >= end) return 0;
    unsigned word = start / 32, shift = start % 32;
    uint32_t value = coefficient[word] >> shift;
    if (shift > 16 && word + 1 < L) value |= coefficient[word + 1] << (32 - shift);
    unsigned bits = end - start < 16 ? end - start : 16;
    return value & ((1u << bits) - 1);
}

FOLD_HD uint32_t encode_digit(const uint32_t chunks[14], unsigned digit, const Prime& prime) {
    uint64_t sum = 0;
    for (unsigned k = 0; k < 14; ++k)
        sum += uint64_t(chunks[k]) * prime.digit_weights[digit][k];
    // sum < 14 * 65536 * 2^31. Reciprocal quotient undershoots by at most one.
#ifdef __CUDA_ARCH__
    uint64_t quotient = __umul64hi(sum, prime.reciprocal);
    uint64_t remainder = sum - quotient * prime.p;
    return uint32_t(remainder >= prime.p ? remainder - prime.p : remainder);
#else
    return uint32_t(sum % prime.p);
#endif
}

FOLD_HD void recover_digit(const uint32_t values[16], uint32_t* dst, const Plan& plan) {
    double estimate = 0;
    for (unsigned i = 0; i < 16; ++i) estimate += double(values[i]) / plan.primes[i].p;
    // Q > 4*bound puts the exact sum within 1/4 of its nearest integer.
    int alpha = int(estimate + .5);
    int64_t carry = 0;
    for (unsigned j = 0; j < 16; ++j) {
        int64_t offset = (int64_t(1) - alpha) * plan.product[j];
        int64_t low = carry + uint32_t(offset);
        int64_t high = offset >> 32;
        for (unsigned i = 0; i < 16; ++i) {
            uint64_t term = uint64_t(values[i]) * plan.primes[i].crt[j];
            low += uint32_t(term);
            high += int64_t(term >> 32);
        }
        dst[j] = uint32_t(low);
        carry = high + (low >> 32);
    }
}

// Sum the eight nonnegative, biased CRT digits then fold the high bits twice.
FOLD_HD void assemble(const uint32_t digits[8][16], uint32_t* dst, const Plan& plan) {
    uint32_t wide[64];
    uint64_t carry = 0;
    for (int j = 0; j < 64; ++j) {
        uint64_t sum = carry + (j < 55 ? plan.bias_correction[j] : 0);
        for (unsigned d = 0; d < 8; ++d) {
            unsigned bit = position(d), shift = bit % 32;
            int offset = j - int(bit / 32);
            if (offset >= 0 && offset < 16) sum += uint32_t(digits[d][offset] << shift);
            if (shift && offset > 0 && offset <= 16) sum += digits[d][offset-1] >> (32-shift);
        }
        wide[j] = uint32_t(sum);
        carry = sum >> 32;
    }
    constexpr uint32_t mask = (1u << 19) - 1;
    carry = 0;
    for (unsigned j = 0; j < L; ++j) {
        unsigned source = 54 + j;
        uint32_t high = source < 64 ? wide[source] >> 19 : 0;
        if (source+1 < 64) high |= wide[source+1] << 13;
        uint32_t low = j == 54 ? wide[j] & mask : wide[j];
        carry += low + uint64_t(high) * gap;
        dst[j] = uint32_t(carry);
        carry >>= 32;
    }
    carry = uint64_t(dst[54] >> 19) * gap;
    dst[54] &= mask;
    for (unsigned j = 0; j < L; ++j) {
        carry += dst[j];
        dst[j] = uint32_t(carry);
        carry >>= 32;
    }
    bool subtract = dst[54] > mask;
    if (dst[54] == mask) {
        subtract = dst[0] >= uint32_t(0u-gap);
        for (unsigned j = 1; j < 54; ++j) subtract = subtract && dst[j] == UINT32_MAX;
    }
    if (subtract) {
        carry = gap;
        for (unsigned j = 0; j < L; ++j) {
            carry += dst[j];
            dst[j] = uint32_t(carry);
            carry >>= 32;
        }
        dst[54] &= mask;
    }
}
}
#undef FOLD_HD

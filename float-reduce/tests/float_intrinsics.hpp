#pragma once
#include <cstdint>
#define __device__
#define __forceinline__ inline
inline float __ull2float_rn(uint64_t value) { volatile float r = static_cast<float>(value); return r; }
inline float __fmul_rn(float a, float b) { volatile float r = a * b; return r; }
inline uint32_t __float2uint_rz(float value) { return static_cast<uint32_t>(value); }

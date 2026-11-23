// SPDX-License-Identifier: Apache-2.0
// ByteCRT MT32 CRT with a cooperative exact-alpha calculation.
#pragma once
#include "warp-carry.cuh"
#include "core/g1-crt.cuh"
#include "alpha-sum.cuh"
#include "warp-mod.hpp"
namespace bytecrt {

// Each lane owns one column and four rows of the odd-root inverse.
template<unsigned Distance=1>
__device__ __forceinline__ void inverse_four_columns(uint32_t (&v)[4],unsigned row,
        const warp_ntt::Twiddle* roots,const warp_ntt::Mod& mod) {
    constexpr unsigned groups=32/(2*Distance);
    #pragma unroll
    for(unsigned part=0;part<4;++part) {
        if constexpr(Distance>=8) {
            if(part&(Distance/8))continue;
        }
        const unsigned index=row+8*part;
        const auto root=roots[groups+index/(2*Distance)];
        if constexpr(Distance>=8) {
            const uint32_t a=v[part],b=v[part+Distance/8];
            v[part]=warp_ntt::add(a,b,mod.p);
            v[part+Distance/8]=warp_ntt::multiply_twiddle(warp_ntt::sub(a,b,mod.p),root,mod);
        } else {
            const uint32_t value=v[part];
            const uint32_t peer=__shfl_xor_sync(0xffffffffu,value,4*Distance);
            v[part]=(row&Distance)
                ?warp_ntt::multiply_twiddle(warp_ntt::sub(peer,value,mod.p),root,mod)
                :warp_ntt::add(value,peer,mod.p);
        }
    }
    if constexpr(Distance<16)inverse_four_columns<2*Distance>(v,row,roots,mod);
}
template<unsigned L, unsigned Kpad>
struct alignas(16) CrtHalfStorage {
    alignas(16) uint8_t residues[4][16][Kpad+16];
    uint32_t lows[2][16][(L+2)|1], highs[2][16][(L+2)|1];
};

struct CrtCarryWarp {
    __device__ __forceinline__ uint32_t ballot(bool value) const {
        return __ballot_sync(0xffffffffu,value);
    }
    __device__ __forceinline__ uint32_t get(uint32_t value,unsigned lane) const {
        return __shfl_sync(0xffffffffu,value,lane);
    }
};

__device__ __forceinline__ void accumulate_crt_tile(int32_t* c, uint32_t a0,
        uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}

template <unsigned W, unsigned L, unsigned P, unsigned Kpad>
__global__ __launch_bounds__(224, 2) void g1_crt32_warp_alpha(const uint32_t* __restrict__ partial,
                        const uint8_t* __restrict__ b,
                        const float* __restrict__ invp,
                        const uint32_t* primes,const warp_ntt::Twiddle* inverse_theta,
                        const warp_ntt::Twiddle* roots,const warp_ntt::Mod* mods,
                        uint32_t gap,
                        uint32_t* __restrict__ out,
                        unsigned block_begin = 0) {
    static_assert((W == 1747 && L == 55 && P == 116 && Kpad == 480) ||
                  (W == 868 && L == 28 && P == 60 && Kpad == 256),
                  "G1 full-width geometries");
    constexpr unsigned MT = 16, NT = 56, threads = 224;
    constexpr unsigned groups = (L + 13) / 14;
    // Four padding words make the m16n8k32 A gather bank-bijective:
    // stride_words mod 32 is 28 (main) or 4 (aux), bank = g*stride+t.
    constexpr unsigned row_words = Kpad / 4;
    constexpr unsigned shared_stride = Kpad + 16;
    constexpr unsigned shared_word_stride = shared_stride / 4;
    constexpr unsigned N=W==1747?65536:32768;
    constexpr unsigned J=N/32;
    __shared__ CrtHalfStorage<L,Kpad> storage;
    auto& columns = storage.residues;

    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31, warp = tid >> 5;
    const unsigned g = lane >> 2, t = lane & 3;
    const unsigned k_lane = 4 * t;
    const unsigned first=4*(block_begin+blockIdx.x);
    const unsigned local_column=lane%4, row=lane/4;
    const unsigned column_offset=first*32+lane;
    auto* own_column=reinterpret_cast<uint32_t*>(columns[local_column]);
    constexpr unsigned count=P/2, warps=threads/32, waves=(count+warps-1)/warps;
    // Each warp owns its second set of 16 inverse rows until shared
    // storage is released by the first set.
    uint32_t held[waves][2][2];
    #pragma unroll
    for(unsigned wave=0;wave<waves;++wave) {
        const unsigned prime=warp+wave*warps;
        if(prime>=count)continue;
        const warp_ntt::Mod mod=mods[prime];
        const auto* factors=roots+size_t(prime)*(64+J)+32;
        const auto* plus_base=partial+size_t(prime)*N+column_offset;
        const auto* minus_base=partial+size_t(count+prime)*N+column_offset;
        uint32_t plus[4]={plus_base[0],plus_base[32],plus_base[64],plus_base[96]};
        uint32_t minus[4]={minus_base[0],minus_base[32],minus_base[64],minus_base[96]};
        inverse_four_columns(plus,row,factors,mod);
        inverse_four_columns(minus,row,factors,mod);
        const auto theta=inverse_theta[prime];
        #pragma unroll
        for(unsigned part=0;part<4;++part) {
            const uint32_t lo=warp_ntt::add(plus[part],minus[part],mod.p);
            const uint32_t hi=warp_ntt::multiply_twiddle(
                warp_ntt::sub(plus[part],minus[part],mod.p),theta,mod);
            if(part<2) {
                auto* terms=own_column+(row+8*part)*shared_word_stride;
                terms[prime]=lo;
                terms[count+prime]=hi;
            } else {
                held[wave][part-2][0]=lo;
                held[wave][part-2][1]=hi;
            }
        }
    }
    if(warp==0) {
        for(unsigned part=0;part<2;++part) {
            auto* terms=own_column+(row+8*part)*shared_word_stride;
            for(unsigned j=P;j<row_words;++j)terms[j]=0;
        }
    }
    #pragma unroll 1
    for(unsigned half=0;half<2;++half) {
        if(half) {
            #pragma unroll
            for(unsigned wave=0;wave<waves;++wave) {
                const unsigned prime=warp+wave*warps;
                if(prime>=count)continue;
                #pragma unroll
                for(unsigned part=0;part<2;++part) {
                    auto* terms=own_column+(row+8*part)*shared_word_stride;
                    terms[prime]=held[wave][part][0];
                    terms[count+prime]=held[wave][part][1];
                }
            }
        }
        __syncthreads();

        for (unsigned column=0;column<4;++column) {
            auto* shared_words=reinterpret_cast<uint32_t*>(columns[column]);
            for (unsigned row=warp;row<MT;row+=threads/32) {
                uint32_t* terms=shared_words+row*shared_word_stride;
                float low=warp_alpha::sum(terms,invp,count,lane);
                float high=warp_alpha::sum(terms+count,invp,count,lane);
                if(!lane) { terms[P]=__float2uint_rn(low); terms[P+1]=__float2uint_rn(high); }
            }
        }
        __syncthreads();

        // Keep four residue tiles but reuse the carry scratch for each pair.
        #pragma unroll 1
        for(unsigned pair=0;pair<4;pair+=2) {
            for (unsigned group=0; group<groups; ++group) {
                const unsigned b_col=group*NT+warp*8+g;
                int32_t sums[2][4]{};
                for (unsigned k=0;k<Kpad;k+=32) {
                    uint32_t b0=0,b1=0;
                    if (b_col<4*L) {
                        const auto* weight=reinterpret_cast<const uint32_t*>(b+b_col*Kpad+k+k_lane);
                        b0=weight[0];
                        b1=weight[4];
                    }
                    #pragma unroll
                    for(unsigned column=0;column<2;++column) {
                        const auto& matrix=columns[pair+column];
                        const auto* top=reinterpret_cast<const uint32_t*>(matrix[g]+k+k_lane);
                        const auto* lower=reinterpret_cast<const uint32_t*>(matrix[g+8]+k+k_lane);
                        accumulate_crt_tile(sums[column],top[0],lower[0],top[4],lower[4],b0,b1);
                    }
                }
                #pragma unroll
                for(unsigned column=0;column<2;++column) {
                    #pragma unroll
                    for(unsigned part=0;part<2;++part) {
                        uint32_t x=sums[column][2*part],y=sums[column][2*part+1];
                        uint32_t z=__shfl_xor_sync(0xffffffffu,x,1),w=__shfl_xor_sync(0xffffffffu,y,1);
                        warp_carry::store_dot<L>(storage.lows[column],storage.highs[column],lane,warp,group,part,x,y,z,w);
                    }
                }
            }
            __syncthreads();

            for(unsigned column=0;column<2;++column) {
                for(unsigned row=warp;row<MT;row+=threads/32) {
                    const unsigned coefficient=first+pair+column+(half*MT+row)*J;
                    warp_carry::finish<W,L>(storage.lows[column][row],storage.highs[column][row],
                                            gap,out+size_t(coefficient)*L,lane,CrtCarryWarp{});
                }
            }
            if(pair==0 || half==0)__syncthreads();
        }
    }
}

inline void launch_g1_crt32_warp_alpha_transposed(unsigned W, unsigned M,
        const uint32_t* a, const uint8_t* b, const float* inverse, const uint32_t* primes, const warp_ntt::Twiddle* inverse_theta, const warp_ntt::Twiddle* roots,const warp_ntt::Mod* mods,
        uint32_t gap, uint32_t* output, cudaStream_t stream) {
    if (W == 1747 && M == 65536)
        g1_crt32_warp_alpha<1747,55,116,480><<<M/128,224,0,stream>>>(a,b,inverse,primes,inverse_theta,roots,mods,gap,output);
    else if (W == 868 && M == 32768)
        g1_crt32_warp_alpha<868,28,60,256><<<M/128,224,0,stream>>>(a,b,inverse,primes,inverse_theta,roots,mods,gap,output);
    else throw std::runtime_error("Cooperative alpha requires full staged geometries");
}
} // namespace bytecrt

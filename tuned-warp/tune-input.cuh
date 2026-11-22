// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace tune_detail {
__device__ uint32_t hash(uint32_t x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
__global__ void synthetic(uint32_t* a,uint32_t* b,size_t words,unsigned L,unsigned top_mask){
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<words;i+=size_t(gridDim.x)*blockDim.x){
        uint32_t x=hash(uint32_t(i)^0xa1379b45u),y=hash(uint32_t(i)^0x6192fe7bu);
        if(i%L==L-1){x%=top_mask;y%=top_mask;} // Top limb is strictly below q's top limb.
        a[i]=x;b[i]=y;
    }
}
__global__ void compare(const uint32_t* a,const uint32_t* b,size_t words,unsigned* bad){
    bool differs=false;
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<words;i+=size_t(gridDim.x)*blockDim.x)
        differs |= a[i]!=b[i];
    unsigned mask=__ballot_sync(0xffffffffu,differs);
    if(!(threadIdx.x&31) && mask)atomicExch(bad,1u);
}
} // namespace tune_detail

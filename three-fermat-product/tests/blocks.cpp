#define _XOPEN_SOURCE 600
#include <ucontext.h>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <vector>
struct Index {unsigned x,y;};
Index threadIdx{},blockIdx{};
namespace sim {
struct Lane {ucontext_t ctx{};std::array<char,32768> stack;int waiting=-1;};
std::array<Lane,256> lanes;
std::array<unsigned,33> arrived{};
std::array<unsigned,256> data{};
ucontext_t scheduler;
std::function<void()> kernel;
void invoke() {kernel();lanes[threadIdx.x].waiting=-2;}
void wait(unsigned group) {
    unsigned i=threadIdx.x;
    lanes[i].waiting=int(group);
    unsigned count=group==32?256:8,first=group==32?0:8*group;
    if(++arrived[group]==count) {
        for(unsigned j=first;j<first+count;++j){assert(lanes[j].waiting==int(group));lanes[j].waiting=-1;}
        arrived[group]=0;
    }
    swapcontext(&lanes[i].ctx,&scheduler);
}
void launch(std::function<void()> f) {
    kernel=std::move(f);
    for(auto& lane:lanes) {
        getcontext(&lane.ctx);lane.ctx.uc_stack.ss_sp=lane.stack.data();lane.ctx.uc_stack.ss_size=lane.stack.size();
        lane.ctx.uc_stack.ss_flags=0;lane.ctx.uc_link=&scheduler;lane.waiting=-1;makecontext(&lane.ctx,invoke,0);
    }
    for(;;) {
        unsigned ready=0,live=0;
        for(unsigned i=0;i<256;++i) {
            live+=lanes[i].waiting!=-2;
            if(lanes[i].waiting==-1){++ready;threadIdx.x=i;swapcontext(&scheduler,&lanes[i].ctx);}
        }
        if(!live)break;
        assert(ready);
    }
}
}
void __syncwarp(unsigned mask) {assert(mask==(255u<<((threadIdx.x%32)&~7u)));sim::wait(threadIdx.x/8);}
unsigned __ballot_sync(unsigned mask,unsigned pred) {
    unsigned id=threadIdx.x;
    sim::data[id]=pred;__syncwarp(mask);
    unsigned bits=0;
    for(unsigned i=0;i<8;++i)bits|=unsigned(bool(sim::data[(id&~7u)+i]))<<((id%32&~7u)+i);
    __syncwarp(mask);return bits;
}
int __shfl_sync(unsigned mask,int value,unsigned source,unsigned width) {
    assert(width==8);
    unsigned id=threadIdx.x;sim::data[id]=uint32_t(value);__syncwarp(mask);
    int output=int32_t(sim::data[(id&~7u)+source%8]);__syncwarp(mask);return output;
}
unsigned __brev(unsigned x) {unsigned y=0;for(unsigned i=0;i<32;++i,x>>=1)y=2*y+(x&1);return y;}
#define __shared__
#define __device__
#define __global__
#define __forceinline__ inline
#define __syncthreads() sim::wait(32)
#include "../transform.cuh"
namespace fermat_native {alignas(16) uint32_t scratch[512*49];}
using namespace fermat_native;

// Feed the actual block one row. Preserve the judge's full array strides.
template<unsigned W>
void test_stage(unsigned stage,const uint32_t* input,const uint32_t* other,uint32_t* out) {
    constexpr unsigned n=65536,channel=19,slot=37;
    unsigned count=stage==2?512:256;
    auto packed=[&](const uint32_t* ptr,unsigned i){Stored<W> v;for(unsigned j=0;j<W;++j)v.limb[j]=ptr[i*(W+1)+j];v.zero=ptr[i*(W+1)+W];return v;};
    auto export_row=[&](unsigned i,const Stored<W>& v){for(unsigned j=0;j<W;++j)out[i*(W+1)+j]=v.limb[j];out[i*(W+1)+W]=v.zero;};
    if(stage==0) {
        std::vector<uint32_t> in(n*55),blank(n*55);
        std::vector<Stored<W>> output(2*n);
        for(unsigned i=0;i<count;++i)for(unsigned j=0;j<55;++j)in[(i*256+channel)*55+j]=input[i*55+j];
        blockIdx={channel,1};sim::launch([&]{outer_forward<W>(blank.data(),in.data(),output.data());});
        for(unsigned i=0;i<256;++i)export_row(i,output[n+i*256+channel]);
    } else if(stage==1) {
        std::vector<Stored<W>> in(2*n),output(4*n);
        for(unsigned i=0;i<count;++i)in[n+slot*256+i]=packed(input,i);
        blockIdx={slot,1};sim::launch([&]{inner_forward<W>(in.data(),output.data());});
        for(unsigned i=0;i<512;++i)export_row(i,output[(256+slot)*512+i]);
    } else if(stage==2) {
        std::vector<Stored<W>> in(4*n),output(n);
        for(unsigned i=0;i<count;++i){in[slot*512+i]=packed(input,i);in[(256+slot)*512+i]=packed(other,i);}
        blockIdx={slot,0};sim::launch([&]{inner_inverse<W>(in.data(),output.data());});
        for(unsigned i=0;i<256;++i)export_row(i,output[i*256+slot]);
    } else {
        std::vector<Stored<W>> in(n),output(n);
        for(unsigned i=0;i<count;++i)in[channel*256+i]=packed(input,i);
        blockIdx={channel,0};sim::launch([&]{outer_inverse<W>(in.data(),output.data());});
        for(unsigned i=0;i<256;++i)export_row(i,output[i*256+channel]);
    }
}
extern "C" void block(unsigned w,unsigned stage,const uint32_t* x,const uint32_t* y,uint32_t* output) {
    if(w==32)test_stage<32>(stage,x,y,output);
    else if(w==40)test_stage<40>(stage,x,y,output);
    else test_stage<48>(stage,x,y,output);
}

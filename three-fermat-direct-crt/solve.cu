#include "fherma.h"
#include "host_copy.hpp"
#include "transform.cuh"
#include "direct.cuh"
#include <array>
#include <cstdio>
#include <memory>

void* wide_init(const fherma::Point&);
fherma::Outputs wide_run(void*, const fherma::Inputs&);
void wide_free(void*);
void wide_h2d(void*, const fherma::Inputs&);
fherma::Outputs wide_d2h(void*, fherma::Outputs);
std::vector<std::pair<std::string,double>> wide_metrics(void*);

namespace {
using namespace fermat_native;
constexpr size_t n=65536, words_per_product=n*55;
void check(cudaError_t e,const char* what) {
    if(e!=cudaSuccess)throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(e));
}
struct State {
    unsigned batch=0,runs=0;
    size_t words=0;
    uint32_t *a=nullptr,*b=nullptr,*out=nullptr,*host_a=nullptr,*host_b=nullptr,*host_out=nullptr;
    void *outer=nullptr,*inner=nullptr,*fallback=nullptr;
    Stored<48>* large=nullptr;
    Stored<40>* medium=nullptr;
    Stored<32>* small=nullptr;
    cudaStream_t stream=nullptr;
    cudaGraph_t graph=nullptr;
    cudaGraphExec_t executable=nullptr;
    std::array<cudaEvent_t,6> events{};
    ~State() {
        if(stream)cudaStreamSynchronize(stream);
        if(executable)cudaGraphExecDestroy(executable);
        if(graph)cudaGraphDestroy(graph);
        for(auto event:events)if(event)cudaEventDestroy(event);
        for(auto p:{a,b,out})cudaFree(p);
        cudaFree(outer);cudaFree(inner);cudaFree(large);cudaFree(medium);cudaFree(small);
        for(auto p:{host_a,host_b,host_out})if(p)cudaFreeHost(p);
        if(stream)cudaStreamDestroy(stream);
        if(fallback)wide_free(fallback);
    }
};
__global__ void final_recovery(const Stored<48>* a,const Stored<40>* b,const Stored<32>* c,uint32_t* output) {
    __shared__ direct::Scratch scratch[128/8];
    unsigned i=blockIdx.x*(128/8)+threadIdx.x/8;
    direct::recover(a[i],b[i],c[i],output+size_t(i)*55,scratch[threadIdx.x/8]);
}
template<unsigned W> void configure() {
    constexpr int small_bytes=256*sizeof(Stored<W>),large_bytes=512*sizeof(Stored<W>);
    check(cudaFuncSetAttribute(outer_forward<W>,cudaFuncAttributeMaxDynamicSharedMemorySize,small_bytes),"outer forward shared memory");
    check(cudaFuncSetAttribute(outer_inverse<W>,cudaFuncAttributeMaxDynamicSharedMemorySize,288*sizeof(Stored<W>)),"outer inverse shared memory");
    check(cudaFuncSetAttribute(inner_forward<W>,cudaFuncAttributeMaxDynamicSharedMemorySize,large_bytes),"inner forward shared memory");
    check(cudaFuncSetAttribute(inner_inverse<W>,cudaFuncAttributeMaxDynamicSharedMemorySize,large_bytes),"inner inverse shared memory");
}
template<unsigned W>
void ring_product(State& r,const uint32_t* a,const uint32_t* b,Stored<W>* result) {
    auto* outer=static_cast<Stored<W>*>(r.outer);
    auto* inner=static_cast<Stored<W>*>(r.inner);
    constexpr unsigned small_bytes=256*sizeof(Stored<W>),large_bytes=512*sizeof(Stored<W>);
    outer_forward<W><<<dim3(256,2),threads,small_bytes,r.stream>>>(a,b,outer);
    inner_forward<W><<<dim3(256,2),threads,large_bytes,r.stream>>>(outer,inner);
    inner_inverse<W><<<256,threads,large_bytes,r.stream>>>(inner,outer);
    outer_inverse<W><<<256,threads,288*sizeof(Stored<W>),r.stream>>>(outer,result);
}
void stamp(State& r,unsigned i) {
    check(cudaEventRecordWithFlags(r.events[i],r.stream,cudaEventRecordExternal),"record arithmetic stage");
}
bool ranked(const fherma::Point& p) {
    if(!p.B || p.B>128 || p.N!=n || p.W!=1747 || p.L!=55 || p.q.data.size()!=55)return false;
    for(unsigned i=0;i<55;++i) {
        uint32_t expected=i==0?0u-9699327u:i==54?(1u<<19)-1:UINT32_MAX;
        if(p.q.data[i]!=expected)return false;
    }
    return true;
}
void validate(const State& r,const fherma::Inputs& in) {
    std::vector<int64_t> shape={r.batch,65536,55};
    if(in.a.shape!=shape || in.b.shape!=shape || in.a.data.size()!=r.words || in.b.data.size()!=r.words)
        throw std::runtime_error("Invalid polynomial batch shape");
}
}

void* fherma_init(const fherma::Point& p) {
    auto r=std::make_unique<State>();
    if(!ranked(p)) {r->fallback=wide_init(p);return r.release();}
    r->batch=p.B;r->words=words_per_product*p.B;
    for(auto ptr:{&r->a,&r->b,&r->out})check(cudaMalloc(ptr,r->words*4),"device polynomial buffer");
    for(auto ptr:{&r->host_a,&r->host_b,&r->host_out})check(cudaMallocHost(ptr,r->words*4),"host polynomial buffer");
    check(cudaMalloc(&r->outer,2*n*sizeof(Stored<48>)),"outer spectra");
    check(cudaMalloc(&r->inner,4*n*sizeof(Stored<48>)),"inner spectra");
    check(cudaMalloc(&r->large,n*sizeof(Stored<48>)),"large residue");
    check(cudaMalloc(&r->medium,n*sizeof(Stored<40>)),"medium residue");
    check(cudaMalloc(&r->small,n*sizeof(Stored<32>)),"small residue");
    check(cudaStreamCreateWithFlags(&r->stream,cudaStreamNonBlocking),"arithmetic stream");
    for(auto& e:r->events)check(cudaEventCreate(&e),"timing event");
    configure<48>();configure<40>();configure<32>();
    check(cudaStreamBeginCapture(r->stream,cudaStreamCaptureModeThreadLocal),"capture polynomial batch");
    stamp(*r,0);
    for(unsigned i=0;i<p.B;++i) {
        size_t off=i*words_per_product;
        ring_product(*r,r->a+off,r->b+off,r->large);
        if(!i)stamp(*r,1);
        ring_product(*r,r->a+off,r->b+off,r->medium);
        if(!i)stamp(*r,2);
        ring_product(*r,r->a+off,r->b+off,r->small);
        if(!i)stamp(*r,3);
        final_recovery<<<n/(128/8),128,0,r->stream>>>(r->large,r->medium,r->small,r->out+off);
        if(!i)stamp(*r,4);
    }
    check(cudaGetLastError(),"launch polynomial stages");
    stamp(*r,5);
    check(cudaStreamEndCapture(r->stream,&r->graph),"finish capture");
    check(cudaGraphInstantiate(&r->executable,r->graph,nullptr,nullptr,0),"instantiate graph");
    return r.release();
}

void fherma_h2d(void* state,const fherma::Inputs& in) {
    auto& r=*static_cast<State*>(state);
    if(r.fallback){wide_h2d(r.fallback,in);return;}
    validate(r,in);
    host_copy::upload_bytes(r.host_a,in.a.data);host_copy::upload_bytes(r.host_b,in.b.data);
    check(cudaMemcpyAsync(r.a,r.host_a,r.words*4,cudaMemcpyHostToDevice,r.stream),"copy a");
    check(cudaMemcpyAsync(r.b,r.host_b,r.words*4,cudaMemcpyHostToDevice,r.stream),"copy b");
    check(cudaStreamSynchronize(r.stream),"finish upload");
}
fherma::Outputs fherma_run(void* state,const fherma::Inputs& in) {
    auto& r=*static_cast<State*>(state);
    if(r.fallback)return wide_run(r.fallback,in);
    validate(r,in);
    check(cudaGraphLaunch(r.executable,r.stream),"launch polynomial product");
    check(cudaStreamSynchronize(r.stream),"finish polynomial product");
    ++r.runs;
    fherma::Outputs output;output.c.shape={r.batch,65536,55};return output;
}
fherma::Outputs fherma_d2h(void* state,fherma::Outputs out) {
    auto& r=*static_cast<State*>(state);
    if(r.fallback)return wide_d2h(r.fallback,std::move(out));
    check(cudaMemcpy(r.host_out,r.out,r.words*4,cudaMemcpyDeviceToHost),"download product");
    return host_copy::download_bytes(std::move(out),r.host_out,r.words);
}
std::vector<std::pair<std::string,double>> fherma_metrics(void* state) {
    auto& r=*static_cast<State*>(state);
    if(r.fallback)return wide_metrics(r.fallback);
    const char* labels[]={"first_product_large_ring_ms","first_product_medium_ring_ms","first_product_small_ring_ms","first_product_reconstruction_ms","batch_compute_ms"};
    std::vector<std::pair<std::string,double>> result;
    for(unsigned i=0;i<5;++i) {
        float ms=0;check(cudaEventElapsedTime(&ms,r.events[i==4?0:i],r.events[i+1]),"stage duration");
        result.emplace_back(labels[i],ms);
    }
    result.emplace_back("completed_runs",r.runs);return result;
}
void fherma_free(void* state) {
    auto* r=static_cast<State*>(state);
    if(r->runs) {
        try {for(const auto& item:fherma_metrics(r))std::fprintf(stderr,"RNS last_case %s=%.6f\n",item.first.c_str(),item.second);}
        catch(const std::exception& e){std::fprintf(stderr,"RNS diagnostic unavailable: %s\n",e.what());}
    }
    delete r;
}

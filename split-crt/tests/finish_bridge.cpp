#include <coroutine>
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <exception>
#include <iostream>
#include <utility>
#include <vector>
struct Index { unsigned x=0; } threadIdx,blockIdx;
struct Thread {
    struct promise_type {
        Thread get_return_object(){return Thread{std::coroutine_handle<promise_type>::from_promise(*this)};}
        std::suspend_always initial_suspend() noexcept{return {};}
        std::suspend_always final_suspend() noexcept{return {};}
        void return_void() noexcept{}
        void unhandled_exception(){std::terminate();}
    };
    std::coroutine_handle<promise_type> handle;
    Thread(std::coroutine_handle<promise_type> h):handle(h){}
    Thread(Thread&& other):handle(std::exchange(other.handle,{})){}
    ~Thread(){if(handle)handle.destroy();}
};
#define __device__
#define __forceinline__ inline
#include "finish-kernel.inc"

template<unsigned W,unsigned L>
void evaluate(unsigned rows,uint32_t gap) {
    assert(rows && rows%32==0);
    std::vector<uint32_t> q(L),input(size_t{rows}*(L+2));
    for(auto& word:q)std::cin>>word;
    for(auto& word:input)std::cin>>word;
    assert(std::cin.good());
    auto before=input;
    constexpr uint32_t guard=0xd42a93c1;
    std::vector<uint32_t> output(size_t{rows}*L+4,guard);
    for(unsigned block=0;block<rows/32;++block) {
        blockIdx.x=block;
        std::vector<Thread> threads;
        for(unsigned lane=0;lane<32;++lane)
            threads.push_back(bytecrt::finish_crt<W,L>(input.data(),q.data(),gap,output.data()+2));
        for(unsigned phase=0;phase<3;++phase)
            for(unsigned lane=0;lane<32;++lane){
                threadIdx.x=lane;assert(!threads[lane].handle.done());
                threads[lane].handle.resume();
            }
        for(auto& thread:threads)assert(thread.handle.done());
    }
    assert(input==before);
    assert(output[0]==guard && output[1]==guard && output[output.size()-1]==guard && output[output.size()-2]==guard);
    for(size_t i=2;i+2<output.size();++i)std::cout<<output[i]<<' ';
    std::cout<<'\n';
}
int main(){
    unsigned w,rows;uint32_t gap;
    while(std::cin>>w>>rows>>gap){
        if(w==1747)evaluate<1747,55>(rows,gap);
        else if(w==868)evaluate<868,28>(rows,gap);
        else return 2;
    }
}

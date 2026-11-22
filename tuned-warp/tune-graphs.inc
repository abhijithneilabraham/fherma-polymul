// Included within the adapter's namespace after enqueue_batch.
// Calibration operands depend only on public point parameters and fixed seeds.
// No actual case operands are available here. Every scored call replays the
// selected complete graph after the pure hook copies fresh input bytes.
void calibrate(Resident& r){
    struct Temporary {
        std::array<cudaGraph_t,9> graph{};
        std::array<cudaGraphExec_t,9> executable{};
        cudaEvent_t begin=nullptr,end=nullptr;
        uint32_t* reference=nullptr;unsigned* bad=nullptr;
        ~Temporary(){
            for(auto e:executable)if(e)cudaGraphExecDestroy(e);
            for(auto g:graph)if(g)cudaGraphDestroy(g);
            if(begin)cudaEventDestroy(begin);if(end)cudaEventDestroy(end);
            if(reference)cudaFree(reference);if(bad)cudaFree(bad);
        }
    } t;
    const auto select=[&](unsigned v){
        r.variant=v;unsigned bits=v?v-1:7;
        r.forward_group=(bits&1)?8:4;r.middle_group=(bits&2)?8:4;r.inverse_group=(bits&4)?8:4;
    };
    unsigned W=r.core->plan.w,top_mask=(uint32_t{1}<<(W-32*(r.l-1)))-1;
    tune_detail::synthetic<<<4096,256,0,r.stream>>>(r.a,r.b,r.words,r.l,top_mask);
    check(cudaStreamSynchronize(r.stream),"synthetic point-only inputs");
    check(cudaMalloc(&t.reference,r.words*sizeof(uint32_t)),"calibration reference buffer");
    check(cudaMalloc(&t.bad,sizeof(unsigned)),"calibration comparison flag");
    check(cudaEventCreate(&t.begin),"calibration begin event");check(cudaEventCreate(&t.end),"calibration end event");
    for(unsigned v=0;v<9;++v){
        select(v);
        check(cudaStreamBeginCapture(r.stream,cudaStreamCaptureModeThreadLocal),"calibration capture");
        if(r.n==65536)enqueue_batch<65536,114,480>(r);else enqueue_batch<32768,57,256>(r);
        check(cudaStreamEndCapture(r.stream,&t.graph[v]),"calibration capture end");
        check(cudaGraphInstantiate(&t.executable[v],t.graph[v],nullptr,nullptr,0),"calibration instantiate");
    }
    const auto replay=[&](unsigned v){
        check(cudaGraphLaunch(t.executable[v],r.stream),"calibration replay");
        check(cudaStreamSynchronize(r.stream),"calibration completion");
    };
    const auto measure=[&](unsigned v){
        check(cudaEventRecord(t.begin,r.stream),"calibration start");
        check(cudaGraphLaunch(t.executable[v],r.stream),"calibration timed replay");
        check(cudaEventRecord(t.end,r.stream),"calibration finish");
        check(cudaEventSynchronize(t.end),"calibration timed completion");
        float ms=0;check(cudaEventElapsedTime(&ms,t.begin,t.end),"calibration interval");return double(ms);
    };
    replay(0);
    check(cudaMemcpyAsync(t.reference,r.out,r.words*sizeof(uint32_t),cudaMemcpyDeviceToDevice,r.stream),"calibration reference copy");
    check(cudaStreamSynchronize(r.stream),"calibration reference ready");
    unsigned best=0;double best_ratio=1;
    for(unsigned v=1;v<9;++v){
        replay(v);unsigned bad=0;
        check(cudaMemsetAsync(t.bad,0,sizeof(unsigned),r.stream),"calibration flag reset");
        tune_detail::compare<<<2048,256,0,r.stream>>>(r.out,t.reference,r.words,t.bad);
        check(cudaMemcpyAsync(&bad,t.bad,sizeof(unsigned),cudaMemcpyDeviceToHost,r.stream),"calibration comparison");
        check(cudaStreamSynchronize(r.stream),"calibration comparison complete");
        if(bad){std::fprintf(stderr,"TUNE variant=%u match=0\n",v);continue;}
        std::array<double,3> ratios{},times{};double aa=0;
        for(unsigned repeat=0;repeat<3;++repeat){
            double before=measure(0),candidate=measure(v),after=measure(0);
            if(!(std::isfinite(before) && std::isfinite(candidate) && std::isfinite(after) && before>0 && candidate>0 && after>0))throw std::runtime_error("nonpositive calibration interval");
            ratios[repeat]=candidate/((before+after)/2);times[repeat]=candidate;
            aa=std::max(aa,std::abs(before-after)/std::min(before,after));
        }
        std::sort(ratios.begin(),ratios.end());std::sort(times.begin(),times.end());
        double margin=std::max(.02,2*aa);select(v);
        std::fprintf(stderr,"TUNE variant=%u match=1 forward=%u middle=%u inverse=%u gpu_ms=%.6f paired_ratio=%.6f aa_spread=%.6f\n",
            v,r.forward_group,r.middle_group,r.inverse_group,times[1],ratios[1],aa);
        if(ratios[1]<best_ratio && ratios[1]<1-margin){best=v;best_ratio=ratios[1];}
    }
    select(best);r.graph=t.graph[best];r.executable=t.executable[best];
    t.graph[best]=nullptr;t.executable[best]=nullptr;
    std::fprintf(stderr,"TUNE selected=%u paired_ratio=%.6f scope=point-only-synthetic-setup\n",best,best_ratio);
}

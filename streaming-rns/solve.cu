#include "fherma.h"
#include <memory>

void* bytecrt_stream_init(const fherma::Point&);
fherma::Outputs bytecrt_stream_run(void*, const fherma::Inputs&);
void bytecrt_stream_free(void*);
void* wide_init(const fherma::Point&);
fherma::Outputs wide_run(void*, const fherma::Inputs&);
void wide_free(void*);
void wide_h2d(void*, const fherma::Inputs&);
fherma::Outputs wide_d2h(void*, fherma::Outputs);

namespace {
struct Choice {
    bool streaming = false;
    void* inner = nullptr;
    size_t words = 0;
    std::vector<int64_t> shape;
    ~Choice() {
        if (inner) {
            if (streaming) bytecrt_stream_free(inner);
            else wide_free(inner);
        }
    }
};
}
void* fherma_init(const fherma::Point& p) {
    auto s = std::make_unique<Choice>();
    if (!p.B || p.B > 128 || !p.N || !p.W || p.W > 4096 || p.L != (p.W + 31) / 32)
        throw std::runtime_error("invalid streaming point");
    s->streaming = (p.N == 65536 && p.W == 1747) ||
                   (p.B == 1 && p.N == 32768 && p.W == 868);
    s->shape = {p.B, p.N, p.L};
    s->words = size_t(p.B) * p.N * p.L;
    s->inner = s->streaming ? bytecrt_stream_init(p) : wide_init(p);
    return s.release();
}
fherma::Outputs fherma_run(void* state, const fherma::Inputs& in) {
    auto& s = *static_cast<Choice*>(state);
    if (in.a.shape != s.shape || in.b.shape != s.shape ||
        in.a.data.size() != s.words || in.b.data.size() != s.words)
        throw std::runtime_error("streaming input layout mismatch");
    if (s.streaming) return bytecrt_stream_run(s.inner, in);
    // This adapter times all transfers. Preserve the general arithmetic by
    // invoking its pure-copy helpers within this same timed operation.
    wide_h2d(s.inner, in);
    auto out = wide_run(s.inner, in);
    return wide_d2h(s.inner, std::move(out));
}
void fherma_free(void* state) { delete static_cast<Choice*>(state); }

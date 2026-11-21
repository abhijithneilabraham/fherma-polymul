// General exact fallback retained from the wide-field candidate.
#define fherma_init wide_init
#define fherma_run wide_run
#define fherma_free wide_free
#define fherma_h2d wide_h2d
#define fherma_d2h wide_d2h
#define fherma_metrics wide_metrics
#include "fherma.h"
#include "field_setup.hpp"
#include <cupqc/bigint.hpp>
#include <cuda_runtime.h>
#include <limits>
#include <memory>
#include <chrono>

namespace {
constexpr unsigned threads = 256;
void check(cudaError_t status, const char* action) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
}
unsigned blocks(size_t work) {
    const size_t count = (work + threads - 1) / threads;
    if (count > 2147483647) throw std::runtime_error("grid limit exceeded");
    return unsigned(count);
}
// Eight lanes share wide values. The SDK provides 256-bit arithmetic in
// thread mode, so narrow points use one thread instead of an unavailable warp type.
template<unsigned K> struct Traits {
    static constexpr unsigned lanes = 8;
    using descriptor = decltype(cupqc::BitWidth<K * 32>() + cupqc::SM<800>() + cupqc::TPI<8>() + cupqc::Warp());
};
template<> struct Traits<8> {
    static constexpr unsigned lanes = 1;
    using descriptor = decltype(cupqc::BitWidth<256>() + cupqc::SM<800>() + cupqc::Thread());
};
template<unsigned K> using Descriptor = typename Traits<K>::descriptor;

__global__ void prepare(const uint32_t* a, const uint32_t* b, uint32_t* ab,
                        size_t count, unsigned n, unsigned l, unsigned k, unsigned log_n) {
    const size_t word = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (word >= 2 * count * k) return;
    const unsigned limb = word % k;
    const size_t coefficient = word / k, local = coefficient % count;
    const size_t output = (coefficient / count * count + local) * k + limb;
    const uint32_t* source = coefficient < count ? a : b;
    ab[output] = limb < l ? source[local * l + limb] : 0;
}
template<unsigned K>
__global__ void twist(uint32_t* ab, const uint32_t* factors, const uint32_t* q,
                      size_t count, unsigned n, unsigned log_n) {
    const size_t i = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) / Traits<K>::lanes;
    if (i >= 2 * count) return; // The whole cooperating group takes this branch.
    using Big = typename Descriptor<K>::bigint;
    typename Descriptor<K>::modulus modulus(q);
    Big value(ab, i), factor(factors, i % n);
    value.mul_montgomery(factor, modulus).store(ab, i);
}
template<unsigned K, bool DIF = false>
__global__ void butterfly(uint32_t* data, const uint32_t* twiddles, const uint32_t* q,
                          size_t work, unsigned n, unsigned half) {
    const size_t i = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) / Traits<K>::lanes;
    if (i >= work) return;
    using Big = typename Descriptor<K>::bigint;
    typename Descriptor<K>::modulus modulus(q);
    const size_t transform = i / (n / 2);
    const unsigned pair = i % (n / 2), j = pair % half;
    const size_t left = transform * n + pair / half * (2 * half) + j;
    Big u(data, left), v(data, left + half), prime(q);
    if constexpr (DIF) {
        Big sum = u.add_mod(v, prime);
        Big difference = u.sub_mod(v, prime);
        // omega^0 is Montgomery one: multiplying by it is the identity.
        if (j) {
            Big factor(twiddles, j * (n / (2 * half)));
            difference = difference.mul_montgomery(factor, modulus);
        }
        sum.store(data, left);
        difference.store(data, left + half);
    } else {
        if (j) {
            Big factor(twiddles, j * (n / (2 * half)));
            v = v.mul_montgomery(factor, modulus);
        }
        u.add_mod(v, prime).store(data, left);
        u.sub_mod(v, prime).store(data, left + half);
    }
}
template<unsigned K>
__global__ void product(const uint32_t* ab, uint32_t* c, const uint32_t* q,
                        size_t count, unsigned n, unsigned log_n) {
    const size_t i = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) / Traits<K>::lanes;
    if (i >= count) return;
    using Big = typename Descriptor<K>::bigint;
    typename Descriptor<K>::modulus modulus(q);
    Big a(ab, i), b(ab, count + i);
    a.mul_montgomery(b, modulus).store(c, i);
}
// The last DIF stages, spectral product, and first inverse DIT stages share
// the same contiguous tile. Keep both operands resident across this boundary.
// Each arithmetic group owns one butterfly; barriers are reached by every thread.
template<unsigned K, unsigned Tile>
__global__ void fused_middle(const uint32_t* ab, uint32_t* c,
                             const uint32_t* forward, const uint32_t* inverse,
                             const uint32_t* q, size_t count, unsigned n) {
    static_assert(Tile * K * 2 * sizeof(uint32_t) <= 48 * 1024, "shared memory limit");
    static_assert(threads / Traits<K>::lanes >= Tile / 2, "not enough arithmetic groups");
    __shared__ uint32_t a_tile[Tile * K];
    __shared__ uint32_t b_tile[Tile * K];
    const size_t base = size_t(blockIdx.x) * Tile;
    const unsigned group = threadIdx.x / Traits<K>::lanes;
    for (unsigned word = threadIdx.x; word < Tile * K; word += blockDim.x) {
        a_tile[word] = ab[base * K + word];
        b_tile[word] = ab[(count + base) * K + word];
    }
    __syncthreads();
    using Big = typename Descriptor<K>::bigint;
    typename Descriptor<K>::modulus modulus(q);
    Big prime(q);
    for (unsigned half = Tile / 2; half; half /= 2) {
        if (group < Tile / 2) {
            const unsigned j = group % half;
            const unsigned left = group / half * (2 * half) + j;
            // The two operands reuse the same twiddle but keep independent data.
            Big factor(forward, j * (n / (2 * half)));
            {
                Big u(a_tile, left), v(a_tile, left + half);
                Big difference = u.sub_mod(v, prime);
                u.add_mod(v, prime).store(a_tile, left);
                if (j) difference = difference.mul_montgomery(factor, modulus);
                difference.store(a_tile, left + half);
            }
            {
                Big u(b_tile, left), v(b_tile, left + half);
                Big difference = u.sub_mod(v, prime);
                u.add_mod(v, prime).store(b_tile, left);
                if (j) difference = difference.mul_montgomery(factor, modulus);
                difference.store(b_tile, left + half);
            }
        }
        __syncthreads();
    }
    if (group < Tile / 2) {
        for (unsigned i = group; i < Tile; i += Tile / 2) {
            Big a(a_tile, i), b(b_tile, i);
            a.mul_montgomery(b, modulus).store(a_tile, i);
        }
    }
    __syncthreads();
    for (unsigned half = 1; half < Tile; half *= 2) {
        if (group < Tile / 2) {
            const unsigned j = group % half;
            const unsigned left = group / half * (2 * half) + j;
            Big u(a_tile, left), v(a_tile, left + half);
            if (j) {
                Big factor(inverse, j * (n / (2 * half)));
                v = v.mul_montgomery(factor, modulus);
            }
            u.add_mod(v, prime).store(a_tile, left);
            u.sub_mod(v, prime).store(a_tile, left + half);
        }
        __syncthreads();
    }
    for (unsigned word = threadIdx.x; word < Tile * K; word += blockDim.x)
        c[base * K + word] = a_tile[word];
}
template<unsigned K>
__global__ void normalize(uint32_t* c, const uint32_t* factors, const uint32_t* q, size_t count, unsigned n) {
    const size_t i = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) / Traits<K>::lanes;
    if (i >= count) return;
    using Big = typename Descriptor<K>::bigint;
    typename Descriptor<K>::modulus modulus(q);
    Big value(c, i), factor(factors, i % n);
    value.mul_montgomery(factor, modulus).store(c, i);
}
__global__ void pack(const uint32_t* padded, uint32_t* output, size_t words, unsigned l, unsigned k) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < words) output[i] = padded[i / l * k + i % l];
}
struct State {
    uint32_t b, n, l, k, log_n;
    size_t count, words;
    uint32_t *a = nullptr, *b_input = nullptr, *ab = nullptr, *c = nullptr;
    uint32_t *q = nullptr, *forward = nullptr, *inverse = nullptr, *twist_table = nullptr, *scale = nullptr;
    cudaEvent_t stage[8]{};
    double output_allocation_ms = 0;
    ~State() {
        for (auto event : stage) if (event) cudaEventDestroy(event);
        for (auto p : {a, b_input, ab, c, q, forward, inverse, twist_table, scale}) cudaFree(p);
    }
};
void allocate(uint32_t*& p, size_t words) {
    if (words) check(cudaMalloc(&p, words * sizeof(uint32_t)), "allocate device words");
}
void constant(uint32_t*& p, const setup::Words& values) {
    allocate(p, values.size());
    if (!values.empty()) check(cudaMemcpy(p, values.data(), values.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), "copy constant");
}
template<unsigned K>
void execute(State& s) {
    check(cudaEventRecord(s.stage[0]), "record start");
    prepare<<<blocks(2 * s.count * K), threads>>>(s.a, s.b_input, s.ab, s.count, s.n, s.l, K, s.log_n);
    check(cudaGetLastError(), "prepare");
    check(cudaEventRecord(s.stage[1]), "record prepare");
    twist<K><<<blocks(2 * s.count * Traits<K>::lanes), threads>>>(s.ab, s.twist_table, s.q, s.count, s.n, s.log_n);
    check(cudaGetLastError(), "twist");
    check(cudaEventRecord(s.stage[2]), "record twist");
    constexpr unsigned tile = K <= 64 ? 64 : 32;
    const bool fuse = s.n >= tile;
    for (unsigned half = s.n / 2; half >= (fuse ? tile : 1); half /= 2) {
        butterfly<K, true><<<blocks(s.count * Traits<K>::lanes), threads>>>(s.ab, s.forward, s.q, s.count, s.n, half);
        check(cudaGetLastError(), "forward transform");
    }
    check(cudaEventRecord(s.stage[3]), "record forward");
    if (fuse) {
        fused_middle<K, tile><<<unsigned(s.count / tile), threads>>>(
            s.ab, s.c, s.forward, s.inverse, s.q, s.count, s.n);
        check(cudaGetLastError(), "fused transform middle");
    } else {
        product<K><<<blocks(s.count * Traits<K>::lanes), threads>>>(s.ab, s.c, s.q, s.count, s.n, s.log_n);
        check(cudaGetLastError(), "pointwise product");
    }
    check(cudaEventRecord(s.stage[4]), "record product");
    for (unsigned half = fuse ? tile : 1; half < s.n; half *= 2) {
        const size_t work = s.count / 2;
        butterfly<K><<<blocks(work * Traits<K>::lanes), threads>>>(s.c, s.inverse, s.q, work, s.n, half);
        check(cudaGetLastError(), "inverse transform");
    }
    check(cudaEventRecord(s.stage[5]), "record inverse");
    normalize<K><<<blocks(s.count * Traits<K>::lanes), threads>>>(s.c, s.scale, s.q, s.count, s.n);
    check(cudaGetLastError(), "normalize");
    check(cudaEventRecord(s.stage[6]), "record normalize");
    pack<<<blocks(s.words), threads>>>(s.c, s.a, s.words, s.l, K);
    check(cudaGetLastError(), "pack");
    check(cudaEventRecord(s.stage[7]), "record pack");
    check(cudaDeviceSynchronize(), "finish all device work");
}
} // namespace

void* fherma_init(const fherma::Point& p) {
    if (!p.B || !p.W || p.W > 4096 || p.L != (p.W + 31) / 32 || p.q.data.size() != p.L)
        throw std::runtime_error("unsupported or malformed point");
    auto state = std::make_unique<State>();
    auto& s = *state;
    s.b = p.B; s.n = p.N; s.l = p.L;
    s.k = p.L <= 8 ? 8 : p.L <= 32 ? 32 : p.L <= 64 ? 64 : 128;
    if (!p.N || size_t(p.B) > std::numeric_limits<size_t>::max() / p.N / s.k / 8)
        throw std::runtime_error("point exceeds address space");
    s.count = size_t(p.B) * p.N; s.words = s.count * s.l;
    setup::Words modulus(s.k, 0);
    std::copy(p.q.data.begin(), p.q.data.end(), modulus.begin());
    setup::Plan plan(p.N, std::move(modulus));
    s.log_n = plan.log_n;
    allocate(s.a, s.words); allocate(s.b_input, s.words);
    allocate(s.ab, 2 * s.count * s.k); allocate(s.c, s.count * s.k);
    constant(s.q, plan.field.q); constant(s.forward, plan.forward); constant(s.inverse, plan.inverse);
    constant(s.twist_table, plan.twist_r2); constant(s.scale, plan.scale_normal);
    for (auto& event : s.stage) check(cudaEventCreate(&event), "create stage event");
    return state.release();
}
fherma::Outputs fherma_run(void* state, const fherma::Inputs& input) {
    auto& s = *static_cast<State*>(state);
    const std::vector<int64_t> shape = {s.b, s.n, s.l};
    if (input.a.shape != shape || input.b.shape != shape || input.a.data.size() != s.words || input.b.data.size() != s.words)
        throw std::runtime_error("input layout does not match point");
    fherma::Outputs output;
    const auto allocation_start = std::chrono::steady_clock::now();
    output.c.shape = shape; output.c.data.resize(s.words);
    s.output_allocation_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - allocation_start).count();
    switch (s.k) {
        case 8: execute<8>(s); break;
        case 32: execute<32>(s); break;
        case 64: execute<64>(s); break;
        case 128: execute<128>(s); break;
        default: throw std::runtime_error("invalid limb dispatch");
    }
    return output;
}
// Transfer hooks preserve the packed host representation byte for byte. Every
// transform and pack kernel remains in execute(), which synchronizes before return.
void fherma_h2d(void* state, const fherma::Inputs& input) {
    auto& s = *static_cast<State*>(state);
    const std::vector<int64_t> shape = {s.b, s.n, s.l};
    if (input.a.shape != shape || input.b.shape != shape || input.a.data.size() != s.words || input.b.data.size() != s.words)
        throw std::runtime_error("input layout does not match point");
    check(cudaMemcpy(s.a, input.a.data.data(), s.words * sizeof(uint32_t), cudaMemcpyHostToDevice), "copy a");
    check(cudaMemcpy(s.b_input, input.b.data.data(), s.words * sizeof(uint32_t), cudaMemcpyHostToDevice), "copy b");
}
fherma::Outputs fherma_d2h(void* state, fherma::Outputs output) {
    auto& s = *static_cast<State*>(state);
    check(cudaMemcpy(output.c.data.data(), s.a, s.words * sizeof(uint32_t), cudaMemcpyDeviceToHost), "copy answer");
    return output;
}
std::vector<std::pair<std::string, double>> fherma_metrics(void* state) {
    auto& s = *static_cast<State*>(state);
    const char* names[] = {"prepare_ms", "twist_ms", "forward_outer_ms", "fused_middle_or_product_ms", "inverse_outer_ms", "normalize_ms", "pack_ms"};
    std::vector<std::pair<std::string, double>> result;
    for (unsigned i = 0; i < 7; ++i) {
        float ms = 0;
        check(cudaEventElapsedTime(&ms, s.stage[i], s.stage[i + 1]), "read stage time");
        result.emplace_back(names[i], ms);
    }
    result.emplace_back("output_allocation_ms", s.output_allocation_ms);
    return result;
}
void fherma_free(void* state) { delete static_cast<State*>(state); }

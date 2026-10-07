// src/kernels/cpu/x86_kernels_absent.cpp - the aarch64 build's stand-ins for Strata's x86 expert kernels
// (docs/DGX_SPARK.md).  expert.cpp, q2_avx2.cpp, iq_avx2.cpp, iq_avx512.cpp, kq_avx2.cpp and kq_avx1.cpp are not
// compiled there: the CPU probes (expert_layout.cpp) say no to every x86 feature, so the engine takes ggml-cpu's
// dot products for native packs, the plain C++ router dot, and refuses the canonical AVX-512 Q2_0 pack - the same
// paths an older-CPU (STRATA_ISA_FLOOR) engine takes on a CPU without AVX2.  What is still called on those paths is
// real code here (the probes' answers, the scalar activation quantizer, the oracle flag); a kernel that only runs
// behind a "yes" from a probe aborts with its name, so a missed check shows up as that instead of a wrong answer.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/kq_avx1.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels::cpu {
namespace {

std::atomic<bool> oracle_q8_0{false};

[[noreturn]] void absent(const char* name) {
    std::fprintf(stderr, "strata: %s is an x86 kernel and this is the aarch64 build; it should not have been "
                         "called (a CPU check is missing)\n", name);
    std::abort();
}

}  // namespace

const char* CpuFeatures::reason() const { return "not an x86 CPU (aarch64 build)"; }

CpuFeatures cpu_features() { return {}; }

void cpu_require_expert_support() {
    std::fprintf(stderr,
                 "strata: the canonical Q2_0 expert pack (experts.bin) needs Strata's AVX-512 kernels, which an\n"
                 "        aarch64 CPU does not have. Use a native (GGUF) pack instead: IQ2_XS, IQ3_XXS, IQ3_S or the\n"
                 "        Unsloth UD-IQ4_XS (docs/DGX_SPARK.md).\n");
    std::exit(1);
}

void expert_set_oracle_q8_0(bool enabled) { oracle_q8_0.store(enabled, std::memory_order_relaxed); }

bool expert_oracle_q8_0_enabled() { return oracle_q8_0.load(std::memory_order_relaxed); }

// expert.cpp's scalar loop, unchanged: round half away from zero, clamp to +-127.  The oracle Q8_0 quantizer is only
// for the S2 kernels of the canonical pack, which this build refuses.
void act_quant_q8_1(const float* x, int n, ActQ& a) {
    a.nchunks = n / QKA;
    for (int k = 0; k < a.nchunks; ++k) {
        const float* xb = x + k * QKA;
        float amax = 0.f;
        for (int j = 0; j < QKA; ++j) amax = std::fmax(amax, std::fabs(xb[j]));
        const float s = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = s > 0.f ? 1.f / s : 0.f;
        int32_t sum = 0;
        int8_t* q = a.q + k * QKA;
        for (int j = 0; j < QKA; ++j) {
            const float t = xb[j] * inv;
            const float r = t + (t >= 0.f ? 0.5f : -0.5f);
            int v = (int) r;
            v = v < -127 ? -127 : (v > 127 ? 127 : v);
            q[j] = (int8_t) v;
            sum += v;
        }
        a.scale[k] = s;
        a.sum[k] = sum;
        a.hx[k] = s * (float) sum;
    }
}

void act_quant_q8_1_avx2(const float* x, int n, ActQ& a) { act_quant_q8_1(x, n, a); }

bool q2_bitplane_enabled() { return false; }
bool iq256_supported(int) noexcept { return false; }
bool iq512_supported(int) noexcept { return false; }
bool kq256_supported(int) noexcept { return false; }
int iq256_variant() noexcept { return 0; }
int iq256_variants() noexcept { return 0; }

void s2_expert_vnni(const uint8_t*, const float*, float*, ExpertScratch&) { absent("s2_expert_vnni"); }
void s2_expert_vnni_q(const uint8_t*, const ActQ&, float*, ExpertScratch&) { absent("s2_expert_vnni_q"); }
void s2_expert_gu_rows(const uint8_t*, const ActQ&, float*, int, int) { absent("s2_expert_gu_rows"); }
void s2_expert_down_rows(const uint8_t*, const ActQ&, float*, int, int) { absent("s2_expert_down_rows"); }
void s2_expert_gu_rows_multi(const uint8_t*, const ActQ* const*, int, float* const*, int, int) {
    absent("s2_expert_gu_rows_multi");
}
void s2_expert_down_rows_multi(const uint8_t*, const ActQ* const*, int, float* const*, int, int) {
    absent("s2_expert_down_rows_multi");
}
void s2_expert_vnni_multi(const uint8_t*, const ActQ* const*, int, float* const*, ExpertScratchMulti&) {
    absent("s2_expert_vnni_multi");
}
void s2_expert_scalar(const uint8_t*, const float*, float*, bool) { absent("s2_expert_scalar"); }
void q2_0_gguf_rows_multi(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    absent("q2_0_gguf_rows_multi");
}
void q2_0_gguf_rows_multi_avx2(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    absent("q2_0_gguf_rows_multi_avx2");
}
void q2_0_gguf_rows_multi_avx2_legacy(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    absent("q2_0_gguf_rows_multi_avx2_legacy");
}
void q2_0_gguf_rows_multi_avx2_v(bool, const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int,
                                 int) {
    absent("q2_0_gguf_rows_multi_avx2_v");
}

void iq256_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    absent("iq256_gu_rows");
}
void iq256_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    absent("iq256_rows");
}
void iq256_gu_rows_v(int, int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int,
                     int) {
    absent("iq256_gu_rows_v");
}
void iq256_rows_v(int, int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    absent("iq256_rows_v");
}
void q8k_quant_avx2(const float*, void*, int64_t) { absent("q8k_quant_avx2"); }
void iq4nl256_down_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    absent("iq4nl256_down_rows");
}
void iq4nl256_down_rows_v(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    absent("iq4nl256_down_rows_v");
}
void iq512_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    absent("iq512_gu_rows");
}
void iq512_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    absent("iq512_rows");
}
void kq256_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    absent("kq256_gu_rows");
}
void kq256_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    absent("kq256_rows");
}
void bf16_rows_dot(const uint16_t*, int, int, const float*, float*) { absent("bf16_rows_dot"); }
void bf16_rows_dot_multi(const uint16_t*, int, int, const float*, int, float*) { absent("bf16_rows_dot_multi"); }
void bf16_rows_dot_multi_avx1(const uint16_t*, int, int, const float*, int, float*) {
    absent("bf16_rows_dot_multi_avx1");
}

}  // namespace strata::kernels::cpu

// include/strata/platform/spin.hpp - `_mm_pause` and `_mm_sfence` on every CPU Strata builds for.
//
// x86: exactly <immintrin.h>.  aarch64 (the DGX Spark, docs/DGX_SPARK.md): `yield` for the spin-wait hint, and
// `dsb st` for the store fence: the fences order the host's stores ahead of a doorbell the GPU reads from mapped host
// memory, so the barrier waits for them to complete system-wide (a `dmb` would only order them for other CPUs).
#pragma once

#if defined(STRATA_ARM64) && defined(__aarch64__)
static inline void _mm_pause() { __asm__ __volatile__("yield" ::: "memory"); }
static inline void _mm_sfence() { __asm__ __volatile__("dsb st" ::: "memory"); }
#else
#include <immintrin.h>
#endif

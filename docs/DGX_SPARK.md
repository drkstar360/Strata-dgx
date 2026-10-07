# NVIDIA DGX Spark (GB10, aarch64): build from source

This fork ports Strata to the NVIDIA DGX Spark: a 20-core ARM Grace CPU (aarch64, NEON/SVE2, no x86 instructions), a Blackwell
GB10 GPU (compute capability 12.1, `sm_121`) and 128 GB of LPDDR5X that the CPU and the GPU share. Upstream Strata targets x86-64
CPUs with discrete GPUs; every change for the Spark sits behind an aarch64 check so the x86 build stays as it is.
**Status: in progress.** Everything here was measured on one machine, a DGX Spark running DGX OS (Ubuntu 24.04.5), driver
580.178.04, CUDA 13.0.88.

## What the Spark reports (recon, 2026-10-08)

| Query | Result |
| --- | --- |
| `uname -m` | `aarch64` |
| `free -g` | 121 GiB total, 119 GiB available (nothing else running) |
| `nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version` | `0, NVIDIA GB10, [N/A], 12.1, 580.178.04` |
| `nvidia-smi --query-gpu=memory.used,memory.free` | `[N/A], [N/A]` |
| `nvidia-smi` memory column | `Not Supported` |
| `nvidia-smi --query-gpu=pcie.link.gen.max,pcie.link.gen.gpumax,pcie.link.gen.hostmax` | `1, 5, 5` (no PCIe hop between CPU and GPU memory, so meaningless here) |
| `nvcc --version` | 13.0, V13.0.88, at `/usr/local/cuda/bin` (not on `PATH`) |
| CPU parts (`/proc/cpuinfo`) | `0xd85` Cortex-X925 and `0xd87` Cortex-A725, 20 cores |
| CPU features | `asimd asimddp sve sve2 i8mm bf16 svei8mm svebf16 asimdhp fphp ...` (no `flags` line, no x86 flags) |
| Compilers | gcc/g++ 13.3.0, clang++ 15 and 18 |
| CMake / Ninja / Python | 3.28.3 / system ninja / 3.12.3 |
| Docker | 29.6.2, `nvidia` runtime installed |
| `ulimit -l` | unlimited |

Two of these drive the port:

- **`memory.total` is `[N/A]`.** Upstream setup parses it as a number, so it would skip the GB10 and stop with "no NVIDIA GPU
  found". On the Spark, setup has to treat the GPU as unified memory and size it from `MemAvailable`.
- **GCC 13 does not know the GB10 cores.** `-mcpu=native` falls back to generic ARMv8: it defines 5 `__ARM_FEATURE_*` macros and
  no dotprod, i8mm or SVE, so ggml's ARM feature checks (`HAVE_DOTPROD`, `HAVE_SVE`, `HAVE_MATMUL_INT8`) all fail and ggml-cpu
  would build without its fast paths. `-march=armv9-a+i8mm+bf16` turns on all of them (DOTPROD, MATMUL_INT8, SVE, SVE2, BF16,
  FP16 vector arithmetic) with the same GCC.

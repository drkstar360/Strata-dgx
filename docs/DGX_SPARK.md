# NVIDIA DGX Spark (GB10, aarch64): build from source

This fork ports Strata to the NVIDIA DGX Spark: a 20-core ARM Grace CPU (aarch64, NEON/SVE2, no x86 instructions), a Blackwell
GB10 GPU (compute capability 12.1, `sm_121`) and 128 GB of LPDDR5X that the CPU and the GPU share. Upstream Strata targets x86-64
CPUs with discrete GPUs; every change for the Spark sits behind an aarch64 check, so the x86 build stays as it is (checked below).
**Status: experimental, no model run yet.** The engine and setup's own engine build and pass the tests on the Spark; the API,
output quality and speed have not been measured. Everything here was measured on one machine, a DGX Spark running DGX OS (Ubuntu
24.04.5), driver 580.178.04, CUDA 13.0.88, GCC 13.3.0, on 2026-10-08.

## What the Spark reports

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
| CPU features | `asimd asimddp sve sve2 i8mm bf16 svei8mm svebf16 asimdhp fphp ...` (no `flags` line, no model name) |
| Compilers | gcc/g++ 13.3.0, clang++ 15 and 18 |
| CMake / Ninja / Python | 3.28.3 / system ninja / 3.12.3 |
| Docker | 29.6.2, `nvidia` runtime installed |
| `ulimit -l` | unlimited |

CUDA 13.0 is the toolkit DGX OS ships, and the one to use: upstream documents CUDA 13.2.0 / 13.2.1 (nvcc 13.2.51) miscompiling
Blackwell sm_120 kernels.

## What the port changes

**The engine (CMake and C++).** On aarch64 (`CMAKE_SYSTEM_PROCESSOR` aarch64/arm64, which defines `STRATA_ARM64=1` for every
target):

- None of Strata's x86 CPU kernels are compiled: `expert.cpp`, `q2_avx2.cpp`, `iq_avx2.cpp`, `iq_avx512.cpp`, `kq_avx2.cpp`,
  `kq_avx1.cpp`, the AVX-VNNI compiler probe, and the parity programs that test those kernels (`router_dot_parity`,
  `iq_avx2_parity`, `q8k_quant_parity`, `expert_multi_test`, `expert_parity`, `pool_test`, `pool_stress`).
- The CPU probes in `expert_layout.cpp` (`cpu_avx2_ok`, `cpu_avx512_ok`, ...) say no to every x86 feature. The engine then
  takes the paths a CPU without AVX2 takes on the older-CPU build (`STRATA_ISA_FLOOR`): ggml-cpu's dot products for the CPU's
  expert rows, the plain C++ router dot. `src/kernels/cpu/x86_kernels_absent.cpp` stands in for the kernels at link time; a
  kernel that should never be called aborts with its name.
- The canonical Q2_0 pack (`experts.bin`) needs Strata's AVX-512 kernels, so the engine refuses it with a message. Native (GGUF)
  packs run: the pinned ggml has NEON dot products for every format they use (IQ2_XS, IQ2_XXS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL,
  IQ4_XS, Q4_K, Q5_K, Q6_K, Q8_0, Q2_0).
- `_mm_pause` and `_mm_sfence` come from `include/strata/platform/spin.hpp`: `<immintrin.h>` on x86, `yield` and `dsb st` on
  aarch64. `dsb st` is the conservative choice, because these fences order the host's stores before a doorbell the GPU reads.
- ggml-cpu gets `-march` from this CPU's `/proc/cpuinfo` features (`cmake/arm_arch.cmake`, used by the vision encoder too). On
  the Spark that is `-march=armv8.2-a+dotprod+fp16+i8mm+bf16+sve+sve2`. `-DSTRATA_ARM_ARCH=<march>` overrides it, and
  `-DSTRATA_ARM_ARCH=native` keeps ggml's own detection.
- Host C/C++ is compiled with `-ffp-contract=off`, as x86 builds behave (see "Two findings").
- `CMAKE_CUDA_ARCHITECTURES` defaults to `121`.
- `STRATA_PORTABLE` and `STRATA_ISA_FLOOR` are refused on aarch64: both are x86 builds.

**Setup (`setup.py`).** `ARM64` is true when `platform.machine()` is aarch64/arm64. Then:

- The CPU is named from `lscpu` ("Cortex-X925 + Cortex-A725"). It is never sent to the x86 floor checks, which would have
  stopped setup with "this CPU has neither AVX2 nor SSE4.2".
- The engine is always compiled on the Spark. The ready-made engines are x86-64, so none is downloaded.
- A GPU whose `memory.total` is `[N/A]` (GB10) counts as unified memory: usable GPU memory = RAM less the 6 GB the OS keeps
  (`UMA_OS_LEFT_GB`, the same figure the engine's `device_free_bytes()` leaves), and nothing beside the RAM. So the RAM is
  never counted twice. Upstream setup skipped such a GPU and stopped with "no NVIDIA GPU found".
- The PCIe probe is skipped for it.
- Q2_0 is not offered. Its pack needs AVX-512.
- Unsloth UD-IQ4_XS is recommended (from 80 GB of memory, the Strix Halo rule).
- The CUDA keyring URL is NVIDIA's `sbsa` repository, never `x86_64`. A DGX OS without nvcc is told to restore its own toolkit.

**`Dockerfile.spark`.** `./Dockerfile` with `CUDA_ARCHITECTURES=121`, the same entrypoint, environment variables and `/data`
volume. **Not built yet.**

**Line endings.** `.gitattributes` pins `Dockerfile*`, `*.py`, `*.cmake` and `CMakeLists.txt` to LF, so a Windows clone with
`core.autocrlf` does not break the Dockerfile heredoc.

## Build it

### Bare metal

```sh
git clone -b spark-port https://github.com/drkstar360/Strata-dgx.git ~/Strata
cd ~/Strata
./setup.sh                      # or unattended: ./setup.sh --yes --family unsloth --model UD-IQ4_XS --no-start
```

Setup finds nvcc in `/usr/local/cuda` by itself. Step 1 printed this on the Spark:

```
  [ok] GPU: NVIDIA GB10, unified memory: shares the RAM, 116 GB usable by the GPU, compute capability 12.1, driver 580.178.04 (docs/DGX_SPARK.md)
  [ok] RAM: 122 GB
  [ok] CPU: Cortex-X925 + Cortex-A725 (ARM64: the engine is compiled here, its CPU experts on ggml-cpu)
```

The UD-IQ4_XS menu line says "keeps ~55 GB of its 60 GB of experts in RAM". 59.5 GB is 55.4 GiB, so that is all of them. The
engine setup built recorded `"archs": [121]` in `engine/BUILD.json`.

By hand, without setup:

```sh
export PATH=/usr/local/cuda/bin:$PATH
cmake -S . -B build-arm -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON
cmake --build build-arm -j 20
cd build-arm && ctest --output-on-failure
```

The first configure takes about 4 minutes (218 s on the Spark), most of it cloning the pinned llama.cpp.

### Docker

Build and run it **on the Spark**. The base image is multi-arch, so docker pulls arm64 there; Docker Desktop on an x86 PC would
make an amd64 image.

```sh
docker build -f Dockerfile.spark -t strata-spark .
docker image inspect strata-spark --format '{{.Architecture}}'      # arm64
docker run -d --name strata --gpus all -p 127.0.0.1:8080:8080 --ulimit memlock=-1 \
  -v strata-data:/data -e FAMILY=unsloth -e MODEL=UD-IQ4_XS strata-spark
```

Add `-e API_KEY=<secret>` before publishing the port beyond `127.0.0.1`.

Or with Docker Compose (`docker-compose.spark.yml`), which keeps the data in `strata-data/` in the project root instead of a
named volume. `.dockerignore` and `.gitignore` leave that folder out, so the model files never go into the image or into git:

```sh
mkdir -p strata-data                                      # created by you, so it is not owned by root
docker compose -f docker-compose.spark.yml up -d --build
docker compose -f docker-compose.spark.yml logs -f
curl -fs http://127.0.0.1:8080/health
```

Its settings come from the environment or a `.env` file in the project root: `cp .env.example .env` and edit it. The
example lists every setting with what it does; `.env` holds the API key, so `.gitignore` and `.dockerignore` leave it out.
The settings: `FAMILY` (default `unsloth`), `MODEL` (`UD-IQ4_XS`), `CONTEXT`
(`65536`), `VISION` (`no`), `API_KEY`, `GGUF_DIR`, `KV`, `MODEL_ALIASES`, `REINSTALL`, `STRATA_PORT` (`8080`),
`BUILD_VISION` (`1`). `MODEL_ALIASES=qwen,local-model` gives the model other names: `/v1/models` lists them, and a
request naming one is answered under it. The entrypoint writes them into the config's `aliases` on every start (the
same key the web page's About tab edits; docs/DETAILS.md "Model aliases"). Left empty, the config's own aliases stay. For GGUF
files already on the Spark, uncomment the `/ggufs` mount in the file and set `GGUF_DIR=/ggufs`. The container runs as root, so
the files it writes in `strata-data/` are owned by root.

## Applying the port to a newer upstream

`patches/dgx-spark.patch` is the whole port as one patch against upstream `main` (Niko1221/Strata). It holds every change
listed above, this document included. Applied to upstream commit `e8ca9af` it gives exactly this fork's tree.

On a fresh upstream checkout:

```sh
git clone https://github.com/Niko1221/Strata.git && cd Strata
git apply --3way /path/to/dgx-spark.patch
git status                       # every file it touched; a conflict is marked in the file like a merge conflict
```

`--3way` falls back to a three-way merge where upstream has changed the same lines since. That works because the patch
names the upstream file versions it was made from, which every upstream clone has.

In this fork, merging upstream keeps the history and is usually simpler: `git fetch upstream && git merge upstream/main`.
Afterwards, regenerate the patch so it matches the new upstream (it leaves `patches/` itself out):

```sh
git diff --binary upstream/main main -- . ':(exclude)patches' > patches/dgx-spark.patch
```

## Results so far

**Build.** The engine compiles and links on the Spark: an `ELF 64-bit ... ARM aarch64` executable, with GPU code for
`compute_121` only. The compile log has 0 hits for `-mavx`, `-mf16c` and `-mfma`.

**Tests** (`ctest -j 4` on the Spark):

| Run | Passed | Failed |
| --- | --- | --- |
| first build | 83 / 87 | `qsa_select_bench`, `quantize_act_parity`, `expert_cache_segmented_test`, `ple_parity` |
| with `-ffp-contract=off` | 84 / 87 | `ple_parity`, `expert_cache_segmented_test`, `ple_reader_selftest` |

What is left in the second run:

- **`ple_parity`** needs the Q2_0 model file (`../../Q2_0/...gguf`), which was not on the Spark. It fails the same way on any PC
  without it.
- **`expert_cache_segmented_test`** failed in both parallel runs and passed run alone (0.78 s).
- **`ple_reader_selftest`** failed after 156.57 s while setup was compiling and downloading next to it; it passed alone (106.99 s).

**x86 unchanged.** On Windows (MSVC 14.42, CMake 4.4.3, Ninja), upstream `main` and this branch were configured the same way and
their `compile_commands.json` compared. Both have the same 66 compile commands, with identical flags: the same `/arch:AVX512`,
`/arch:AVX2` and `/arch:AVX` on the same files, and identical configure messages. This covers the CPU side only, configured with
`STRATA_ENABLE_CUDA=OFF`, because that PC has no Windows SDK and CUDA's compiler check needs to link. A full x86 build has not
been run.

## Two findings

- **GCC 13 does not know the GB10 cores.** `-mcpu=native` falls back to generic ARMv8: it defines 5 `__ARM_FEATURE_*` macros and
  no dotprod, i8mm or SVE. ggml's ARM checks (`HAVE_DOTPROD`, `HAVE_SVE`, `HAVE_MATMUL_INT8`) all failed, so ggml-cpu would have
  built without its fast paths. `-march=armv8.2-a+dotprod+fp16+i8mm+bf16+sve+sve2` defines 21, including DOTPROD,
  MATMUL_INT8, SVE, SVE2 and BF16. Hence `cmake/arm_arch.cmake`.
- **GCC on aarch64 fuses `a*b+c` into one FMA by default** (`-ffp-contract=fast`). An x86 build without `-mfma` never does.
  ggml's `nearest_int(iscale * x)` is `iscale*x + 12582912.0f`. Fused, the product is not rounded first, so an exact .5 tie
  rounds the other way. `quantize_act_parity`'s Q8_K reference got -37 where the GPU kernel (and x86) gives -38 (block 343,
  element 144, `iscale*x` = -37.5). `qsa_select_bench`'s fast-scorer error was 0.000273 against a limit of about 0.000269 (1e-6
  of the score scale 269). Both pass with `-ffp-contract=off`. NEON code with explicit FMA intrinsics, and device code (nvcc's
  `--fmad`), are not affected.

The pinned CMake default of `CMAKE_CUDA_ARCHITECTURES` 120 never applies upstream either. `enable_language(CUDA)` runs first and
sets nvcc's default, 75 for CUDA 13. The first Spark build compiled `compute_75` for that reason. Setup always passes the
architecture, so x86 users are not affected. On aarch64 the 121 default is set before `enable_language(CUDA)`.

## Not done yet

- **A model run.** No model was downloaded. The API checks, temperature-0 output checks (code, a math answer, a 32K-token recall
  test) and decode / prompt tok/s at a short prompt and at 32K are still to do.
- **`Dockerfile.spark`** and **`docker-compose.spark.yml`** have not been built or run.
- **Pinning on unified memory.** The engine page-locks tens of GB of host RAM for the GPU (`cudaHostRegister`, `mlock`). On the
  Spark host and GPU memory are one pool, so this may be pointless. Startup time, peak RSS and tok/s with the pinning as it is,
  and with less (`STRATA_RESIDENT_PIN=0`, `STRATA_ARENA_LOCK=0`, `STRATA_ARENA_PIN_GIB`), are to be measured.
- **The expert cache on unified memory.** `device_free_bytes()` already counts `MemAvailable` less `STRATA_UMA_HEADROOM_GIB`
  (default 6) for any integrated GPU on Linux, on the CUDA path too (`src/core/expert_cache.cpp`, read, not measured). So
  `--expert-cache auto` may copy experts that are already in the same RAM. Whether that helps or only uses memory is to be
  measured.
- **The mixed cores.** The 20 cores are 10 Cortex-X925 and 10 Cortex-A725, and the expert pool pins one worker per core. CPU expert
  work may wait on the slower cores. Not measured.
- **Q2_0 as a native GGUF.** ggml has a NEON Q2_0 dot product, so the native Q2_0 GGUF (not the canonical pack) may run. Setup
  hides Q2_0 on ARM until it is tested.

# Jetson Thor (SM110)

Flare admits Jetson Thor (compute capability 11.0) as a preparation and serving
device. Recipe preparation on Thor is not yet available: every current recipe's
prepared paths run b12x kernels that cannot execute on SM110. This document
records what works on the hardware, what remains blocked, and the paths to
full recipe support. The device-admission change and its validation are in
commit `633d58c8`; the pinned FlashInfer is `hengwen1/flashinfer` at `jthor`.

## Device

- Thor reports compute capability `(11, 0)`, 20 SMs, and the device name
  `NVIDIA Thor` through NVML and CUDA. It is an integrated GPU sharing
  physical DRAM with the CPU.
- NVML raises `NotSupportedError` for GPU memory on Thor, as on GB10. Ignition
  identifies the integrated device through CUDA and plans against Linux
  `MemAvailable` capped by CUDA's device total. `discover_gpus()` resolves
  123 GiB total on this unit through that path.
- This host's `nvidia-smi` CLI hangs indefinitely, while the NVML Python API
  Ignition uses responds normally. Inspect the device through the CUDA and
  NVML Python bindings.
- The runtime is architecture-agnostic: packages record the prepared
  capability as `[major, minor]` data, Helios validates it at load, and Orbit
  names the hardware revision `sm110` through its capability fallback.

The surrounding stack is ready: torch 2.14.1+cu132 ships sm_110 SASS, Triton
3.8.0 compiles for sm_110a, the cuTile toolchain carries sm_110 targets, and
the CuTe DSL resolves this device to sm_110a.

## Kernel support on SM110

Measured on one Thor unit with the pinned FlashInfer. The fork's b12x GPU
suites skip non-SM12x devices by default; these results ran with a local
conftest relaxation that is not committed.

| Path | Status | Evidence |
| --- | --- | --- |
| FlashInfer SM110 attention (`sm110_xqa`, five kernel families including tcgen05/TMEM) | works | 306 parametrized cases pass the numerical and runtime suite; 35 skip on the `pair` route's documented Q-tile geometry constraint; covers contiguous and page-128 paged KV, FP16 and E4M3 KV, and packed draft-tree queries with prepared-plan replay |
| b12x paged decode attention | works | 55/55 cases pass |
| b12x GEMM without block-scaled FP4 (plain FP8, `mma.sync` since SM89) | works | 14 cases of the blockscaled suite pass; every failure is FP4-flavored |
| b12x block-scaled FP4/FP6/MXFP8 GEMM | blocked | kernels emit the SM120-family tcgen05 block-scale MMA; ptxas rejects it on `.target sm_110a` |
| b12x NVFP4 MoE | blocked | rejects SM110 explicitly; same instruction family |
| NVFP4 quantization | works | `fp4_quantization_110` JIT module builds and runs |
| NVFP4 dense GEMM (`mm_fp4`) | works | `cutlass`, `cudnn`, and `auto` backends reach cosine 0.991 against bf16 at 64×256×128; the `trtllm` backend is not claimed on capability 110 |
| NVFP4 fused MoE (`CutlassNvfp4Config`) | works | `CutlassNvfp4Runner` end-to-end (8 experts, top-2, SwiGLU): the suite accuracy gate passes with 1.0000 element match at atol 0.05 |

The blocked paths share one cause: b12x builds on the SM120-family warp-MMA
operations of the pinned CuTe DSL, whose block-scaled instruction does not
assemble for sm_110a. Thor's FP4 tensor cores are reachable through
SM100-family kernels instead — FlashInfer's CUTLASS NVFP4 GEMM and MoE target
that family, and FlashInfer ships prebuilt sm_110a cubins for the trtllm-gen
FMHA, BMM, GEMM, and DeepGEMM artifact sets.

FlashInfer's experimental `sm110_xqa` module provides Thor-native attention
with prepared-plan semantics: `prepare()` before graph capture, allocation-free
`run()` on the caller's stream, an explicit workspace contract, and frozen
route selection recorded on the plan.

## Recipe preparation

Every current recipe fails preparation on Thor. Projections and routed experts
prepare through b12x block-scaled kernels (`blockscaled` linear weights,
`nvfp4_csf` MoE), which cannot compile for SM110. MiniMax M3 rejects SM110
explicitly; other recipes fail inside FlashInfer with the CuTe DSL arch check
or the ptxas error. Weight loading completes before those failures, so the
error surfaces minutes into preparation.

Kernel availability does not imply serving support. Per-recipe qualification
on Thor — measured numerical bounds, preparation and serving memory under
unified memory, latency under pressure, and every prepared capacity — remains
outstanding, and SM120/SM121 results do not establish SM110 behavior.

## Paths to recipe support

Two routes exist; both keep the architecture boundaries in
[the core design](core-design.md).

1. **Ignition SM110 components over FlashInfer mainline.** Wrap `mm_fp4` and
   `MoELayer` with `CutlassNvfp4Config` as SM110 component variants, and
   `sm110_xqa` prepared plans where a recipe's attention contract matches.
   This is Python-side component and recipe work within the ordinary
   contribution rules. The weight-preparation adapter is the main cost:
   Flare's CSF-compressed FP4 checkpoint scales must convert to the TRTLLM
   canonical NVFP4 pack (`prepare_weights`; hidden size divisible by 64,
   intermediate size by 16). Graph capture, tensor-parallel sharding, and
   per-recipe numerical bounds need verification; the fork's accuracy gates do
   not substitute for Flare's measured bounds.
2. **SM100-family kernels in b12x.** Adding SM110 to b12x's block-scaled paths
   requires new CuTe DSL kernels targeting the SM100 family plus tuning tables
   validated on Thor. That is kernel development in the pinned fork and falls
   under [the CUDA kernel authorship restriction](../AGENTS.md); route it to
   the qualified maintainer.

## Working on the Thor host

- Setup follows [the preparation guide](model-preparation.md#environment-setup).
  The build additionally needs `libssl-dev`, `liburing-dev`, and
  `protobuf-compiler`, and flashinfer's JIT builds need `ninja` on `PATH`.
- The flashinfer JIT cache lives under `~/.cache/flashinfer`. The first build
  of the CUTLASS fused-MoE module took roughly forty minutes on this 20-core
  ARM host; `FLASHINFER_NVCC_THREADS=4` parallelizes compilation units and
  ninja resumes interrupted builds.
- `/tmp` does not survive reboots. Keep probe scripts elsewhere.

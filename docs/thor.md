# Jetson Thor (SM110)

Flare admits Jetson Thor (compute capability 11.0) as a preparation and serving
device, and Qwen3.8-Flash-Next prepares, exports, and serves on it through the
[Ignition SM110 components](thor-components.md). Every other recipe still fails
preparation because its prepared paths run b12x kernels that cannot execute on
SM110. This document records what works on the hardware, how to build and run
there, what remains blocked, and the paths to full recipe support. The
device-admission change and its validation are in commit `633d58c8`; the pinned
FlashInfer is `hengwen1/flashinfer` at `jthor`.

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
  names the hardware revision `sm110` through its capability fallback. Package
  export naming maps NVML's `0x2b00` device id to the `thor` platform.

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

Qwen3.8-Flash-Next prepares, exports, and serves on Thor through the
[Ignition SM110 components](thor-components.md). `ignition export
qwen3.8-flash` writes a package named for the device (`thor-tp1-...`), and
`flare run` loads it and answers requests: on this unit a 65-token chat request
returned the correct answer, a 5652-token prompt prefilled through chunked
prefill in 1.2 s, and repeated requests reported prefix-cache hits. Loading
reports 86.78 GiB of weights and a 23.89 GiB cache budget (1,945,600 KV tokens)
limited by unified memory.

Other recipes still fail preparation on Thor. Their projections and routed
experts prepare through b12x block-scaled kernels (`blockscaled` linear
weights, `nvfp4_csf` MoE), which cannot compile for SM110. MiniMax M3 rejects
SM110 explicitly; other recipes fail inside FlashInfer with the CuTe DSL arch
check or the ptxas error. Weight loading completes before those failures, so
the error surfaces minutes into preparation.

Kernel availability does not imply serving support. Per-recipe qualification
on Thor — measured numerical bounds, preparation and serving memory under
unified memory, latency under pressure, and every prepared capacity — remains
outstanding beyond this export and serving pass, and SM120/SM121 results do not
establish SM110 behavior.

## Build and run

Run commands from the repository root. Every number below was measured on one
Thor unit; treat it as that unit's behavior rather than a general claim.

### Host prerequisites

Thor shares one 122 GiB DRAM pool between CPU and GPU, so preparation and
serving draw on the same memory. Install the build dependencies first:

```sh
sudo apt-get install -y build-essential libssl-dev liburing-dev \
    protobuf-compiler ninja-build libibverbs-dev
```

`libibverbs-dev` is required because Flare's `roce` feature compiles
`helios/native/roce_proxy.c`; without it the Helios build script fails on
`infiniband/verbs.h`.

Install the Rust toolchain that builds Helios, the extension, and the CLI:

```sh
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
```

Python dependencies follow the [environment setup](model-preparation.md#environment-setup).
FlashInfer's JIT builds need `ninja` and a compiler on `PATH`, and the
FlashInfer build cache lives under `~/.cache/flashinfer`.

### Build

```sh
# Python dependencies
uv sync --locked --extra prepare --extra test

# Heliopause extension (Python-side export entry points)
uv pip install --python .venv/bin/python --no-deps ./heliopause

# Flare CLI and Helios runtime
cargo build --release --manifest-path flare/Cargo.toml
```

Rebuild the extension after changing `heliopause/` or the Helios code it uses,
and rebuild the CLI after changing `flare/` or `helios/`. The extension build is
the long one; a release build of the CLI alone finishes in about a minute once
Helios is compiled.

### Get the checkpoint

The recipe pins a repository and revision in
`ignition/preparation/qwen38.py`: `nvidia/Qwen3.8-Flash-Next-FP4-Local` at
`b9ce18a6887e18c089fdea8509a33e63429a3300`.

```sh
hf download nvidia/Qwen3.8-Flash-Next-FP4-Local \
    --revision b9ce18a6887e18c089fdea8509a33e63429a3300
```

The snapshot is about 102 GiB in 21 shards; leave room for the export, which
writes another 99 GiB into the package cache. Other Qwen3.8 repositories on the
Hub are not interchangeable: their configuration or index differs, and
preparation rejects them.

### Export

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m ignition export qwen3.8-flash \
    --gpus 0 --max-concurrency 1 --no-mtp --prefill-chunk-size 1024 --verbose
```

Flags that matter on Thor:

- `--gpus 0` selects the integrated GPU; the package records tensor-parallel
  size 1.
- `--max-concurrency 1` keeps the prepared capacities small.
- `--no-mtp` skips speculative decoding, which is not prepared here.
- `--prefill-chunk-size 1024` is a memory fit, not a component limit. The
  default 4096-token prefill graphs plus the 89 GiB of prepared weights exceed
  Thor's unified memory during compiled-forward warming, and the OOM killer ends
  the run. 1024 fits with room to spare.

With warm kernel and compiled-forward caches the export takes about ten minutes
and ends by printing the cache entry:

```
Exported cache entry: ~/.cache/flare/packages/models/nvidia_2FQwen3.8-Flash-Next-FP4-Local/thor-tp1-202610272207
```

### Serve

```sh
CUDA_VISIBLE_DEVICES=0 ./flare/target/release/flare run \
    ~/.cache/flare/packages/models/nvidia_2FQwen3.8-Flash-Next-FP4-Local/thor-tp1-202610272207
```

The first run stops at the signing-key prompt and prints the fingerprint of the
local development key. Re-run with the two acknowledgements it names, which is
also what unattended runs need:

```sh
CUDA_VISIBLE_DEVICES=0 ./flare/target/release/flare run PACKAGE \
    --dangerously-accept-publisher local-development \
    --dangerously-approve-key 47958bf701392e197aced25eeaaedf038d10b8b4a887a07645092730c4ef3019
```

The fingerprint is generated per machine; read it from the prompt instead of
copying the value above. Progress reports the loaded weights, the cache budget,
and the listening address:

```
[    OK    ] Load weights.  71.3 GiB  10.7s
[    OK    ] Allocate cache.  23.9 GiB · 1945600 KV tokens · 8 shared snapshot slots · limited by memory  25.6s
[    OK    ] Start server.  http://127.0.0.1:8000 · 262144 tokens/request  1.5s
```

Stop Flare with Ctrl-C or SIGTERM so native replay drains before CUDA resources
are released.

### Verify

```sh
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nvidia/Qwen3.8-Flash-Next-FP4-Local",
       "messages":[{"role":"user","content":"What is the capital of France?"}],
       "max_tokens":48}'
```

On this unit that request answered `Paris is the capital of France.` A
5652-token prompt prefilled through several 1024-token chunks in 1.2 s, and a
repeat of an earlier prompt reported `"cached_tokens":57`, confirming prefix
reuse. The model reports reasoning text in `reasoning_content`, so short
`max_tokens` budgets can return `content: null` with `finish_reason: "length"`.

## Troubleshooting

**Ninja compilation is killed during preparation.** FlashInfer's JIT compiles
the CUTLASS fused-MoE module while the prepared weights are resident. Unbounded
parallel `nvcc` jobs exhaust memory and the kernel OOM killer ends the build.
Limit the job count and resume; ninja keeps finished objects:

```sh
MAX_JOBS=2 FLASHINFER_NVCC_THREADS=2 .venv/bin/python -m ignition export ...
```

The first build of that module takes tens of minutes on this 20-core ARM host,
and `FLASHINFER_NVCC_THREADS` parallelizes compilation units within it.

**Preparation is OOM-killed after "Warm compiled forwards".** That is the
4096-token prefill graphs described above; pass `--prefill-chunk-size 1024`.

**`setStorage ... out of bounds` while warming a compiled forward.** Compiled
forwards cached under a different prefill chunk are being reused. Clear the
cache and re-export:

```sh
rm -rf ~/.cache/flare/kernels/forwards
```

**`NVML reports unsupported GPU PCI id`.** The package-naming table in
`helios/src/package/naming.rs` does not know this GPU. Thor reports device
`0x2b00` and maps to the `thor` platform identifier; rebuild the extension after
changing the table.

**Checkpoint index rejected.** Preparation validates the snapshot against the
recipe's expected manifest. A different Qwen3.8 repository fails with missing
quantized-layer declarations or an unexpected `ple_embedding_dtype`; use the
pinned revision.

**`/tmp` does not survive reboots.** Keep probe scripts elsewhere.

## Paths to recipe support

Two routes exist; both keep the architecture boundaries in
[the core design](core-design.md).

1. **Ignition SM110 components over FlashInfer mainline.** Wrap `mm_fp4` and
   `MoELayer` with `CutlassNvfp4Config` as SM110 component variants, and
   `sm110_xqa` prepared plans where a recipe's attention contract matches.
   [The component plan](thor-components.md) stages this work and its validation.
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

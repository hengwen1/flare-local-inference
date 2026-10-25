# Ignition SM110 components for Thor recipe support

This is the plan for [route 1](thor.md#paths-to-recipe-support): preparing and
serving supported recipes on Jetson Thor (SM110) through Ignition component
variants over FlashInfer mainline, with no fork kernel development. Helios and
heliopause are untouched — packages already record the prepared capability as
`[11, 0]` data that Helios validates at load. All work is Python-side:
components, checkpoint loading, and preparation, following the
[SM103 component precedent](../ignition/components/moe_sm103.py) — FlashInfer
SM100-family kernels where b12x cannot execute, selected by device capability
in component constructors, never by environment variables or recipe identity.

## Measured ground truth

Claims below trace to runs on one Thor unit with the pinned
`hengwen1/flashinfer` `jthor` branch; see the [kernel matrix](thor.md#kernel-support-on-sm110).

- Works: `fp4_quantization_110` JIT, NVFP4 dense GEMM (`mm_fp4`, cutlass and
  cudnn, cosine 0.991 vs bf16 at 64×256×128), NVFP4 fused MoE
  (`CutlassNvfp4Config` end-to-end probe: 8 experts, top-2, SwiGLU, 1.0000
  match at atol 0.05), b12x paged decode attention (55/55), b12x plain FP8
  GEMM, `sm110_xqa` attention (306 cases).
- Blocked: b12x block-scaled FP4/FP6/MXFP8 GEMM and NVFP4 MoE (the SM120-family
  block-scale instruction does not assemble for sm_110a); the `trtllm` mm_fp4
  backend is not claimed on capability 110.

## Components

### Capability detection

`ignition/components/sm110.py` with `is_sm110(device)`, mirroring
`ignition/components/sm103.py`. Component constructors branch on it exactly as
the SM103 paths do today; shared policy never sees the capability.

### Dense NVFP4 projections

An SM110 variant of `QuantizedProjection` through `mm_fp4` with the `cutlass`
or `cudnn` backend. Caller-owned `out` keeps capture allocation-free; workspace
behavior per backend must be probed before wiring.

The adapter reuses checkpoint payloads. Checkpoints store packed E2M1 values
with CSF-compressed scales and calibrated global scales; `mm_fp4` consumes
packed FP4 operands, linear E4M3 block scales (`use_8x4_sf_layout` selects the
scale-factor layout), and a global alpha. CSF scales decode losslessly once at
load, as `moe_sm103` does for trtllm-gen. Weights are never requantized;
checkpoint global scales stay authoritative.

### Routed experts

`Sm110Experts` / `Sm110RoutedExperts` in `ignition/components/moe_sm110.py`,
dispatched from `PreparedRoutedExperts.__new__` beside the SM103 dispatch.
Execution goes through FlashInfer `MoELayer` with `CutlassNvfp4Config` — the
only fused-MoE path verified on capability 110. Routing stays b12x
`route_topk`. `CutlassNvfp4Config` requires packed precomputed routing with
`do_finalize=True`, hidden size divisible by 64, intermediate size divisible by
16, and does not support expert parallelism or shared experts; geometry
validation rejects unsupported checkpoints with explicit messages, and shared
experts stay on the dense component path.

The weight adapter is the main cost. `CutlassNvfp4Config.prepare_weights`
quantizes from BF16, which checkpoints do not carry — expert weights are
already FP4. The adapter must re-lay out existing E2M1 payloads into the
CUTLASS-swizzled weight layout and feed decoded linear scales plus the
checkpoint's global scales, then verify the result against the
`prepare_weights` reference before any recipe wiring.

### Attention and router

b12x paged decode attention works on SM110 and needs no variant. Per recipe,
the context/prefill path and any sparse-selection machinery (Qwen's QSA, GLM's
DSA indexer) must be probed on SM110 before wiring; `sm110_xqa` prepared plans
are the replacement where a recipe's attention contract matches its semantics.
The Triton kernels in the SM103 sparse-attention component compile for
sm_110a but are unmeasured. The b12x router (`route_topk`) is portable and
unchanged.

## Recipe scope and phasing

1. **Qwen3.8 first.** Its experts are NVFP4 with CSF scales and its dense
   weights are NVFP4 — exactly the verified kernel mapping. Component probes at
   toy shapes are done for MoE; the checkpoint-format adapter, dense path, and
   attention probes come next, then integration, per-capacity capture, export,
   and a Helios serving pass.
2. **Other recipes follow only where kernel coverage exists.** MXFP4 and FP6
   expert storage has no verified SM110 path (b12x blocked; the CUTLASS runner
   is NVFP4-only), so GLM-, DeepSeek-, and Kimi-class recipes keep failing
   preparation with explicit rejections until route 2 or a verified mainline
   path exists. MiniMax M3's explicit SM110 rejection stays until its paths are
   covered. Recipes with plain FP8 dense weights can keep b12x there and need
   only the expert and attention paths.

Per [repository guidance](../AGENTS.md#validation-and-performance), each
recipe's numerical bounds are measured on Thor through the new component path —
the fork's accuracy gates and SM120 b12x bounds do not substitute. Cross-device
comparison against the same checkpoint on SM120 is the reference method for
adapter correctness. Qualification covers every prepared capacity, graph
transitions, state lifetimes, and preparation and serving memory under unified
memory, and ends with Helios service validation, not component probes alone.

## Validation per phase

- **Adapter probes.** Decode CSF scales and execute both paths at toy shapes;
  then run checkpoint-format weights through the adapters and compare against
  the BF16-quantized `prepare_weights` reference and against b12x on SM120
  within FP4-specific bounds. Measure repeatability.
- **Preparation on Thor.** The complete serving shape catalog through one
  preparation batch with tuning declarations; memory budgeted against
  `MemAvailable` under unified memory; export succeeds.
- **Serving.** `flare run PACKAGE` on Thor: every prepared capacity, decode and
  prefill, graph transitions, prefix reuse, and measured native replay. Report
  untested models and paths separately from implemented restrictions.

## Open questions to resolve by probe before wiring

1. Whether the CUTLASS runner accepts externally quantized weights (checkpoint
   payloads plus decoded scales) or only its own `prepare_weights` output;
   this determines the adapter's depth.
2. The `mm_fp4` weight scale-factor layout (8x4 vs 128x4) matching decoded
   linear scales, and where checkpoint global scales fold into alpha.
3. Whether the packed precomputed routing accepts b12x `route_topk` outputs
   directly or needs its own packing step.
4. Allocation-free graph replay for both paths at every prepared capacity.
   Graph-pool intermediates follow the `moe_sm103` precedent; steady-state
   allocation stays prohibited.
5. Expert TP sharding: the runner declares no expert-parallel support, so
   tensor-parallel intermediate slicing must be verified against the swizzled
   weight layout and its scales.
6. The context-attention path per recipe on SM110, before any recipe wiring.

If the SM110 path needs captured-graph native operations beyond the contracts
the SM103 components introduced, that requires prior approval from project
owner Luke Alonso <lalonso@gmail.com> per
[the design review](design-review.md); prefer reusing the existing contracts.

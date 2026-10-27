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

### Dense projections

Qwen3.8's dense weights are MXFP8 (E4M3 payload + 32-wide UE8M0 scales), not
NVFP4, so the dense SM110 variant runs through `mm_mxfp8` (cutlass backend; the
JIT module builds on sm_110a). `Sm110Mxfp8Projection` in
`ignition/components/linear_sm110.py`, dispatched from `QuantizedProjection`
beside the SM103 branch, passes checkpoint E4M3 payloads through and takes
FlashInfer's own `block_scale_interleave` for the 128x4 block-scale grid —
verified byte-identical to the probe's swizzle — so no weight is requantized.
The module's sm100 quantizer writes quantized activations into caller-owned
scratch, and `mm_mxfp8` writes into the caller's output, so a launch neither
allocates nor copies in steady state. Probed with checkpoint tensors (layer-0
`in_proj_qkv`, 16 tokens): cosine 0.9999986, relative L2 0.0017 against exact
dequantization.

The dense component rejects what the kernel cannot honor rather than
approximating it: A16 activations on MXFP8 weights, and the two-stream parallel
projection path.

Sixteen-bit activations with NVFP4 weights (`W4A16_NVFP4`) run through
`Sm110Nvfp4A16Projection`, dispatched from `QuantizedProjection` when a call
site asks for `a16`. FlashInfer's remaining W4A16 backends reject capability
110, so the capability's kernel is cuDNN's `mm_bf16_fp4`, prepared once at load
through `prepare_bf16_fp4_weights`. It needs no scratch because nothing is
quantized, and the folded bias is added after the launch. Checkpoints pad the
stored input width to 128 and zero the extra columns, so the component trims
the payload and unswizzles the block scales back to the logical width instead of
running a wider GEMM; the trimmed matrix is exact. Qwen3.8's vision MLP and
merger need this path.

Unquantized BF16 projections run through `Sm110Bf16Projection` in the same
module, dispatched from `UnquantizedProjection`. b12x's BF16 GEMV requires
SM120/SM121 and FlashInfer's cutlass BF16 backend rejects capability 110, so the
capability's dense BF16 kernel is cuDNN's, which folds the bias in its epilogue
and writes the caller's output. It needs no scratch: nothing is quantized.
Recipes that pass FP32 unquantized operands stay rejected explicitly.

Stream overlap is rejected on SM110. Both overlap paths fork prepared
projections onto an auxiliary stream: the quantized branches run b12x's parallel
block-scaled GEMMs, which report no support for the capability, and the BF16
branch runs b12x's small-N GEMV, which is unverified there. The shared-expert
and Qwen3.8 attention overlap constructors therefore raise with the reason and
the remedy instead of capturing a fork that cannot be prepared. Recipes disable
stream overlap on SM110; single-stream composition is the supported path.

Activation quantization on both dense and expert paths uses the runner's
canonical dynamic grid rather than the checkpoint's static calibration scales;
the checkpoint's `input_scale` values do not fold into the runner's scalar
activation global scale (probing that variant fails). Replacing static
checkpoint activation scales with dynamic quantization is a semantic decision
for the component design and its measured bounds, not a silent substitution.

### Routed experts

`CutlassNvfp4View` / `Sm110RoutedExperts` in `ignition/components/moe_sm110.py`,
dispatched from `PreparedRoutedExperts.__new__` beside the SM103 dispatch.
Execution goes through FlashInfer `MoELayer` with `CutlassNvfp4Config` — the
only fused-MoE path verified on capability 110. Routing stays b12x
`route_topk`. `CutlassNvfp4Config` requires packed precomputed routing with
`do_finalize=True`, hidden size divisible by 64, intermediate size divisible by
16, and does not support expert parallelism or shared experts; geometry
validation rejects unsupported checkpoints with explicit messages, and shared
experts stay on the dense component path.

The weight adapter is the main cost, and probing has resolved its shape:
`prepare_weights` quantizes from BF16, which checkpoints do not carry, so the
adapter re-lays out existing E2M1 payloads into the CUTLASS runner's view
without requantizing: payloads pass through, CSF scales decode to linear E4M3
rows through the same expansion machinery the SM103 loader uses, a 128x4
swizzle (byte-exact against `fp4_quantize`'s layout, verified) produces the
weight block scales, and the checkpoint's per-expert global multipliers become
`fc1_dequant_scale`/`fc2_dequant_scale`. The probed checkpoint pack reaches
cosine 0.9964 against exact dequantization at layer-0 geometry — closer than
the double-quantized `prepare_weights` round trip (0.9827) — and its prepared
plan replays allocation-stable under CUDA graph capture. That probe compares
eager launches only: its replay comparison re-drew routing ids through
`ids.to(torch.int32)`, which copies, so the kernel replayed the original routes
while the reference combined the new ones. Replay numerics come from the
component test below, not from that probe. The w13 half order is `[up, gate]`
with the activation applied to the second half, matching Qwen semantics.

Weights must keep checkpoint magnitudes. Fixtures an order of magnitude away
from the release measured 0.53 relative L2 through every global-scale
convention tried, and only checkpoint-magnitude weights reproduced the probe's
0.086; the mechanism was not isolated. The component test therefore quantizes
at the release's weight scale and derives the per-expert multipliers the way
the release does (`amax / 448`).

### Linear attention

Qwen3.8's gated-delta layers need no SM110 component. b12x's `gdn_decode`,
`gdn_prefill`, and `short_conv` kernels compile for sm_110a and run correctly,
measured on Thor at Qwen3.8 layer-0 geometry (48 value heads, 16 key heads,
128-wide heads, 10240 convolution channels, width 4) against b12x's own
references: `gdn_decode` matched output exactly and recurrent state to 3.5e-8,
and `gdn_prefill` reached 4.3e-3 output and 2.9e-3 state relative L2. The
component therefore takes its existing non-SM103 branch, which is also the
SM120 serving path; the SM103 chunked prefill stays SM103-only.
`short_conv` ran, but the standalone reference used for that probe did not
reproduce the mode's activation and rounding semantics (0.577 relative L2 raw,
0.154 after SiLU), so its numerics remain established only through the recipe
component and its tests, not by that probe.

### Attention and router

b12x paged decode attention works on SM110 and needs no variant, and so does
b12x's QSA. Prepared on Thor and run eagerly, repeated, and through a captured
graph at tp1 geometry over 8 packed decode rows at context 2048, QSA produced
finite output, identical results across repeated eager runs, and replay output
equal to eager with no steady-state allocation. That probe checked finiteness,
repeatability, and replay identity rather than b12x's references, because the
fork's benchmark harness still reads plan attributes b12x has since removed
(`plan.caps`, `plan.workspace_q_rows`); the recipe tests remain the numeric
authority. Prefill row counts and the sparse-selection variants of other
recipes (GLM's DSA indexer) stay unprobed, and `sm110_xqa` prepared plans are
the replacement where a recipe's attention contract matches its semantics. The
Triton kernels in the SM103 sparse-attention component compile for sm_110a but
are unmeasured. The b12x router (`route_topk`) needs no SM110 variant: measured
on Thor over 64 rows of 128 experts with top 8, sigmoid scores, and
renormalized weights, it returned exactly `torch.topk`'s ids and the reference
renormalized weights. The CUTLASS runner's packed precomputed routing takes
exactly the `topk_ids` int32 and `topk_weights` float32 tensors that routing
emits, and captured graphs replay those buffers, so the component must own
stable routing storage across replay.

## Recipe scope and phasing

1. **Qwen3.8 first.** Its experts are NVFP4 with CSF scales and its dense
   weights are MXFP8 — exactly the verified kernel mapping. The routed-expert
   and dense components are prepared and measured at Qwen3.8 geometry;
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
- **Routed experts.** `tests/components/test_moe_sm110_cuda.py` runs the
  prepared component at Qwen3.8 expert geometry through eager and compiled
  forwards, captures it, and replays allocation-stable over tokens 1, 4, 24,
  and 512. Measured on Thor: relative L2 0.0894-0.0906 against dequantized
  activations and 0.1631-0.1640 against the BF16 source, bounded at 0.10 and
  0.18.
- **Dense projections.** `tests/components/test_linear_sm110_cuda.py` runs the
  prepared projection at Qwen3.8 `in_proj_qkv` geometry the same way over
  tokens 1, 16, 128, and 512. Measured on Thor: relative L2 0.001652-0.001667
  against dequantized activations and 0.026510-0.026743 against the BF16
  source, bounded at 0.003 and 0.035.
- **16-bit-activation dense.** `tests/components/test_linear_sm110_nvfp4_a16_cuda.py`
  covers Qwen3.8 vision MLP and merger geometry at tokens 4, 16, 128, and 512 in
  eager and compiled forwards with folded bias, plus the padded-width trim and
  the MXFP8 rejection. Measured on Thor: relative L2 0.002303-0.002367 against
  the exact decode of the same tensors, bounded at 0.003.
- **Unquantized projections.** `tests/components/test_linear_sm110_bf16_cuda.py`
  covers the same shapes for the BF16 kernel over the same token counts.
  Measured on Thor: relative L2 0.001642-0.001695 against the FP32 product with
  its folded bias, bounded at 0.003.
- **Linear attention.** The GDN decode, GDN prefill, and short-convolution
  kernels run on SM110; the recipe's gated-delta layers keep their existing
  non-SM103 branch and are validated through the Qwen3.8 GDN tests, which need
  the validated checkpoint release.
- **Attention.** QSA prepares and runs on SM110: finite, repeatable eager
  output, and replay equal to eager with no steady-state allocation, at tp1
  geometry over 8 packed decode rows at context 2048.
- **Stream overlap.** `tests/components/test_overlap_sm110_cuda.py` asserts the
  capability reports no parallel block-scaled GEMM support and that both overlap
  constructors reject it.
- **Router.** `route_topk` matched `torch.topk` ids and reference renormalized
  weights at 64 rows of 128 experts with top 8.
- **Preparation on Thor.** The complete serving shape catalog through one
  preparation batch with tuning declarations; memory budgeted against
  `MemAvailable` under unified memory; export succeeds.
- **Serving.** `flare run PACKAGE` on Thor: every prepared capacity, decode and
  prefill, graph transitions, prefix reuse, and measured native replay. Report
  untested models and paths separately from implemented restrictions.

## Open questions remaining after probing

Probes on this unit (scripts and raw logs kept outside the repository under
`~/work/thor-probes/`) resolved the adapter questions above. Still open:

1. Whether TP sharding of the swizzled expert layout preserves the scale
   grid's expert and tile boundaries (the runner declares no expert-parallel
   support; slicing plan needed before TP qualification).
2. The context-attention path per recipe on SM110 (QSA/indexer probes), before
   any recipe wiring.
3. Numerical bounds for the dynamic-activation-quantization decision above,
   measured on the complete prepared path. Component-level bounds are recorded
   under [validation per phase](#validation-per-phase); recipe bounds still
   require the validated checkpoint release.
4. Production loading must run against the validated checkpoint release; the
   cached HF snapshots on this unit fail the manifest validator and probes
   read tensors directly.

If the SM110 path needs captured-graph native operations beyond the contracts
the SM103 components introduced, that requires prior approval from project
owner Luke Alonso <lalonso@gmail.com> per
[the design review](design-review.md); prefer reusing the existing contracts.

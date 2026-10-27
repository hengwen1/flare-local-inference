"""Routed experts on SM110 (Jetson Thor) through FlashInfer's CUTLASS NVFP4 MoE.

b12x executes routed experts on SM120/SM121 only, and its SM120-family
block-scale instruction does not assemble for sm_110a. On compute capability
11.0 the same checkpoint storage runs through FlashInfer's CUTLASS NVFP4
runner: packed E2M1 payloads pass through unchanged, CSF-expanded linear E4M3
scales take the 128x4 interleave (verified byte-exact against FlashInfer's
own swizzle), and the checkpoint's per-expert weight global scales become the
kernel's dequant scalars. FC1 rows stay in checkpoint ``[up; gate]`` order;
the kernel applies its activation to the second half, matching Qwen
semantics.

Routing remains b12x's ``route_topk``. Activation quantization into
caller-owned scratch uses FlashInfer's sm_110a ``fp4_quantize`` JIT module,
producing the canonical unit-global pack the runner consumes; the checkpoint's
static activation scales are validated and retained on the prepared storage
but are not folded into this kernel's ABI, which has no per-token activation
scale input. The runner's internal tuner selects tactics during warmup and is
frozen by capture; native intermediates allocate from the capture's graph
pool, and the output is the caller's tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache, partial

import torch
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.preparation.scratch import ScratchArena, planned_scratch, prime_owners
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import benchmark_resource, declare, register_finalizer

from .native_operator import native_contract, native_plan


@cache
def _quantizer():
    from flashinfer.jit.fp4_quantization import gen_fp4_quantization_sm110_module

    return gen_fp4_quantization_sm110_module().build_and_load()


def _vector(values, count, device, name):
    values = torch.as_tensor(values).reshape(-1).to("cpu", torch.float64)
    if values.shape != (count,) or not bool((torch.isfinite(values) & (values > 0)).all()):
        raise ValueError(f"{name} must be {count} positive finite values")
    return values


class CutlassNvfp4Experts:
    """Routed experts prepared for FlashInfer's CUTLASS NVFP4 runner."""

    kind = "cutlass_nvfp4"

    @property
    def num_experts(self) -> int:
        return self.w13.shape[0]

    @property
    def hidden_size(self) -> int:
        return self.w2.shape[1]

    @property
    def intermediate_size(self) -> int:
        return self.w2.shape[2] * 2

    @property
    def device(self) -> torch.device:
        return self.w13.device


@dataclass(frozen=True, eq=False)
class CutlassNvfp4View(CutlassNvfp4Experts):
    """Checkpoint NVFP4 expert storage in the CUTLASS runner's packed layout.

    ``w13`` keeps the checkpoint's packed ``[up; gate]`` rows and ``w2`` the
    packed ``[hidden, intermediate]`` plane; nothing is requantized. The
    ``*_scale`` fields are the 128x4-interleaved views of the expanded linear
    E4M3 scales. The ``*_global`` fields are the kernel's per-expert FC1 and
    FC2 weight dequant scalars. ``input_scale`` and ``intermediate_scale``
    are the checkpoint's reciprocal activation scales: validated and retained
    so the static-versus-dynamic activation decision stays explicit, and not
    consumed by the kernel ABI.
    """

    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    w13_global: torch.Tensor
    w2_global: torch.Tensor
    input_scale: torch.Tensor
    intermediate_scale: torch.Tensor


def _check_packed(w13, s13, w2, s2):
    if w13.dtype != torch.uint8 or w2.dtype != torch.uint8 or w13.ndim != 3 or w2.ndim != 3:
        raise ValueError("packed NVFP4 experts must be rank-three uint8 tensors")
    experts, rows, half = w13.shape
    intermediate, hidden = rows // 2, half * 2
    if rows % 2 or w2.shape != (experts, hidden, intermediate // 2):
        raise ValueError("FC1 must hold [up; gate] rows matching FC2 geometry")
    if hidden % 64 or intermediate % 16:
        raise ValueError("CUTLASS NVFP4 experts require hidden size divisible by 64 and intermediate size divisible by 16")
    for scales, shape in ((s13, (experts, rows, hidden // 16)), (s2, (experts, hidden, intermediate // 16))):
        if scales.dtype not in (torch.uint8, torch.float8_e4m3fn) or tuple(scales.shape) != shape:
            raise ValueError("NVFP4 block scales must be row-major E4M3 with one scale per 16 values")
    tensors = (w13, s13, w2, s2)
    if any(t.device != w13.device or not t.is_contiguous() for t in tensors):
        raise ValueError("expert tensors must be contiguous on one CUDA device")
    return experts, hidden, intermediate


def _interleaved_scales(scales):
    """128x4-interleaved E4M3 weight scales in the CUTLASS runner's padded layout."""
    flat = scales.view(torch.uint8)
    experts, rows, columns = flat.shape
    padded_rows = (rows + 127) // 128 * 128
    padded_columns = (columns + 3) // 4 * 4
    padded = flat.new_zeros((experts, padded_rows, padded_columns))
    padded[:, :rows, :columns] = flat
    padded = padded.view(experts, padded_rows // 128, 4, 32, padded_columns // 4, 4)
    # The runner's scale shape is the padded grid; the interleave order matches
    # fp4_quantize's swizzled layout (verified byte-exact by probe).
    return padded.permute(0, 1, 4, 3, 2, 5).contiguous().view(experts, padded_rows, padded_columns)


def prepare_cutlass_nvfp4_experts(w13, s13, w2, s2, scale_vectors) -> CutlassNvfp4View:
    """Re-lay out checkpoint NVFP4 experts for the CUTLASS runner; payloads pass through.

    ``s13``/``s2`` are expanded row-major E4M3 scales. ``scale_vectors`` holds
    the FC1 and FC2 weight global scales followed by the FC1 and FC2 activation
    global scales (reciprocal checkpoint input scales), as the loader's
    ``_expert_scale_vectors`` produces.
    """
    experts, hidden, intermediate = _check_packed(w13, s13, w2, s2)
    if len(scale_vectors) != 4:
        raise ValueError("NVFP4 A4 experts require weight and activation global scales")
    w13_global, w2_global, input_scale, intermediate_scale = (
        _vector(v, experts, w13.device, name) for v, name in zip(scale_vectors, (
            "FC1 weight global scales", "FC2 weight global scales",
            "FC1 activation global scales", "FC2 activation global scales"), strict=True))
    device = w13.device
    as_device = lambda values: torch.as_tensor(values, dtype=torch.float32, device=device)
    return CutlassNvfp4View(
        w13=w13, w13_scale=_interleaved_scales(s13), w2=w2, w2_scale=_interleaved_scales(s2),
        w13_global=as_device(w13_global), w2_global=as_device(w2_global),
        input_scale=as_device(input_scale), intermediate_scale=as_device(intermediate_scale))


@dataclass(frozen=True)
class RoutedExpertsQuery:
    """Exact routed-expert invocation; tactics depend only on these values."""

    num_tokens: int
    num_experts: int
    hidden_size: int
    intermediate_size: int
    top_k: int


def _tactics(query: RoutedExpertsQuery):
    # One Ignition tactic: the flat CUTLASS kernel's internal tuner owns tactic
    # selection, runs during warmup, and is frozen by capture.
    return (0,)


CUTLASS_CONTRACT = native_contract(component_id="flare.moe.sm110_cutlass_nvfp4",
                                   query_type=RoutedExpertsQuery, tactics=_tactics,
                                   semantic_version=1)


def _aligned(nbytes):
    return (nbytes + 255) // 256 * 256


def _scratch_layout(query):
    """Byte extents of the packed activations and of their 16-row-padded scales."""
    tokens, hidden = query.num_tokens, query.hidden_size
    payload = _aligned(tokens * hidden // 2)
    return payload, _aligned((tokens + 15) // 16 * 16 * hidden // 16)


def _scratch_bytes(query):
    """NVFP4 activations and their row-major scales, padded to 16 rows as FlashInfer writes them."""
    return sum(_scratch_layout(query))


class _CutlassState:
    """The prepared CUTLASS MoE layer over one exact shape, bound to fixed storage.

    ``run`` quantizes the source into caller-owned scratch and launches the
    layer's flat kernel through ``MoELayer``, which owns the runner, its
    workspace, and the tactic selection that warmup seeds and capture freezes.
    The layer allocates its output from the capture's graph pool and offers no
    caller-supplied buffer, so ``run`` copies it into the caller's output, which
    survives graph boundaries.
    """

    implementation = "cutlass_nvfp4"

    def __init__(self, experts: CutlassNvfp4View, query: RoutedExpertsQuery, tactic):
        from flashinfer.fused_moe import (BackendOptions, CutlassNvfp4Config, ExpertConfig,
                                          MoEActivationPack, MoEConfig, MoELayer, MoEWeightPack,
                                          QuantConfig, QuantFormat, RoutingConfig, RoutingMethodType,
                                          SwiGLU)

        self.experts, self.query = experts, query
        config = MoEConfig(
            routing=RoutingConfig(num_experts=query.num_experts, top_k=query.top_k,
                                  method=RoutingMethodType.Renormalize),
            quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4),
            experts=ExpertConfig(intermediate_size=query.intermediate_size),
            activation=SwiGLU(),
            backend=BackendOptions(candidates=(CutlassNvfp4Config(),)),
        )
        ceiling = config.execution.tune_max_num_tokens
        if query.num_tokens > ceiling:
            raise ValueError(f"SM110 routed experts plan {query.num_tokens} tokens beyond the "
                             f"runner's {ceiling}-token tuning ceiling")
        self.quantize = _quantizer().fp4_quantize
        self.unit_scale = torch.ones((1,), dtype=torch.float32, device=experts.device)
        self.layer = MoELayer(config, device=experts.device)
        tokens, hidden = query.num_tokens, query.hidden_size
        self.weight_pack = MoEWeightPack()
        self.weight_pack.prepare_for("cutlass_nvfp4", {
            "fc1_expert_weights": experts.w13, "fc2_expert_weights": experts.w2,
            "fc1_act_global_scale": torch.ones((), dtype=torch.float32, device=experts.device),
            "fc1_weight_block_scale": experts.w13_scale, "fc1_dequant_scale": experts.w13_global,
            "fc2_act_global_scale": torch.ones((), dtype=torch.float32, device=experts.device),
            "fc2_weight_block_scale": experts.w2_scale, "fc2_dequant_scale": experts.w2_global,
        })
        self.act_pack = MoEActivationPack(
            hidden_states_q=torch.empty((tokens, hidden // 2), dtype=torch.uint8, device=experts.device),
            hidden_states_scale=torch.empty((tokens, hidden // 16), dtype=torch.uint8, device=experts.device),
            topk_ids=torch.zeros((tokens, query.top_k), dtype=torch.int32, device=experts.device),
            topk_weights=torch.full((tokens, query.top_k), 1.0 / query.top_k, dtype=torch.float32,
                                    device=experts.device))

    def scratch_bytes(self):
        return _scratch_bytes(self.query)

    def bind(self, a, topk_ids, topk_weights, output, scratch):
        """Point the activation pack at the caller's exact storage; shapes were planned exactly."""
        tokens, hidden = a.shape
        payload, _ = _scratch_layout(self.query)
        self.act_pack.hidden_states_q = scratch[:payload].view(tokens, hidden // 2)
        scales = scratch[payload:_scratch_bytes(self.query)]
        self.act_pack.hidden_states_scale = scales[:tokens * hidden // 16].view(tokens, hidden // 16)
        self.act_pack.topk_ids = topk_ids
        self.act_pack.topk_weights = topk_weights
        self.payload, self.scales = self.act_pack.hidden_states_q, self.act_pack.hidden_states_scale

    def run(self, a, topk_ids, topk_weights, output, scratch):
        self.quantize(a, self.unit_scale, self.payload, self.scales, 16, False, False, False, False, True, 0)
        produced = self.layer(self.act_pack, self.weight_pack)
        output.copy_(produced)
        return output


def plan_routed_experts(experts: CutlassNvfp4View, *, tokens: int, top_k: int):
    """Declare one exact-shape native routed-expert operator for ``experts``."""
    query = RoutedExpertsQuery(int(tokens), experts.num_experts, experts.hidden_size,
                               experts.intermediate_size, int(top_k))
    return native_plan(contract=CUTLASS_CONTRACT, query=query, device=experts.device,
                       scratch_bytes=lambda tactic: _scratch_bytes(query),
                       materialize=lambda tactic: _CutlassState(experts, query, tactic))


@torch.library.custom_op("flare::sm110_routed_experts", mutates_args=("output", "scratch"))
def _sm110_routed_experts(a: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor,
                          output: torch.Tensor, scratch: torch.Tensor, plan_handle: int) -> None:
    plan = plan_from_handle(plan_handle)
    require_prepared(plan, plan.component_id, a.device).run(a, topk_ids, topk_weights, output, scratch)


@_sm110_routed_experts.register_fake
def _fake_sm110_routed_experts(a, topk_ids, topk_weights, output, scratch, plan_handle) -> None:
    pass


@dataclass(frozen=True, eq=False)
class Sm110Binding:
    """Exact tensors of one prepared native routed-expert launch.

    Eager calls launch the bound state directly. Traced calls record the
    compiler-visible custom op, which resolves the prepared plan by handle.
    """

    plan: object
    state: object
    a: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    output: torch.Tensor
    scratch: torch.Tensor
    implementation: str

    def run(self):
        if _tracing():
            torch.ops.flare.sm110_routed_experts(self.a, self.topk_ids, self.topk_weights,
                                                 self.output, self.scratch, self.plan._handle)
        else:
            self.state.run(self.a, self.topk_ids, self.topk_weights, self.output, self.scratch)
        return self.output


def _tracing():
    from torch.fx.experimental.proxy_tensor import get_proxy_mode

    return torch.compiler.is_compiling() or get_proxy_mode() is not None


def _bind(plan, state, a, topk_ids, topk_weights, output, scratch):
    tokens, hidden = a.shape
    query = plan.query
    if (a.dtype != torch.bfloat16 or (tokens, hidden) != (query.num_tokens, query.hidden_size)
            or not a.is_contiguous()):
        raise ValueError("SM110 routed experts require the prepared contiguous BF16 source")
    for tensor, dtype in ((topk_ids, torch.int32), (topk_weights, torch.float32)):
        if tensor.dtype != dtype or tensor.shape != (tokens, query.top_k) or not tensor.is_contiguous():
            raise ValueError("SM110 routes require contiguous int32 IDs and FP32 weights per token")
    if output.dtype != torch.bfloat16 or output.shape != a.shape or not output.is_contiguous():
        raise ValueError("SM110 routed output must be contiguous BF16 matching the source")
    if scratch.dtype != torch.uint8 or scratch.ndim != 1 or scratch.numel() < state.scratch_bytes():
        raise ValueError("SM110 routed-expert scratch is smaller than its plan")
    if any(t.device != a.device for t in (topk_ids, topk_weights, output, scratch)):
        raise ValueError("SM110 routed-expert tensors must share one device")
    state.bind(a, topk_ids, topk_weights, output, scratch)
    return Sm110Binding(plan, state, a, topk_ids, topk_weights, output, scratch, state.implementation)


def _benchmark_call(state, *, plan, source, experts, top_k, session, name, workspace,
                    allocate_router, router_plan):
    from ignition.preparation.fixtures.fused_moe import ExpertBenchmarkInputs

    operands = benchmark_resource(session, ("moe_inputs", name),
        lambda: ExpertBenchmarkInputs(source, experts=experts.num_experts, top_k=top_k,
            allocate_router=allocate_router, router_plan=router_plan))
    scratch = planned_scratch(plan.memory_requirements().scratch, workspace)
    binding = _bind(plan, state, operands.source, operands.router.ids, operands.router.weights,
                    operands.output, scratch)
    return PreparedCall(run=binding.run, output=operands.output, produce=operands.producers[0],
                        benchmark_producers=operands.producers,
                        owners=(operands, scratch, binding, experts))


class Sm110RoutedExperts:
    """Prepared native expert execution with fixed source and route storage.

    Mirrors ``Sm103RoutedExperts``: one exact-shape plan declared to the
    preparation session, bound to caller-owned scratch after tuning.
    """

    def __init__(self, source, experts: CutlassNvfp4View, ids, weights, output, *, session, name: str,
                 workspace: ScratchArena | torch.Tensor | None = None,
                 router_plan=None, a4_prefill=False, router=None, route_output=None):
        from ignition.components.moe import RouterScratch

        del router_plan
        if a4_prefill or route_output is not None or router is not None:
            raise ValueError("SM110 routed experts provide fused finalize only; shared-expert overlap and A4 prefill are unsupported")
        if source.ndim != 2 or source.shape[1] != experts.hidden_size or ids.ndim != 2:
            raise ValueError("SM110 routed experts require [tokens, hidden] source and [tokens, top_k] routes")
        for field in ("w13", "w13_scale", "w2", "w2_scale", "w13_global", "w2_global",
                      "input_scale", "intermediate_scale"):
            tensor = getattr(experts, field, None)
            if tensor is not None:
                record_numeric_tensor(tensor)
        self.source, self.output, self.scratch = source, output, None
        self.plan = plan_routed_experts(experts, tokens=source.shape[0], top_k=ids.shape[1])
        plan = self.plan

        def prepare_call(state):
            scratch = planned_scratch(plan.memory_requirements().scratch, workspace)
            self.scratch = scratch
            binding = _bind(plan, state, source, ids, weights, output, scratch)
            return PreparedCall(run=binding.run, owners=prime_owners(workspace, scratch, binding))

        declare(session, (plan.request(name=name, prepare_call=prepare_call,
            benchmark_call=partial(_benchmark_call, plan=plan, source=source, experts=experts,
                top_k=ids.shape[1], session=session, name=name, workspace=workspace,
                allocate_router=RouterScratch.allocate)),))

        def finalize():
            self.scratch = planned_scratch(plan.scratch_specs(), workspace)
            state = require_prepared(plan, plan.component_id, source.device)
            self.binding = _bind(plan, state, source, ids, weights, output, self.scratch)

        register_finalizer(session, finalize, workspace=workspace)

    def __call__(self) -> torch.Tensor:
        return self.binding.run()

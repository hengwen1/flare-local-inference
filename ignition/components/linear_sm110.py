"""Dense MXFP8 projections on SM110 (Jetson Thor) through FlashInfer's cutlass MXFP8 GEMM.

b12x executes block-scaled dense GEMMs on SM120/SM121 only, so dense projections
on compute capability 11.0 run through FlashInfer's ``mm_mxfp8`` cutlass kernel,
the dense path verified on that capability. Checkpoint E4M3 payloads pass
through unchanged and the checkpoint's linear UE8M0 rows take FlashInfer's own
128x4 block-scale interleave, so no weight is requantized.

Activations are quantized into caller-owned scratch by the module's sm100
quantizer, which builds for sm_110a; the checkpoint's static activation scales
do not exist for MXFP8 dense weights, whose storage carries no activation
global scale. ``mm_mxfp8`` writes into the caller's output, so a launch neither
allocates nor copies in steady state.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache, partial

import torch
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.preparation.fixtures import projection as projection_fixtures
from ignition.preparation.scratch import ScratchArena, planned_scratch, prime_owners
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import benchmark_resource, declare, register_finalizer

from .native_operator import native_contract, native_plan


@cache
def _quantizer():
    from flashinfer.jit.fp8_quantization import gen_mxfp8_quantization_sm100_module

    return gen_mxfp8_quantization_sm100_module().build_and_load()


@cache
def _interleave():
    from flashinfer.fp4_quantization import block_scale_interleave

    return block_scale_interleave


def _aligned(nbytes):
    return (nbytes + 255) // 256 * 256


def serves(device) -> bool:
    """Whether dense projections on ``device`` run through this module's kernels."""
    from .sm110 import is_sm110

    return is_sm110(device)


@dataclass(frozen=True, eq=False)
class Mxfp8DenseWeight:
    """Checkpoint MXFP8 dense storage in the cutlass MXFP8 GEMM's layout.

    ``values`` is the checkpoint's E4M3 payload in ``[out_features, in_features]``
    row-major order, unchanged. ``descale`` is the flattened 128x4 grid of the
    checkpoint's linear UE8M0 rows, which the kernel consumes as ``b_descale``.
    """

    values: torch.Tensor
    descale: torch.Tensor

    @property
    def out_features(self) -> int:
        return self.values.shape[0]

    @property
    def in_features(self) -> int:
        return self.values.shape[1]

    @property
    def device(self) -> torch.device:
        return self.values.device


def prepare_mxfp8_dense(values, scales) -> Mxfp8DenseWeight:
    """Interleave linear UE8M0 rows into the GEMM's 128x4 grid; the payload passes through."""
    if values.dtype != torch.float8_e4m3fn or values.ndim != 2:
        raise ValueError("MXFP8 dense payloads must be rank-two E4M3 tensors")
    rows, columns = values.shape
    if columns % 32:
        raise ValueError("MXFP8 dense payloads require a 32-aligned input width")
    if tuple(scales.shape) != (rows, columns // 32):
        raise ValueError("MXFP8 dense scales must hold one UE8M0 value per 32 inputs")
    if scales.dtype not in (torch.uint8, torch.float8_e8m0fnu):
        raise ValueError("MXFP8 dense scales must be UE8M0 bytes")
    if values.device != scales.device or not values.is_contiguous():
        raise ValueError("MXFP8 dense weights must be contiguous on one CUDA device")
    return Mxfp8DenseWeight(values=values, descale=_interleave()(scales.view(torch.uint8)))


def dense_weight(weight) -> Mxfp8DenseWeight:
    """Adapt a prepared b12x MXFP8 dense weight to this kernel's layout."""
    from b12x.gemm.blockscaled._linear import MXFP8LinearWeight

    if not isinstance(weight, MXFP8LinearWeight):
        raise ValueError("SM110 dense projections support MXFP8 weights only")
    rows = weight.weight.values
    scales = weight.weight.scale_rows
    if scales.ndim == 3:
        if scales.shape[0] != 1:
            raise ValueError("SM110 dense projections require a single packed matrix")
        scales = scales[0]
    if weight.padded_in_features != weight.in_features:
        raise ValueError("SM110 dense projections require an unpadded input width")
    return prepare_mxfp8_dense(rows, scales)


@dataclass(frozen=True)
class Mxfp8ProjectionQuery:
    """Exact dense projection invocation; tactics depend only on these values."""

    num_tokens: int
    in_features: int
    out_features: int


def _tactics(query: Mxfp8ProjectionQuery):
    # One Ignition tactic: the cutlass MXFP8 kernel selects its own tile schedule.
    return (0,)


MXFP8_CONTRACT = native_contract(component_id="flare.sm110.mxfp8_projection",
                                 query_type=Mxfp8ProjectionQuery, tactics=_tactics,
                                 semantic_version=1)


def _scratch_layout(query):
    """Quantized activations and their 128x4-interleaved UE8M0 grid."""
    tokens, hidden = query.num_tokens, query.in_features
    values = _aligned(tokens * hidden)
    grid = (tokens + 127) // 128 * 128 * ((hidden // 32 + 3) // 4 * 4)
    return values, grid


def _scratch_bytes(query):
    return sum(_scratch_layout(query))


class _Mxfp8State:
    """The cutlass MXFP8 GEMM over one exact shape, bound to fixed storage."""

    implementation = "cutlass_mxfp8"

    def __init__(self, weight: Mxfp8DenseWeight, query: Mxfp8ProjectionQuery, tactic):
        from flashinfer.tllm_enums import SfLayout

        self.weight, self.query = weight, query
        self.quantize = _quantizer().mxfp8_quantize
        self.layout = SfLayout.layout_128x4.value
        # mm_mxfp8 takes B column-major, so the row-major checkpoint matrix is
        # transposed at bind time; the kernel reads it as [in, out].
        self.operand = weight.values.t()

    def scratch_bytes(self):
        return _scratch_bytes(self.query)

    def bind(self, source, output, scratch):
        tokens, hidden = source.shape
        values, _ = _scratch_layout(self.query)
        self.quantized = scratch[:values].view(torch.float8_e4m3fn).view(tokens, hidden)
        grid = _scratch_bytes(self.query) - values
        self.scales = scratch[values:values + grid].view(torch.uint8)

    def run(self, source, output, scratch):
        self.quantize(source, self.quantized, self.scales, self.layout, 32, True)
        from flashinfer import mm_mxfp8

        mm_mxfp8(self.quantized, self.operand, self.scales, self.weight.descale,
                 out=output, backend="cutlass")
        return output


def plan_projection(weight: Mxfp8DenseWeight, *, tokens: int):
    """Declare one exact-shape native dense MXFP8 projection."""
    query = Mxfp8ProjectionQuery(int(tokens), weight.in_features, weight.out_features)
    return native_plan(contract=MXFP8_CONTRACT, query=query, device=weight.device,
                       scratch_bytes=lambda tactic: _scratch_bytes(query),
                       materialize=lambda tactic: _Mxfp8State(weight, query, tactic))


@torch.library.custom_op("flare::sm110_mxfp8_out", mutates_args=("out", "scratch"))
def _sm110_mxfp8_out(source: torch.Tensor, out: torch.Tensor, scratch: torch.Tensor,
                     plan_handle: int) -> None:
    plan = plan_from_handle(plan_handle)
    require_prepared(plan, plan.component_id, source.device).run(source, out, scratch)


@_sm110_mxfp8_out.register_fake
def _fake_sm110_mxfp8_out(source, out, scratch, plan_handle) -> None:
    pass


def _tracing():
    from torch.fx.experimental.proxy_tensor import get_proxy_mode

    return torch.compiler.is_compiling() or get_proxy_mode() is not None


@dataclass(frozen=True, eq=False)
class Sm110Binding:
    """Exact tensors of one prepared native dense projection launch."""

    plan: object
    state: object
    source: torch.Tensor
    output: torch.Tensor
    scratch: torch.Tensor
    implementation: str

    def run(self):
        if _tracing():
            torch.ops.flare.sm110_mxfp8_out(self.source, self.output, self.scratch, self.plan._handle)
        else:
            self.state.run(self.source, self.output, self.scratch)
        return self.output


def _bind(plan, state, source, output, scratch):
    tokens, hidden = source.shape
    query = plan.query
    if (source.dtype != torch.bfloat16 or (tokens, hidden) != (query.num_tokens, query.in_features)
            or not source.is_contiguous()):
        raise ValueError("SM110 dense MXFP8 projections require the prepared contiguous BF16 source")
    if output.dtype != torch.bfloat16 or output.shape != (tokens, query.out_features) or not output.is_contiguous():
        raise ValueError("SM110 dense MXFP8 output must be contiguous BF16 [tokens, out_features]")
    if scratch.dtype != torch.uint8 or scratch.ndim != 1 or scratch.numel() < state.scratch_bytes():
        raise ValueError("SM110 dense MXFP8 scratch is smaller than its plan")
    if any(t.device != source.device for t in (output, scratch)):
        raise ValueError("SM110 dense MXFP8 tensors must share one device")
    state.bind(source, output, scratch)
    return Sm110Binding(plan, state, source, output, scratch, state.implementation)


def _benchmark_call(state, *, plan, source, output, session, name, workspace):
    operands = benchmark_resource(session, ("sm110_dense_inputs", name),
                                  projection_fixtures.input_factory(source, output))
    scratch = planned_scratch(plan.memory_requirements().scratch, workspace)
    binding = _bind(plan, state, operands.source, operands.output, scratch)
    return PreparedCall(run=binding.run, output=operands.output, produce=operands.produce,
                        owners=(operands, scratch, binding))


class Sm110Mxfp8Projection:
    """Prepared dense MXFP8 projection with fixed source and output storage.

    Mirrors ``Sm110RoutedExperts``: one exact-shape plan declared to the
    preparation session, bound to caller-owned scratch after tuning.
    """

    def __init__(self, source, weight, output, *, session, name: str,
                 workspace: ScratchArena | torch.Tensor | None = None,
                 bias: torch.Tensor | None = None, activation_mode: str = "auto"):
        # The kernel quantizes activations itself; a BF16-times-quantized GEMM is
        # not part of its contract, and no checkpoint scale exists to fold in.
        if activation_mode not in ("auto", "quantized"):
            raise ValueError("SM110 dense MXFP8 projections quantize activations dynamically; "
                             f"{activation_mode!r} activations are unsupported")
        prepared = dense_weight(weight)
        for tensor in (prepared.values, prepared.descale):
            record_numeric_tensor(tensor)
        self.source, self.output, self.bias = source, output, bias
        self.scratch = None
        self.plan = plan_projection(prepared, tokens=source.shape[0])
        plan = self.plan

        def prepare_call(state):
            scratch = planned_scratch(plan.memory_requirements().scratch, workspace)
            self.scratch = scratch
            binding = _bind(plan, state, source, output, scratch)
            return PreparedCall(run=binding.run, owners=prime_owners(workspace, scratch, binding))

        declare(session, (plan.request(
            name=name,
            prepare_call=prepare_call,
            benchmark_call=partial(_benchmark_call, plan=plan, source=source, output=output,
                                   session=session, name=name, workspace=workspace),
        ),))

        def finalize():
            self.scratch = planned_scratch(plan.scratch_specs(), workspace)
            state = require_prepared(plan, plan.component_id, source.device)
            self.binding = _bind(plan, state, source, output, self.scratch)

        register_finalizer(session, finalize, workspace=workspace)

    def __call__(self, source: torch.Tensor) -> torch.Tensor:
        output = self.binding.run()
        if self.bias is not None:
            output.add_(self.bias)
        return output


@dataclass(frozen=True)
class Bf16ProjectionQuery:
    """Exact unquantized dense projection; the kernel has one launch shape."""

    num_tokens: int
    in_features: int
    out_features: int


def _bf16_tactics(query: Bf16ProjectionQuery):
    # One Ignition tactic: cuDNN picks its own engine for the shape.
    return (0,)


BF16_CONTRACT = native_contract(component_id="flare.sm110.bf16_projection",
                                query_type=Bf16ProjectionQuery, tactics=_bf16_tactics,
                                semantic_version=1)


class _Bf16State:
    """The cuDNN BF16 GEMM over one exact shape.

    FlashInfer rejects the cutlass backend on capability 110, so this is the
    capability's dense BF16 kernel. It folds the bias in its epilogue and writes
    the caller's output, so a launch neither allocates nor copies.
    """

    implementation = "cudnn_bf16"

    def __init__(self, query: Bf16ProjectionQuery, tactic):
        self.query = query

    def run(self, source, weight, bias, output):
        from flashinfer import mm_bf16

        mm_bf16(source, weight.t(), bias=bias, out=output, backend="cudnn")
        return output


def plan_bf16_projection(*, tokens: int, in_features: int, out_features: int, device):
    """Declare one exact-shape native unquantized dense projection."""
    query = Bf16ProjectionQuery(int(tokens), int(in_features), int(out_features))
    return native_plan(contract=BF16_CONTRACT, query=query, device=device,
                       scratch_bytes=lambda tactic: 0,
                       materialize=lambda tactic: _Bf16State(query, tactic))


@torch.library.custom_op("flare::sm110_bf16_out", mutates_args=("out",))
def _sm110_bf16_out(source: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
                    out: torch.Tensor, plan_handle: int) -> None:
    plan = plan_from_handle(plan_handle)
    require_prepared(plan, plan.component_id, source.device).run(source, weight, bias, out)


@_sm110_bf16_out.register_fake
def _fake_sm110_bf16_out(source, weight, bias, out, plan_handle) -> None:
    pass


@dataclass(frozen=True, eq=False)
class Sm110Bf16Binding:
    """Exact tensors of one prepared native unquantized projection launch."""

    plan: object
    state: object
    source: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor | None
    output: torch.Tensor
    implementation: str

    def run(self):
        if _tracing():
            torch.ops.flare.sm110_bf16_out(self.source, self.weight, self.bias, self.output,
                                           self.plan._handle)
        else:
            self.state.run(self.source, self.weight, self.bias, self.output)
        return self.output


def _bind_bf16(plan, state, source, weight, bias, output):
    tokens, hidden = source.shape
    query = plan.query
    if (source.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or (tokens, hidden) != (query.num_tokens, query.in_features)
            or tuple(weight.shape) != (query.out_features, query.in_features)):
        raise ValueError("SM110 unquantized projections require the prepared contiguous BF16 operands")
    if not source.is_contiguous() or not weight.is_contiguous():
        raise ValueError("SM110 unquantized projections require contiguous operands")
    if output.dtype != torch.bfloat16 or output.shape != (tokens, query.out_features) or not output.is_contiguous():
        raise ValueError("SM110 unquantized output must be contiguous BF16 [tokens, out_features]")
    if bias is not None and (bias.dtype != torch.bfloat16 or tuple(bias.shape) != (query.out_features,)):
        raise ValueError("SM110 unquantized bias must be BF16 [out_features]")
    if any(t.device != source.device for t in (weight, output, *(() if bias is None else (bias,)))):
        raise ValueError("SM110 unquantized projection tensors must share one device")
    return Sm110Bf16Binding(plan, state, source, weight, bias, output, state.implementation)


def _benchmark_bf16(state, *, plan, source, weight, bias, output):
    binding = _bind_bf16(plan, state, source, weight, bias, output)
    return PreparedCall(run=binding.run, output=output, owners=(binding,))


class Sm110Bf16Projection:
    """Prepared unquantized dense projection with fixed source and output storage."""

    def __init__(self, source, weight, output, *, session, name: str,
                 bias: torch.Tensor | None = None):
        self.source, self.weight, self.output, self.bias = source, weight, output, bias
        self.plan = plan_bf16_projection(tokens=source.shape[0], in_features=source.shape[1],
                                         out_features=output.shape[-1], device=source.device)
        plan = self.plan

        def prepare_call(state):
            binding = _bind_bf16(plan, state, source, weight, bias, output)
            return PreparedCall(run=binding.run, owners=(binding,))

        declare(session, (plan.request(
            name=name,
            prepare_call=prepare_call,
            benchmark_call=partial(_benchmark_bf16, plan=plan, source=source, weight=weight,
                                   bias=bias, output=output),
        ),))

        def finalize():
            state = require_prepared(plan, plan.component_id, source.device)
            self.binding = _bind_bf16(plan, state, source, weight, bias, output)

        register_finalizer(session, finalize)

    def __call__(self, source: torch.Tensor) -> torch.Tensor:
        return self.binding.run()

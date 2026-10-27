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
    # The checkpoint's linear UE8M0 rows, kept so a padded payload can be
    # re-interleaved without re-reading the weight.
    scales: torch.Tensor | None = None

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
    linear = scales.view(torch.uint8)
    return Mxfp8DenseWeight(values=values, descale=_interleave()(linear), scales=linear)


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


# The cutlass MXFP8 kernel refuses fewer than 128 outputs; narrower payloads
# pad with zero rows, which contribute nothing to the product.
MINIMUM_OUT_FEATURES = 128


def _pad_narrow_rows(weight: Mxfp8DenseWeight) -> Mxfp8DenseWeight:
    """Pad a payload narrower than the kernel minimum with zero rows and scales."""
    rows = weight.out_features
    if rows >= MINIMUM_OUT_FEATURES:
        return weight
    if weight.scales is None:
        raise ValueError("SM110 dense MXFP8 weights must retain their linear UE8M0 rows")
    values = weight.values.new_zeros((MINIMUM_OUT_FEATURES, weight.in_features))
    values[:rows] = weight.values
    scales = weight.scales.new_zeros((MINIMUM_OUT_FEATURES, weight.scales.shape[1]))
    scales[:rows] = weight.scales
    return prepare_mxfp8_dense(values, scales)


@dataclass(frozen=True)
class Mxfp8ProjectionQuery:
    """Exact dense projection invocation; tactics depend only on these values."""

    num_tokens: int
    in_features: int
    out_features: int
    # A strided caller output, and a payload padded to the kernel's output
    # minimum, are written through contiguous staging storage.
    strided: bool = False
    padded_out_features: int = 0


def _staged(query) -> bool:
    return query.strided or query.padded_out_features > query.out_features


def _tactics(query: Mxfp8ProjectionQuery):
    # One Ignition tactic: the cutlass MXFP8 kernel selects its own tile schedule.
    return (0,)


MXFP8_CONTRACT = native_contract(component_id="flare.sm110.mxfp8_projection",
                                 query_type=Mxfp8ProjectionQuery, tactics=_tactics,
                                 semantic_version=1)


def _output_staging(query) -> int:
    """Contiguous staging bytes for a strided or padded caller output."""
    width = max(query.out_features, query.padded_out_features)
    return _aligned(query.num_tokens * width * 2) if _staged(query) else 0


def _scratch_layout(query):
    """Quantized activations, their 128x4-interleaved UE8M0 grid, and output staging."""
    tokens, hidden = query.num_tokens, query.in_features
    values = _aligned(tokens * hidden)
    grid = (tokens + 127) // 128 * 128 * ((hidden // 32 + 3) // 4 * 4)
    return values, grid, _output_staging(query)


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
        values, grid, staging = _scratch_layout(self.query)
        self.quantized = scratch[:tokens * hidden].view(torch.float8_e4m3fn).view(tokens, hidden)
        self.scales = scratch[values:values + grid].view(torch.uint8)
        width = max(self.query.out_features, self.query.padded_out_features)
        # Staging is aligned to 256 bytes, so the view stops at the exact extent.
        self.staging = (None if not staging else
                        scratch[values + grid:values + grid + staging].view(torch.bfloat16)
                        [:tokens * width].view(tokens, width))

    def run(self, source, output, scratch):
        self.quantize(source, self.quantized, self.scales, self.layout, 32, True)
        from flashinfer import mm_mxfp8

        target = output if self.staging is None else self.staging
        mm_mxfp8(self.quantized, self.operand, self.scales, self.weight.descale,
                 out=target, backend="cutlass")
        if self.staging is not None:
            output.copy_(self.staging[:, : self.query.out_features])
        return output


def plan_projection(weight: Mxfp8DenseWeight, *, tokens: int, strided: bool = False,
                    out_features: int | None = None):
    """Declare one exact-shape native dense MXFP8 projection."""
    query = Mxfp8ProjectionQuery(int(tokens), weight.in_features,
                                 weight.out_features if out_features is None else int(out_features),
                                 bool(strided), weight.out_features)
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
    if output.dtype != torch.bfloat16 or output.shape != (tokens, query.out_features):
        raise ValueError("SM110 dense MXFP8 output must be BF16 [tokens, out_features]")
    if not _staged(query) and not output.is_contiguous():
        raise ValueError("SM110 dense MXFP8 output must be contiguous")
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
        prepared = _pad_narrow_rows(dense_weight(weight))
        for tensor in (prepared.values, prepared.descale):
            record_numeric_tensor(tensor)
        self.source, self.output, self.bias = source, output, bias
        self.scratch = None
        self.plan = plan_projection(prepared, tokens=source.shape[0],
                                    strided=not output.is_contiguous(),
                                    out_features=weight.out_features)
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
    # A strided caller output is written through contiguous staging storage.
    strided: bool = False
    padded_out_features: int = 0


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
        self.staging = None

    def scratch_bytes(self) -> int:
        return _output_staging(self.query)

    def bind(self, scratch):
        query = self.query
        self.staging = (None if not query.strided else
                        scratch[:self.scratch_bytes()].view(torch.bfloat16)
                        [:query.num_tokens * query.out_features]
                        .view(query.num_tokens, query.out_features))

    def run(self, source, weight, bias, output):
        from flashinfer import mm_bf16

        target = output if self.staging is None else self.staging
        mm_bf16(source, weight.t(), bias=bias, out=target, backend="cudnn")
        if self.staging is not None:
            output.copy_(self.staging[:, : self.query.out_features])
        return output


def plan_bf16_projection(*, tokens: int, in_features: int, out_features: int, device,
                         strided: bool = False):
    """Declare one exact-shape native unquantized dense projection."""
    query = Bf16ProjectionQuery(int(tokens), int(in_features), int(out_features), bool(strided))
    return native_plan(contract=BF16_CONTRACT, query=query, device=device,
                       scratch_bytes=lambda tactic: _output_staging(query),
                       materialize=lambda tactic: _Bf16State(query, tactic))


@torch.library.custom_op("flare::sm110_bf16_out", mutates_args=("out",))
def _sm110_bf16_out(source: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
                    out: torch.Tensor, scratch: torch.Tensor, plan_handle: int) -> None:
    plan = plan_from_handle(plan_handle)
    require_prepared(plan, plan.component_id, source.device).run(source, weight, bias, out)


@_sm110_bf16_out.register_fake
def _fake_sm110_bf16_out(source, weight, bias, out, scratch, plan_handle) -> None:
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
    scratch: torch.Tensor
    implementation: str

    def run(self):
        if _tracing():
            torch.ops.flare.sm110_bf16_out(self.source, self.weight, self.bias, self.output,
                                           self.scratch, self.plan._handle)
        else:
            self.state.run(self.source, self.weight, self.bias, self.output)
        return self.output


def _bind_bf16(plan, state, source, weight, bias, output, scratch):
    tokens, hidden = source.shape
    query = plan.query
    if (source.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or (tokens, hidden) != (query.num_tokens, query.in_features)
            or tuple(weight.shape) != (query.out_features, query.in_features)):
        raise ValueError("SM110 unquantized projections require the prepared contiguous BF16 operands")
    if not source.is_contiguous() or not weight.is_contiguous():
        raise ValueError("SM110 unquantized projections require contiguous operands")
    if output.dtype != torch.bfloat16 or output.shape != (tokens, query.out_features):
        raise ValueError("SM110 unquantized output must be BF16 [tokens, out_features]")
    if not query.strided and not output.is_contiguous():
        raise ValueError("SM110 unquantized output must be contiguous")
    if bias is not None and (bias.dtype != torch.bfloat16 or tuple(bias.shape) != (query.out_features,)):
        raise ValueError("SM110 unquantized bias must be BF16 [out_features]")
    if scratch.dtype != torch.uint8 or scratch.ndim != 1 or scratch.numel() < state.scratch_bytes():
        raise ValueError("SM110 unquantized scratch is smaller than its plan")
    if any(t.device != source.device for t in (weight, output, scratch, *(() if bias is None else (bias,)))):
        raise ValueError("SM110 unquantized projection tensors must share one device")
    state.bind(scratch)
    return Sm110Bf16Binding(plan, state, source, weight, bias, output, scratch, state.implementation)


def _benchmark_bf16(state, *, plan, source, weight, bias, output, workspace):
    scratch = planned_scratch(plan.memory_requirements().scratch, workspace)
    binding = _bind_bf16(plan, state, source, weight, bias, output, scratch)
    return PreparedCall(run=binding.run, output=output, owners=(scratch, binding))


class Sm110Bf16Projection:
    """Prepared unquantized dense projection with fixed source and output storage."""

    def __init__(self, source, weight, output, *, session, name: str,
                 bias: torch.Tensor | None = None, workspace: ScratchArena | torch.Tensor | None = None):
        self.source, self.weight, self.output, self.bias = source, weight, output, bias
        self.scratch = None
        self.plan = plan_bf16_projection(tokens=source.shape[0], in_features=source.shape[1],
                                         out_features=output.shape[-1], device=source.device,
                                         strided=not output.is_contiguous())
        plan = self.plan

        def prepare_call(state):
            scratch = planned_scratch(plan.memory_requirements().scratch, workspace)
            self.scratch = scratch
            binding = _bind_bf16(plan, state, source, weight, bias, output, scratch)
            return PreparedCall(run=binding.run, owners=(scratch, binding))

        declare(session, (plan.request(
            name=name,
            prepare_call=prepare_call,
            benchmark_call=partial(_benchmark_bf16, plan=plan, source=source, weight=weight,
                                   bias=bias, output=output, workspace=workspace),
        ),))

        def finalize():
            self.scratch = planned_scratch(plan.scratch_specs(), workspace)
            state = require_prepared(plan, plan.component_id, source.device)
            self.binding = _bind_bf16(plan, state, source, weight, bias, output, self.scratch)

        register_finalizer(session, finalize, workspace=workspace)

    def __call__(self, source: torch.Tensor) -> torch.Tensor:
        return self.binding.run()


@dataclass(frozen=True)
class Nvfp4A16DenseWeight:
    """Checkpoint NVFP4 dense storage for the 16-bit-activation GEMM.

    ``values`` is the checkpoint's packed E2M1 payload in ``[out_features,
    in_features // 2]`` order. ``scales`` holds one E4M3 block scale per 16
    inputs in the kernel's 128x4 grid. ``dequant_scale`` multiplies the decoded
    ``E2M1 * E4M3`` product.
    """

    values: torch.Tensor
    scales: torch.Tensor
    dequant_scale: torch.Tensor

    @property
    def out_features(self) -> int:
        return self.values.shape[0]

    @property
    def in_features(self) -> int:
        return self.values.shape[1] * 2

    @property
    def device(self) -> torch.device:
        return self.values.device


def prepare_nvfp4_a16_dense(values, scales, dequant_scale) -> Nvfp4A16DenseWeight:
    """Validate checkpoint NVFP4 storage; the payload and scales pass through."""
    if values.dtype != torch.uint8 or values.ndim != 2:
        raise ValueError("NVFP4 dense payloads must be rank-two packed bytes")
    rows, columns = values.shape
    if columns % 8:
        raise ValueError("NVFP4 dense payloads require a 16-aligned input width")
    if dequant_scale.dtype != torch.float32 or dequant_scale.numel() != 1:
        raise ValueError("NVFP4 dense weights require one float32 dequantization scalar")
    if scales.dtype not in (torch.uint8, torch.float8_e4m3fn):
        raise ValueError("NVFP4 dense scales must be E4M3 bytes")
    required = ((rows + 127) // 128) * ((columns // 8 + 3) // 4) * 512
    if scales.numel() * scales.element_size() < required:
        raise ValueError("NVFP4 dense scales are smaller than the 128x4 grid")
    if values.device != scales.device or not values.is_contiguous():
        raise ValueError("NVFP4 dense weights must be contiguous on one CUDA device")
    return Nvfp4A16DenseWeight(values=values, scales=_scale_bytes(scales),
                               dequant_scale=dequant_scale)


def _scale_bytes(scales) -> torch.Tensor:
    """The 128x4 grid as the kernel's byte buffer.

    Prepared layouts expose the grid through permuted multi-dimensional views, so
    the buffer is read in storage order rather than in the view's logical order.
    """
    view = scales.view(torch.uint8)
    if view.is_contiguous():
        return view.reshape(-1)
    if view.storage_offset() != 0 or view.untyped_storage().nbytes() != view.numel():
        raise ValueError("NVFP4 dense scales must span one contiguous buffer")
    buffer = torch.empty(view.numel(), dtype=torch.uint8, device=view.device)
    buffer.set_(view.untyped_storage(), 0, (view.numel(),), (1,))
    return buffer


def nvfp4_a16_weight(weight) -> Nvfp4A16DenseWeight:
    """Adapt a prepared b12x NVFP4 dense weight to this kernel's inputs."""
    from b12x.gemm.blockscaled import NVFP4LinearWeight

    if not isinstance(weight, NVFP4LinearWeight):
        raise ValueError("SM110 16-bit-activation dense projections support NVFP4 weights only")
    if weight.global_scale_kind == "multiplier":
        dequant_scale = weight.global_scale
    elif weight.global_scale_kind == "reciprocal":
        dequant_scale = 1.0 / weight.global_scale
    else:
        raise ValueError(f"unknown NVFP4 global scale kind {weight.global_scale_kind!r}")
    values, scales = weight.values, weight.scale_mma
    if weight.padded_in_features != weight.in_features:
        # The checkpoint pads the stored input width to 128 and zeroes the extra
        # columns, so the logical width is the exact matrix the kernel needs.
        from flashinfer.fp4_quantization import block_scale_interleave

        values = values[:, : weight.in_features // 2]
        linear = _linear_scales(scales, weight.out_features)[:, : weight.in_features // 16]
        scales = block_scale_interleave(linear.contiguous().view(torch.uint8))
    return prepare_nvfp4_a16_dense(values.contiguous(), scales,
                                   dequant_scale.reshape(1).to(torch.float32))


def _linear_scales(scales, rows: int) -> torch.Tensor:
    """Invert the 128x4 grid back to one E4M3 value per 16 inputs."""
    from b12x._lib.intrinsics.reference import make_swizzle_indices

    buffer = _scale_bytes(scales)
    rows_padded = (rows + 127) // 128 * 128
    if buffer.numel() % rows_padded:
        raise ValueError("NVFP4 dense scales do not form a 128x4 grid")
    columns_padded = buffer.numel() // rows_padded
    indices = make_swizzle_indices(rows_padded, columns_padded, buffer.device)
    linear = torch.empty_like(buffer)
    linear[indices] = buffer
    return linear.view(torch.float8_e4m3fn).view(rows_padded, columns_padded)[:rows]


@dataclass(frozen=True)
class Nvfp4A16Query:
    """Exact 16-bit-activation dense projection; tactics depend only on these values."""

    num_tokens: int
    in_features: int
    out_features: int


def _a16_tactics(query: Nvfp4A16Query):
    # One Ignition tactic: the cuDNN W4A16 kernel selects its own engine.
    return (0,)


NVFP4_A16_CONTRACT = native_contract(component_id="flare.sm110.bf16_fp4_projection",
                                     query_type=Nvfp4A16Query, tactics=_a16_tactics,
                                     semantic_version=1)


class _Nvfp4A16State:
    """The cuDNN BF16 x FP4 GEMM over one exact shape.

    The remaining W4A16 backends reject capability 110, so cuDNN is the
    capability's 16-bit-activation kernel. Weight preparation happens once when
    this state is materialized; a launch writes the caller's output.
    """

    implementation = "cudnn_bf16_fp4"

    def __init__(self, weight: Nvfp4A16DenseWeight, query: Nvfp4A16Query, tactic):
        import flashinfer

        self.query = query
        self.prepared = flashinfer.prepare_bf16_fp4_weights(
            weight.values, weight.scales, weight.dequant_scale, backend="cudnn")

    def run(self, source, values, scales, dequant_scale, output):
        import flashinfer

        b_prepared, sf_prepared, alpha_prepared = self.prepared
        flashinfer.mm_bf16_fp4(source, b_prepared, sf_prepared, alpha_prepared,
                               backend="cudnn", out=output)
        return output


def plan_nvfp4_a16_projection(*, tokens: int, in_features: int, out_features: int, device,
                              weight: Nvfp4A16DenseWeight):
    """Declare one exact-shape native 16-bit-activation dense projection."""
    query = Nvfp4A16Query(int(tokens), int(in_features), int(out_features))
    return native_plan(contract=NVFP4_A16_CONTRACT, query=query, device=device,
                       scratch_bytes=lambda tactic: 0,
                       materialize=lambda tactic: _Nvfp4A16State(weight, query, tactic))


@torch.library.custom_op("flare::sm110_bf16_fp4_out", mutates_args=("out",))
def _sm110_bf16_fp4_out(source: torch.Tensor, values: torch.Tensor, scales: torch.Tensor,
                        dequant_scale: torch.Tensor, out: torch.Tensor, plan_handle: int) -> None:
    plan = plan_from_handle(plan_handle)
    require_prepared(plan, plan.component_id, source.device).run(
        source, values, scales, dequant_scale, out)


@_sm110_bf16_fp4_out.register_fake
def _fake_sm110_bf16_fp4_out(source, values, scales, dequant_scale, out, plan_handle) -> None:
    pass


@dataclass(frozen=True, eq=False)
class Sm110Nvfp4A16Binding:
    """Exact tensors of one prepared native 16-bit-activation projection launch."""

    plan: object
    state: object
    source: torch.Tensor
    weight: Nvfp4A16DenseWeight
    output: torch.Tensor
    implementation: str

    def run(self):
        if _tracing():
            torch.ops.flare.sm110_bf16_fp4_out(self.source, self.weight.values, self.weight.scales,
                                               self.weight.dequant_scale, self.output,
                                               self.plan._handle)
        else:
            self.state.run(self.source, self.weight.values, self.weight.scales,
                           self.weight.dequant_scale, self.output)
        return self.output


def _bind_nvfp4_a16(plan, state, source, weight, output):
    tokens, hidden = source.shape
    query = plan.query
    if (source.dtype != torch.bfloat16 or not source.is_contiguous()
            or (tokens, hidden) != (query.num_tokens, query.in_features)
            or weight.in_features != query.in_features
            or weight.out_features != query.out_features):
        raise ValueError("SM110 16-bit-activation projections require the prepared contiguous BF16 source")
    if output.dtype != torch.bfloat16 or output.shape != (tokens, query.out_features) or not output.is_contiguous():
        raise ValueError("SM110 16-bit-activation output must be contiguous BF16 [tokens, out_features]")
    if any(t.device != source.device for t in (weight.values, weight.scales, weight.dequant_scale, output)):
        raise ValueError("SM110 16-bit-activation projection tensors must share one device")
    return Sm110Nvfp4A16Binding(plan, state, source, weight, output, state.implementation)


def _benchmark_nvfp4_a16(state, *, plan, source, weight, output):
    binding = _bind_nvfp4_a16(plan, state, source, weight, output)
    return PreparedCall(run=binding.run, output=output, owners=(binding,))


class Sm110Nvfp4A16Projection:
    """Prepared 16-bit-activation dense projection with fixed source and output storage."""

    def __init__(self, source, weight, output, *, session, name: str,
                 bias: torch.Tensor | None = None):
        prepared = nvfp4_a16_weight(weight)
        for tensor in (prepared.values, prepared.scales, prepared.dequant_scale):
            record_numeric_tensor(tensor)
        self.source, self.output, self.bias = source, output, bias
        self.plan = plan_nvfp4_a16_projection(tokens=source.shape[0],
                                             in_features=prepared.in_features,
                                             out_features=prepared.out_features,
                                             device=source.device, weight=prepared)
        plan = self.plan

        def prepare_call(state):
            binding = _bind_nvfp4_a16(plan, state, source, prepared, output)
            return PreparedCall(run=binding.run, owners=(binding,))

        declare(session, (plan.request(
            name=name,
            prepare_call=prepare_call,
            benchmark_call=partial(_benchmark_nvfp4_a16, plan=plan, source=source,
                                   weight=prepared, output=output),
        ),))

        def finalize():
            state = require_prepared(plan, plan.component_id, source.device)
            self.binding = _bind_nvfp4_a16(plan, state, source, prepared, output)

        register_finalizer(session, finalize)

    def __call__(self, source: torch.Tensor) -> torch.Tensor:
        output = self.binding.run()
        if self.bias is not None:
            output.add_(self.bias)
        return output

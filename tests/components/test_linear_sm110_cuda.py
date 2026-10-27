"""Dense MXFP8 projections on SM110 through FlashInfer's cutlass MXFP8 GEMM.

Exercises the prepared component at Qwen3.8 ``in_proj_qkv`` geometry: activation
quantization into caller-owned scratch, compiled forwards, CUDA graph capture,
allocation-stable replay, and numerics against the exact dequantization of the
same tensors.

Weights are synthetic in the checkpoint layout: E4M3 payloads with linear
32-wide UE8M0 rows, which the component re-lays out into the kernel's 128x4 grid
without requantizing. The reference dequantizes the activations the component
consumes, which separates weight-adapter correctness from MXFP8 activation loss;
the total path is bounded separately.
"""

from __future__ import annotations

import json
import os

import pytest
import torch
import torch.nn.functional as F
from b12x.gemm._shared.wo_mxfp8 import MXFP8Rows
from b12x.gemm.blockscaled._linear import MXFP8LinearWeight
from b12x.preparation import PreparationSession
from flashinfer.quantization import mxfp8_quantize

from ignition.components import linear
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an explicitly assigned CUDA_VISIBLE_DEVICES",
)

IN_FEATURES, OUT_FEATURES = 2560, 10240
WEIGHT_STD = 0.02
# Adapter error against dequantized activations, and total error against the
# BF16 source including MXFP8 activation loss. Measured on Thor across tokens
# 1/16/128/512 and both eager and compiled forwards: 0.001652-0.001667 and
# 0.026510-0.026743.
ADAPTER_BOUND, TOTAL_BOUND = 0.003, 0.035


def _decode(values, scales):
    """Exact dequantization of one E4M3 plane with 32-wide UE8M0 block scales."""
    exponents = scales.view(torch.uint8).float()
    return values.float() * torch.pow(2.0, exponents - 127.0).repeat_interleave(32, dim=1)


@pytest.fixture(scope="module")
def weight():
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.skip("assign a CUDA device explicitly")
    if torch.cuda.get_device_capability(0) != (11, 0):
        pytest.skip("SM110 dense projections require a Jetson Thor")
    generator = torch.Generator(device="cuda").manual_seed(110)
    source = torch.randn((OUT_FEATURES, IN_FEATURES), generator=generator, device="cuda") * WEIGHT_STD
    values, scales = mxfp8_quantize(source.to(torch.bfloat16), is_sf_swizzled_layout=False)
    scales = scales.view(OUT_FEATURES, IN_FEATURES // 32)
    prepared = MXFP8LinearWeight(
        weight=MXFP8Rows(values=values, scale_rows=scales.view(torch.float8_e8m0fnu).unsqueeze(0),
                         scale_mma=None),
        in_features=IN_FEATURES, padded_in_features=IN_FEATURES, out_features=OUT_FEATURES)
    return prepared, _decode(values, scales)


def _metrics(actual, expected):
    difference = actual.float().cpu() - expected.float().cpu()
    return {"max_absolute_error": difference.abs().max().item(),
            "relative_l2_error": (difference.norm() / expected.float().norm().clamp_min(1e-12)).item()}


def _run_component(weight, tokens, *, compiled):
    prepared, decoded = weight
    source = torch.zeros((tokens, IN_FEATURES), dtype=torch.bfloat16, device="cuda")
    output = torch.empty((tokens, OUT_FEATURES), dtype=torch.bfloat16, device="cuda")
    workspace = ScratchArena("cuda:0")
    with torch.inference_mode(), PreparationSession(device=source.device, autotune=True,
                                                    compile_workers=0) as session:
        projection = linear.QuantizedProjection(source, prepared, output, session=session,
                                                name=f"sm110_dense:{tokens}", workspace=workspace)
        assert projection.sm110 is not None
        session.freeze()
        workspace.finalize()
        workspace.freeze()
        assert projection.sm110.plan.selection.source in ("tuned", "cached", "fixed")

        run = compile_forward(lambda: projection(source)) if compiled else (lambda: projection(source))
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        generator = torch.Generator(device="cuda").manual_seed(1103 + tokens)
        for iteration in range(2):
            activations = (torch.randn((tokens, IN_FEATURES), generator=generator, device="cuda") * 0.5)
            activations = activations.to(torch.bfloat16)
            source.copy_(activations)
            output.fill_(float("nan"))
            workspace.tensor.fill_(0x7F)
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            # The kernel quantizes activations dynamically; dequantize the same
            # pack to separate the weight adapter from MXFP8 activation loss.
            quantized, scales = mxfp8_quantize(activations, is_sf_swizzled_layout=False)
            dequantized = _decode(quantized, scales.view(tokens, IN_FEATURES // 32))
            adapter = _metrics(output, dequantized @ decoded.T)
            total = _metrics(output, activations.float() @ decoded.T)
            assert adapter["relative_l2_error"] < ADAPTER_BOUND, adapter
            assert total["relative_l2_error"] < TOTAL_BOUND, total
            print(json.dumps({"component": "cutlass_mxfp8", "tokens": tokens, "compiled": compiled,
                              "iteration": iteration,
                              "implementation": projection.sm110.binding.implementation,
                              "adapter": adapter, "total": total}), flush=True)


@pytest.mark.parametrize("tokens", [1, 16, 128, 512])
@pytest.mark.parametrize("compiled", [False, True])
def test_mxfp8_dense_uses_cutlass(weight, tokens, compiled):
    _run_component(weight, tokens, compiled=compiled)


def test_narrow_width_scratch_views_its_exact_extent():
    """Widths whose padding is not 256-aligned view only their own extent."""
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.skip("assign a CUDA device explicitly")
    if torch.cuda.get_device_capability(0) != (11, 0):
        pytest.skip("SM110 dense projections require a Jetson Thor")
    generator = torch.Generator(device="cuda").manual_seed(640)
    in_features, out_features, tokens = 640, 2560, 1
    source = torch.randn((out_features, in_features), generator=generator, device="cuda") * WEIGHT_STD
    values, scales = mxfp8_quantize(source.to(torch.bfloat16), is_sf_swizzled_layout=False)
    scales = scales.view(out_features, in_features // 32)
    prepared = MXFP8LinearWeight(
        weight=MXFP8Rows(values=values, scale_rows=scales.view(torch.float8_e8m0fnu).unsqueeze(0),
                         scale_mma=None),
        in_features=in_features, padded_in_features=in_features, out_features=out_features)
    decoded = _decode(values, scales)
    activations = torch.randn((tokens, in_features), generator=generator, device="cuda") * 0.5
    activations = activations.to(torch.bfloat16)
    output = torch.zeros((tokens, out_features), dtype=torch.bfloat16, device="cuda")
    workspace = ScratchArena("cuda:0")
    with torch.inference_mode(), PreparationSession(device="cuda", autotune=True, compile_workers=0) as session:
        projection = linear.QuantizedProjection(activations, prepared, output, session=session,
                                               name="sm110_mxfp8_narrow", workspace=workspace)
        session.freeze()
        workspace.finalize()
        workspace.freeze()
        projection(activations)
        torch.cuda.synchronize()
        quantized, quantized_scales = mxfp8_quantize(activations, is_sf_swizzled_layout=False)
        dequantized = _decode(quantized, quantized_scales.view(tokens, in_features // 32))
        metrics = _metrics(output, dequantized @ decoded.T)
        assert metrics["relative_l2_error"] < ADAPTER_BOUND, metrics


def test_narrow_output_pads_to_the_kernel_minimum():
    """Outputs below the kernel's 128 minimum pad with zero rows."""
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.skip("assign a CUDA device explicitly")
    if torch.cuda.get_device_capability(0) != (11, 0):
        pytest.skip("SM110 dense projections require a Jetson Thor")
    generator = torch.Generator(device="cuda").manual_seed(96)
    in_features, out_features, tokens = 2560, 96, 16
    source = torch.randn((out_features, in_features), generator=generator, device="cuda") * WEIGHT_STD
    values, scales = mxfp8_quantize(source.to(torch.bfloat16), is_sf_swizzled_layout=False)
    scales = scales.view(out_features, in_features // 32)
    prepared = MXFP8LinearWeight(
        weight=MXFP8Rows(values=values, scale_rows=scales.view(torch.float8_e8m0fnu).unsqueeze(0),
                         scale_mma=None),
        in_features=in_features, padded_in_features=in_features, out_features=out_features)
    decoded = _decode(values, scales)
    activations = torch.randn((tokens, in_features), generator=generator, device="cuda") * 0.5
    activations = activations.to(torch.bfloat16)
    output = torch.zeros((tokens, out_features), dtype=torch.bfloat16, device="cuda")
    workspace = ScratchArena("cuda:0")
    with torch.inference_mode(), PreparationSession(device="cuda", autotune=True, compile_workers=0) as session:
        projection = linear.QuantizedProjection(activations, prepared, output, session=session,
                                               name="sm110_mxfp8_narrow_out", workspace=workspace)
        session.freeze()
        workspace.finalize()
        workspace.freeze()
        projection(activations)
        torch.cuda.synchronize()
        quantized, quantized_scales = mxfp8_quantize(activations, is_sf_swizzled_layout=False)
        dequantized = _decode(quantized, quantized_scales.view(tokens, in_features // 32))
        metrics = _metrics(output, dequantized @ decoded.T)
        assert metrics["relative_l2_error"] < ADAPTER_BOUND, metrics

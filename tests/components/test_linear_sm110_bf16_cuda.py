"""Unquantized BF16 dense projections on SM110 through FlashInfer's cuDNN GEMM.

Exercises the prepared component at Qwen3.8 dense geometry: compiled forwards,
CUDA graph capture, allocation-stable replay, and numerics against the exact
FP32 product including the folded bias.

b12x's BF16 GEMV targets SM120/SM121 and FlashInfer's cutlass BF16 backend
rejects capability 110, so cuDNN is the capability's dense BF16 kernel.
"""

from __future__ import annotations

import json
import os

import pytest
import torch
from b12x.preparation import PreparationSession

from ignition.components import linear
from ignition.preparation.compilation import compile_forward

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an explicitly assigned CUDA_VISIBLE_DEVICES",
)

IN_FEATURES, OUT_FEATURES = 2560, 10240
WEIGHT_STD = 0.02
# BF16 output rounding of an FP32-accumulated product, measured on Thor at
# 0.001642-0.001695 across every case below.
ERROR_BOUND = 0.003


@pytest.fixture(scope="module")
def operands():
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.skip("assign a CUDA device explicitly")
    if torch.cuda.get_device_capability(0) != (11, 0):
        pytest.skip("SM110 dense projections require a Jetson Thor")
    generator = torch.Generator(device="cuda").manual_seed(110)
    weight = torch.randn((OUT_FEATURES, IN_FEATURES), generator=generator, device="cuda") * WEIGHT_STD
    bias = torch.randn((OUT_FEATURES,), generator=generator, device="cuda") * WEIGHT_STD
    return weight.to(torch.bfloat16), bias.to(torch.bfloat16)


def _metrics(actual, expected):
    difference = actual.float().cpu() - expected.float().cpu()
    return {"max_absolute_error": difference.abs().max().item(),
            "relative_l2_error": (difference.norm() / expected.float().norm().clamp_min(1e-12)).item()}


def _run_component(operands, tokens, *, compiled):
    weight, bias = operands
    source = torch.zeros((tokens, IN_FEATURES), dtype=torch.bfloat16, device="cuda")
    output = torch.empty((tokens, OUT_FEATURES), dtype=torch.bfloat16, device="cuda")
    with torch.inference_mode(), PreparationSession(device=source.device, autotune=True,
                                                    compile_workers=0) as session:
        projection = linear.UnquantizedProjection(source, weight, output, session=session,
                                                  name=f"sm110_bf16:{tokens}", bias=bias)
        assert projection.sm110 is not None
        session.freeze()

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
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            expected = activations.float() @ weight.float().T + bias.float()
            metrics = _metrics(output, expected)
            assert metrics["relative_l2_error"] < ERROR_BOUND, metrics
            print(json.dumps({"component": "cudnn_bf16", "tokens": tokens, "compiled": compiled,
                              "iteration": iteration,
                              "implementation": projection.sm110.binding.implementation,
                              **metrics}), flush=True)


@pytest.mark.parametrize("tokens", [1, 16, 128, 512])
@pytest.mark.parametrize("compiled", [False, True])
def test_bf16_dense_uses_cudnn(operands, tokens, compiled):
    _run_component(operands, tokens, compiled=compiled)


def test_strided_output_is_staged(operands):
    """A column slice of a wider buffer is written through contiguous staging."""
    weight, bias = operands
    tokens = 16
    wide = torch.zeros((tokens, OUT_FEATURES + 256), dtype=torch.bfloat16, device="cuda")
    output = wide[:, :OUT_FEATURES]
    assert not output.is_contiguous()
    source = torch.zeros((tokens, IN_FEATURES), dtype=torch.bfloat16, device="cuda")
    with torch.inference_mode(), PreparationSession(device=source.device, autotune=True,
                                                    compile_workers=0) as session:
        projection = linear.UnquantizedProjection(source, weight, output, session=session,
                                                  name="sm110_bf16_strided", bias=bias)
        session.freeze()
        generator = torch.Generator(device="cuda").manual_seed(99)
        activations = (torch.randn((tokens, IN_FEATURES), generator=generator, device="cuda") * 0.5)
        activations = activations.to(torch.bfloat16)
        source.copy_(activations)
        projection(source)
        torch.cuda.synchronize()
        expected = activations.float() @ weight.float().T + bias.float()
        metrics = _metrics(output, expected)
        assert metrics["relative_l2_error"] < ERROR_BOUND, metrics
        assert torch.count_nonzero(wide[:, OUT_FEATURES:]) == 0


def test_fp32_operands_are_rejected(operands):
    weight, bias = operands
    source = torch.zeros((4, IN_FEATURES), dtype=torch.float32, device="cuda")
    output = torch.empty((4, OUT_FEATURES), dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="BF16"):
        with torch.inference_mode(), PreparationSession(device=source.device, autotune=True,
                                                        compile_workers=0) as session:
            linear.UnquantizedProjection(source, weight.float(), output, session=session,
                                         name="sm110_bf16_fp32", bias=bias.float())
            session.freeze()

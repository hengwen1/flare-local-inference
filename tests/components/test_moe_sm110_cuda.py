"""Checkpoint routed experts on SM110 through FlashInfer's CUTLASS NVFP4 MoE.

Exercises the prepared component at Qwen3.8 expert geometry: b12x routing,
activation quantization into caller-owned scratch, compiled forwards, CUDA
graph capture, allocation-stable replay, and numerics against the exact
dequantization of the same tensors.

Weights are synthetic in the checkpoint layout and at checkpoint magnitudes:
packed E2M1 payloads, linear E4M3 block scales, and the per-expert global
multipliers the kernel consumes as dequant scalars. Magnitude decides whether
this test measures the adapter at all: fixtures an order of magnitude away from
the release measured 0.53 relative L2 through every global-scale convention
tried, and only checkpoint-magnitude weights reproduced the probe's 0.086. The
Qwen3.8 release stores weights near sigma 0.02 with per-expert multipliers near
1e-4; ``_quantize_checkpoint_style`` reproduces both.

The reference dequantizes the activations the component consumes rather than
the BF16 source, which separates weight-adapter correctness from FP4
activation loss; the total path is reported separately.
"""

from __future__ import annotations

import json
import os

import pytest
import torch
import torch.nn.functional as F
from b12x.preparation import PreparationSession
from flashinfer.fp4_quantization import fp4_quantize

from ignition.components.moe import PreparedRoutedExperts, PreparedRouter, RouterScratch
from ignition.components.moe_sm110 import Sm110RoutedExperts, prepare_cutlass_nvfp4_experts
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an explicitly assigned CUDA_VISIBLE_DEVICES",
)

EXPERTS, HIDDEN, INTERMEDIATE, TOP_K = 4, 2560, 640, 2
WEIGHT_STD = 0.02
# Adapter error against dequantized activations, and total error against the
# BF16 source including FP4 activation loss. Measured on Thor across tokens
# 1/4/24/512 and both eager and compiled forwards: 0.0894-0.0906 and
# 0.1631-0.1640.
ADAPTER_BOUND, TOTAL_BOUND = 0.10, 0.18


def _decode_nvfp4(packed: torch.Tensor, scales: torch.Tensor, global_scale: float) -> torch.Tensor:
    """Exact dequantization of one packed E2M1 plane with linear E4M3 scales."""
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], device=packed.device)
    low, high = (packed % 16).long(), (packed // 16).long()
    values = lut[torch.stack((low, high), dim=-1)].reshape(packed.shape[0], -1)
    blocks = values.reshape(packed.shape[0], -1, 16) * scales.view(torch.float8_e4m3fn).float().unsqueeze(-1)
    return blocks.reshape(packed.shape[0], -1) * global_scale


def _quantize_checkpoint_style(weight, device, global_scale=None):
    """Packed E2M1 with linear block scales and the checkpoint's global multiplier."""
    if global_scale is None:
        global_scale = float(weight.abs().amax().item()) / 448.0
    packed, scales = fp4_quantize(weight, torch.tensor([1.0 / global_scale], device=device),
                                  sf_vec_size=16, is_sf_swizzled_layout=False)
    return packed, scales.view(torch.uint8), global_scale


@pytest.fixture(scope="module")
def experts():
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.skip("assign a CUDA device explicitly")
    if torch.cuda.get_device_capability(0) != (11, 0):
        pytest.skip("SM110 routed experts require a Jetson Thor")
    device = torch.device("cuda", 0)
    generator = torch.Generator(device="cuda").manual_seed(110)
    planes = {}
    scale_vectors = ([], [], [], [])
    for expert in range(EXPERTS):
        up = torch.randn((INTERMEDIATE, HIDDEN), generator=generator, device=device) * WEIGHT_STD
        gate = torch.randn((INTERMEDIATE, HIDDEN), generator=generator, device=device) * WEIGHT_STD
        down = torch.randn((HIDDEN, INTERMEDIATE), generator=generator, device=device) * WEIGHT_STD
        # The checkpoint stores one FC1 multiplier shared by up and gate.
        fc1 = max(up.abs().amax().item(), gate.abs().amax().item()) / 448.0
        planes[expert, "up"] = _quantize_checkpoint_style(up.to(torch.bfloat16), device, fc1)
        planes[expert, "gate"] = _quantize_checkpoint_style(gate.to(torch.bfloat16), device, fc1)
        planes[expert, "down"] = _quantize_checkpoint_style(down.to(torch.bfloat16), device)
        scale_vectors[0].append(fc1)
        scale_vectors[1].append(planes[expert, "down"][2])
        scale_vectors[2].append(1.0)
        scale_vectors[3].append(1.0)
    w13 = torch.stack([torch.cat([planes[e, "up"][0], planes[e, "gate"][0]]) for e in range(EXPERTS)])
    s13 = torch.stack([torch.cat([planes[e, "up"][1], planes[e, "gate"][1]]) for e in range(EXPERTS)])
    w2 = torch.stack([planes[e, "down"][0] for e in range(EXPERTS)])
    s2 = torch.stack([planes[e, "down"][1] for e in range(EXPERTS)])
    view = prepare_cutlass_nvfp4_experts(w13.cuda(), s13.cuda(), w2.cuda(), s2.cuda(), scale_vectors)
    decoded = {expert: tuple(
        _decode_nvfp4(planes[expert, name][0].cuda(), planes[expert, name][1].cuda(), planes[expert, name][2])
        for name in ("up", "gate", "down")) for expert in range(EXPERTS)}
    return view, decoded


def _reference(decoded, expert, values):
    """Exact per-expert MoE from dequantized weights, silu on the gate branch."""
    up, gate, down = decoded[expert]
    values = values.float()
    return (F.silu(values @ gate.T) * (values @ up.T)) @ down.T


def _metrics(actual, expected):
    difference = actual.float().cpu() - expected.float().cpu()
    return {"max_absolute_error": difference.abs().max().item(),
            "relative_l2_error": (difference.norm() / expected.float().norm().clamp_min(1e-12)).item()}


def _expect(decoded, ids, probabilities, values):
    expected = torch.zeros_like(values, dtype=torch.float32)
    for expert in range(EXPERTS):
        rows, slots = (ids == expert).nonzero(as_tuple=True)
        if rows.numel():
            expected[rows] += _reference(decoded, expert, values[rows]) * probabilities[rows, slots, None]
    return expected


def _run_component(view, decoded, tokens, *, compiled):
    source = torch.zeros((tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
    logits = torch.zeros((tokens, EXPERTS), dtype=torch.bfloat16, device="cuda")
    route = RouterScratch.allocate(tokens, source.device, experts=EXPERTS, top_k=TOP_K)
    output = torch.empty_like(source)
    workspace = ScratchArena("cuda:0")
    with torch.inference_mode(), PreparationSession(device=source.device, autotune=True,
                                                    compile_workers=0) as session:
        routing = PreparedRouter(logits, route, session=session, name=f"sm110_route:{tokens}")
        prepared = PreparedRoutedExperts(source, view, route.ids, route.weights, output,
                                         session=session, name=f"sm110_experts:{tokens}", workspace=workspace)
        assert isinstance(prepared, Sm110RoutedExperts)
        session.freeze()
        workspace.finalize()
        workspace.freeze()
        assert prepared.plan.selection.source in ("tuned", "cached", "fixed")

        def forward():
            routing(logits)
            return prepared()

        run = compile_forward(forward) if compiled else forward
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        generator = torch.Generator(device="cuda").manual_seed(1103 + tokens)
        for iteration in range(2):
            cpu_source = torch.randn((tokens, HIDDEN), generator=generator, device="cuda").to(torch.bfloat16)
            source.copy_(cpu_source)
            logits.copy_(torch.randn((tokens, EXPERTS), generator=generator, device="cuda").to(torch.bfloat16))
            output.fill_(float("nan"))
            workspace.tensor.fill_(0x7F)
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            ids, probabilities = route.ids, route.weights
            # The component quantizes activations dynamically; dequantize the same
            # pack to separate the weight adapter from FP4 activation loss.
            packed, scales = fp4_quantize(cpu_source, torch.ones(1, device="cuda"),
                                          sf_vec_size=16, is_sf_swizzled_layout=False)
            quantized = _decode_nvfp4(packed, scales.view(torch.uint8).view(tokens, HIDDEN // 16), 1.0)
            adapter = _metrics(output, _expect(decoded, ids, probabilities, quantized))
            total = _metrics(output, _expect(decoded, ids, probabilities, cpu_source))
            assert adapter["relative_l2_error"] < ADAPTER_BOUND, adapter
            assert total["relative_l2_error"] < TOTAL_BOUND, total
            print(json.dumps({"component": "cutlass_nvfp4", "tokens": tokens, "compiled": compiled,
                              "iteration": iteration, "implementation": prepared.binding.implementation,
                              "adapter": adapter, "total": total}), flush=True)


@pytest.mark.parametrize("tokens", [1, 4, 24, 512])
@pytest.mark.parametrize("compiled", [False, True])
def test_nvfp4_experts_use_cutlass(experts, tokens, compiled):
    view, decoded = experts
    _run_component(view, decoded, tokens, compiled=compiled)

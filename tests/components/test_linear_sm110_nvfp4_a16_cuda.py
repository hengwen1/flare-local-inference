"""16-bit-activation NVFP4 dense projections on SM110 through cuDNN's W4A16 GEMM.

Qwen3.8's vision tower stores its MLP and merger projections as NVFP4 weights
with 16-bit activations (``W4A16_NVFP4``), so those call sites pass
``activation_mode="a16"`` and need a mixed GEMM rather than dynamic activation
quantization.

Weights are synthetic in the checkpoint layout: packed E2M1 payloads, one E4M3
scale per 16 inputs in the 128x4 grid, and one float32 dequantization scalar.
The reference decodes exactly those tensors, which isolates weight-adapter
correctness from the kernel's own arithmetic. Cases cover Qwen3.8 vision MLP and
merger geometry, compiled forwards, CUDA graph capture, allocation-stable
replay, and the folded bias.
"""

from __future__ import annotations

import json
import os

import pytest
import torch
import torch.nn.functional as F
from dataclasses import replace

from b12x.gemm.blockscaled import NVFP4LinearWeight
from b12x.preparation import PreparationSession

from ignition.components import linear
from ignition.preparation.compilation import compile_forward

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an explicitly assigned CUDA_VISIBLE_DEVICES",
)

GEOMETRY = ((1152, 4304), (4608, 2560))
WEIGHT_STD = 0.02
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# Kernel arithmetic against the exact decode of the same tensors, measured on
# Thor at 0.002303-0.002367 across every case below.
ERROR_BOUND = 0.003


def _quantize_checkpoint_style(weight):
    """Pack a BF16 weight as the release stores it: E2M1 codes, per-16 E4M3 blocks."""
    dequant_scale = weight.abs().max().to(torch.float32) / 448.0
    blocks = (weight.float() / dequant_scale).view(weight.shape[0], -1, 16)
    block_scale = (blocks.abs().amax(-1) / 6.0).clamp_min(1e-12)
    block_scale = block_scale.to(torch.float8_e4m3fn).float()
    codes = (blocks / block_scale.unsqueeze(-1)).clamp(-6, 6).reshape(-1)
    table = E2M1.to(weight.device)
    nearest = (codes.abs().unsqueeze(-1) - table.unsqueeze(0)).abs().argmin(-1)
    code = (nearest + torch.where(codes < 0, 8, 0)).to(torch.uint8).view(-1, 2)
    packed = (code[:, 0] | (code[:, 1] << 4)).view(weight.shape[0], -1)
    return packed, block_scale.to(torch.float8_e4m3fn).view(torch.uint8), dequant_scale


def _decode(packed, scales, dequant_scale):
    """Exact decode of packed E2M1 codes with per-16 E4M3 block scales."""
    low = (packed & 0x0F).to(torch.long)
    high = ((packed >> 4) & 0x0F).to(torch.long)
    table = E2M1.to(packed.device)
    values = torch.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), device=packed.device,
                         dtype=torch.float32)
    values[..., 0::2] = torch.where(low >= 8, -table[low & 7], table[low & 7])
    values[..., 1::2] = torch.where(high >= 8, -table[high & 7], table[high & 7])
    blocks = scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    return values * blocks * dequant_scale


@pytest.fixture(scope="module", params=GEOMETRY, ids=[f"{i}x{o}" for i, o in GEOMETRY])
def weight(request):
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.skip("assign a CUDA device explicitly")
    if torch.cuda.get_device_capability(0) != (11, 0):
        pytest.skip("SM110 dense projections require a Jetson Thor")
    in_features, out_features = request.param
    generator = torch.Generator(device="cuda").manual_seed(110 + in_features)
    source = torch.randn((out_features, in_features), generator=generator, device="cuda") * WEIGHT_STD
    packed, scales, dequant_scale = _quantize_checkpoint_style(source)
    # The loader packs checkpoint storage exactly this way.
    from b12x._lib.intrinsics.reference import swizzle_block_scale
    from b12x.gemm.blockscaled import pack_weight

    prepared = pack_weight(packed, swizzle_block_scale(scales.view(out_features, in_features // 16)),
                           recipe="nvfp4", global_scale=dequant_scale.reshape(1),
                           global_scale_kind="multiplier")
    return prepared, _decode(packed, scales, dequant_scale), in_features, out_features


def _metrics(actual, expected):
    difference = actual.float().cpu() - expected.float().cpu()
    return {"max_absolute_error": difference.abs().max().item(),
            "relative_l2_error": (difference.norm() / expected.float().norm().clamp_min(1e-12)).item()}


def _run_component(weight, tokens, *, compiled):
    prepared, decoded, in_features, out_features = weight
    source = torch.zeros((tokens, in_features), dtype=torch.bfloat16, device="cuda")
    output = torch.empty((tokens, out_features), dtype=torch.bfloat16, device="cuda")
    bias = (torch.randn((out_features,), device="cuda") * WEIGHT_STD).to(torch.bfloat16)
    with torch.inference_mode(), PreparationSession(device=source.device, autotune=True,
                                                    compile_workers=0) as session:
        projection = linear.QuantizedProjection(source, prepared, output, session=session,
                                                name=f"sm110_a16:{tokens}", activation_mode="a16",
                                                bias=bias)
        assert projection.sm110 is not None
        session.freeze()

        run = compile_forward(lambda: projection(source)) if compiled else (lambda: projection(source))
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        generator = torch.Generator(device="cuda").manual_seed(1103 + tokens)
        for iteration in range(2):
            activations = (torch.randn((tokens, in_features), generator=generator, device="cuda") * 0.5)
            activations = activations.to(torch.bfloat16)
            source.copy_(activations)
            output.fill_(float("nan"))
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            expected = activations.float() @ decoded.T + bias.float()
            metrics = _metrics(output, expected)
            assert metrics["relative_l2_error"] < ERROR_BOUND, metrics
            print(json.dumps({"component": "cudnn_bf16_fp4", "tokens": tokens,
                              "in_features": in_features, "out_features": out_features,
                              "compiled": compiled, "iteration": iteration,
                              "implementation": projection.sm110.binding.implementation,
                              **metrics}), flush=True)


@pytest.mark.parametrize("tokens", [4, 16, 128, 512])
@pytest.mark.parametrize("compiled", [False, True])
def test_nvfp4_a16_uses_cudnn(weight, tokens, compiled):
    _run_component(weight, tokens, compiled=compiled)


def test_padded_input_width_is_trimmed():
    """The release pads the stored input width to 128; the extra columns are zero."""
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.skip("assign a CUDA device explicitly")
    if torch.cuda.get_device_capability(0) != (11, 0):
        pytest.skip("SM110 dense projections require a Jetson Thor")
    from b12x._lib.intrinsics.reference import swizzle_block_scale
    from b12x.gemm.blockscaled import pack_weight

    in_features, out_features = 4304, 256
    stored = (in_features + 127) // 128 * 128
    generator = torch.Generator(device="cuda").manual_seed(4304)
    dense = torch.randn((out_features, in_features), generator=generator, device="cuda") * WEIGHT_STD
    packed, scales, dequant_scale = _quantize_checkpoint_style(dense)
    padded_values = torch.zeros((out_features, stored // 2), dtype=torch.uint8, device="cuda")
    padded_values[:, : in_features // 2] = packed
    padded_scales = torch.zeros((out_features, stored // 16), dtype=torch.float8_e4m3fn, device="cuda")
    padded_scales[:, : in_features // 16] = scales.view(out_features, in_features // 16).view(torch.float8_e4m3fn)
    packed_weight = pack_weight(padded_values, swizzle_block_scale(padded_scales), recipe="nvfp4",
                                global_scale=dequant_scale.reshape(1), global_scale_kind="multiplier")
    # The loader records the logical width after packing, as it does for the release.
    prepared = replace(packed_weight, in_features=in_features)
    assert prepared.padded_in_features == stored > prepared.in_features == in_features

    source = torch.zeros((4, in_features), dtype=torch.bfloat16, device="cuda")
    output = torch.empty((4, out_features), dtype=torch.bfloat16, device="cuda")
    with torch.inference_mode(), PreparationSession(device=source.device, autotune=True,
                                                    compile_workers=0) as session:
        projection = linear.QuantizedProjection(source, prepared, output, session=session,
                                                name="sm110_a16_padded", activation_mode="a16")
        session.freeze()
        activations = torch.randn((4, in_features), generator=generator, device="cuda").to(torch.bfloat16)
        source.copy_(activations)
        projection(source)
        torch.cuda.synchronize()
        expected = activations.float() @ _decode(packed, scales, dequant_scale).T
        metrics = _metrics(output, expected)
        assert metrics["relative_l2_error"] < ERROR_BOUND, metrics


def test_mxfp8_weights_reject_a16(weight):
    """A16 selects the FP4 mixed kernel; MXFP8 weights have no such path."""
    _, _, in_features, out_features = weight
    from b12x.gemm.blockscaled._linear import MXFP8LinearWeight
    from b12x.gemm._shared.wo_mxfp8 import MXFP8Rows
    from flashinfer.quantization import mxfp8_quantize

    generator = torch.Generator(device="cuda").manual_seed(7)
    dense = torch.randn((out_features, in_features), generator=generator, device="cuda") * WEIGHT_STD
    values, scales = mxfp8_quantize(dense.to(torch.bfloat16), is_sf_swizzled_layout=False)
    prepared = MXFP8LinearWeight(
        weight=MXFP8Rows(values=values,
                         scale_rows=scales.view(out_features, in_features // 32).view(torch.float8_e8m0fnu).unsqueeze(0),
                         scale_mma=None),
        in_features=in_features, padded_in_features=in_features, out_features=out_features)
    source = torch.zeros((4, in_features), dtype=torch.bfloat16, device="cuda")
    output = torch.empty((4, out_features), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="NVFP4"):
        with torch.inference_mode(), PreparationSession(device=source.device, autotune=True,
                                                        compile_workers=0) as session:
            linear.QuantizedProjection(source, prepared, output, session=session,
                                       name="sm110_a16_mxfp8", activation_mode="a16")
            session.freeze()

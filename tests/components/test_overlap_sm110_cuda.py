"""Stream overlap is rejected where b12x's parallel block-scaled GEMMs are absent.

Both overlap paths fork prepared projections onto an auxiliary stream through
b12x block-scaled parallel GEMMs, which report no support on SM110. Recipes must
disable stream overlap there instead of capturing a graph whose fork cannot run.
"""

from __future__ import annotations

import os

import pytest
import torch

from ignition.components import linear
from ignition.components.expert_overlap import ConcurrentSharedExpert
from ignition.preparation.scratch import ScratchArena

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an explicitly assigned CUDA_VISIBLE_DEVICES",
)


@pytest.fixture(scope="module")
def device():
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.skip("assign a CUDA device explicitly")
    if torch.cuda.get_device_capability(0) != (11, 0):
        pytest.skip("this rejection is the SM110 path")
    return torch.device("cuda", 0)


def test_parallel_projections_report_no_support(device):
    assert linear.parallel_projection_supported(device) is False


def test_shared_expert_overlap_is_rejected(device):
    workspace = ScratchArena(device)
    with pytest.raises(ValueError, match="disable stream overlap"):
        ConcurrentSharedExpert(object(), parallel_plans=[1], session=None, workspace=workspace)


def test_attention_overlap_is_rejected(device):
    from ignition.recipes.qwen38.attention_overlap import ConcurrentAttention

    workspace = ScratchArena(device)
    with pytest.raises(ValueError, match="disable stream overlap"):
        ConcurrentAttention(object(), session=None, workspace=workspace)

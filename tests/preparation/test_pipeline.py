"""Pipeline preparation metadata and stage-range validation."""

import pytest

from ignition.recipes.qwen38.bootstrap import PreparationConfig


def test_pipeline_configuration_requires_complete_stage_boundaries():
    config = PreparationConfig(pipeline_stages=2, stage_layers=(0, 24, 48),
                               devices=(0, 0), hosts=("first", "second"))
    assert config.stage_layers == (0, 24, 48)

    with pytest.raises(ValueError, match="stage layers"):
        PreparationConfig(pipeline_stages=2, devices=(0, 0), hosts=("first", "second"))
    with pytest.raises(ValueError, match="stage layers"):
        PreparationConfig(pipeline_stages=2, stage_layers=(0, 48, 48),
                          devices=(0, 0), hosts=("first", "second"))


def test_pipeline_configuration_rejects_unsupported_combinations():
    with pytest.raises(ValueError, match="MTP"):
        PreparationConfig(pipeline_stages=2, stage_layers=(0, 24, 48), mtp=True,
                          devices=(0, 0), hosts=("first", "second"))
    with pytest.raises(ValueError, match="activation slots"):
        PreparationConfig(pipeline_stages=2, stage_layers=(0, 24, 48),
                          pipeline_activation_slots=0, devices=(0, 0), hosts=("first", "second"))


def test_pipeline_contract_keeps_vision_and_head_on_opposite_stages():
    from ignition.recipes.qwen38.bootstrap import _pipeline_model_contract
    from ignition.preparation import runtime_package_pb2 as p

    base = {
        "vocabulary": 128, "input_vocabulary": 128,
        "vision": {"feature_slots": 1}, "frontend": {},
        "cache": {"page_tokens": 64, "regions": [], "page_tables": []},
        "graphs": [{"graph": "vision256", "kind": p.VISION},
                   {"graph": "decode1", "kind": p.DECODE}],
    }
    config = PreparationConfig(pipeline_stages=2, stage_layers=(0, 24, 48),
                               devices=(0, 0), hosts=("first", "second"))
    contract = _pipeline_model_contract(base, config, 8, 4, 2)
    assert contract["version"] == 3
    stages = contract["pipeline"]["stage"]
    assert stages[0]["first_layer"] == 0 and stages[0]["last_layer"] == 24
    assert stages[1]["first_layer"] == 24 and stages[1]["last_layer"] == 48
    assert stages[0]["has_vision"] and not stages[1]["has_vision"]
    assert not stages[0]["has_head"] and stages[1]["has_head"]
    assert [graph["graph"] for graph in stages[1]["graphs"]] == ["decode1"]
    assert stages[1]["graphs"][0]["inputs"] == [
        {"buffer": "pipeline_activation", "source": p.CONSTANT,
         "indexing": p.SCALAR, "value_offset": 0, "padding": 0, "tensor": ""}]
    assert contract["pipeline"]["boundary"][0]["buffer"] == "pipeline_activation"

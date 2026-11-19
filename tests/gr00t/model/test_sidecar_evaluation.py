"""CPU-only tests of reversible evaluation diagnostics."""

import json

from gr00t.experiment.sidecar_evaluation import (
    export,
    gate_rows,
    intervention,
    offline_rows,
    replay_rng,
    residual_metrics,
    sensitivity,
)
from gr00t.model.extension.point_action_sidecar import PointActionStage
from gr00t.model.extension.sidecar_dit import SidecarConditionedDiT
import numpy as np
import pytest
import torch


def tiny(gate_mode="scalar"):
    return SidecarConditionedDiT(
        num_attention_heads=2,
        attention_head_dim=4,
        output_dim=8,
        num_layers=2,
        dropout=0.0,
        final_dropout=False,
        positional_embeddings=None,
        interleave_self_attention=True,
        cross_attention_dim=6,
        sidecar_hidden_dim=8,
        sidecar_num_heads=2,
        sidecar_num_point_tokens=3,
        contact_hidden_dim=8,
        contact_num_tokens=2,
        tactile_fusion_mode="none",
        sidecar_gate_mode=gate_mode,
    ).eval()


def inputs():
    return dict(
        hidden_states=torch.randn(1, 4, 8),
        encoder_hidden_states=torch.randn(1, 6, 6),
        timestep=torch.tensor([3]),
        image_mask=torch.tensor([[0, 0, 0, 1, 1, 1]]).bool(),
        backbone_attention_mask=torch.ones(1, 6).bool(),
        point_tokens=torch.randn(1, 7, 8),
        point_attention_mask=torch.ones(1, 7).bool(),
        hidden_attention_mask=torch.tensor([[1, 1, 1, 0]]).bool(),
    )


def test_zero_gate():
    stage = PointActionStage(8, 8, 2)
    action, points = torch.randn(1, 3, 8), torch.randn(1, 4, 8)
    rows = []
    result, _ = stage(action, points, torch.ones(1, 4).bool(), observer=lambda *x: rows.append(x))
    assert torch.equal(result, action)
    assert torch.count_nonzero(rows[-1][-1]) == 0


def test_context_skip_restore_mapping_and_parity():
    model, x = tiny(), inputs()
    with torch.no_grad():
        for stage in model.sidecar_stages:
            stage.action_gate.fill_(0.4)
    baseline = model(**x)
    rows = []
    with intervention(model, rows=rows):
        torch.testing.assert_close(model(**x), baseline, rtol=0, atol=0)
    assert rows and all(np.isfinite(r["cosine_similarity"]) for r in rows)
    rows = []
    with intervention(model, disabled=True, rows=rows):
        disabled = model(**x)
    assert not rows
    direct = dict(x, point_tokens=None, point_attention_mask=None)
    torch.testing.assert_close(disabled, model(**direct), rtol=0, atol=0)
    before = [stage.action_gate.clone() for stage in model.sidecar_stages]
    with pytest.raises(RuntimeError), intervention(model, zero_stages=[0]):
        assert model.sidecar_stages[0].action_gate == 0
        raise RuntimeError("failure")
    for value, stage in zip(before, model.sidecar_stages):
        assert torch.equal(value, stage.action_gate)
    assert [r["block_index"] for r in gate_rows(model)] == model.stage_indices
    assert [r["block_number"] for r in gate_rows(model)] == [i + 1 for i in model.stage_indices]
    torch.testing.assert_close(model(**x), baseline, rtol=0, atol=0)


def test_adaptive_stage_ablation_uses_effective_injection_and_restores():
    model, x = tiny("adaptive"), inputs()
    baseline = model(**x)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    rows = gate_rows(model)
    assert all(row["gate_mode"] == "adaptive" and row["action_gate"] is None for row in rows)
    with intervention(model, zero_stages=[0]):
        assert model.sidecar_stages[0]._action_injection_enabled is False
        changed = model(**x)
    assert not torch.equal(baseline, changed)
    assert model.sidecar_stages[0]._action_injection_enabled is True
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_rng_comparisons_and_caller_state():
    state = torch.random.get_rng_state()
    draws = []
    for mode in ["original", "disabled", "swapped"]:
        with replay_rng(17):
            draws.append(torch.randn(2, 3, 8))
    assert all(torch.equal(draws[0], x) for x in draws)
    assert torch.equal(state, torch.random.get_rng_state())


def test_masks_and_zero():
    a = torch.tensor([[[0.0, 0.0], [100.0, 100.0]]])
    mask = torch.tensor([[True, False]])
    metrics = residual_metrics(a, a, mask)
    assert metrics == dict(
        feature_norm=0.0, residual_norm=0.0, residual_ratio=0.0, cosine_similarity=0.0
    )
    result = sensitivity(np.zeros((1, 2, 2)), np.ones((1, 2, 2)), mask.numpy()[..., None])
    assert result["mae"] == result["rmse"] == 1
    assert result["per_timestep_mae"] == [1.0, 0.0]


def test_offline_export(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text(
        json.dumps(
            {
                "global_step": 3,
                "sidecar_action_gate_stage_0": 0.2,
                "sidecar_point_residual_ratio_stage_0": 0.1,
                "sidecar_action_gate_grad_stage_0": 10,
            }
        )
        + "\n"
    )
    rows = offline_rows(source)
    assert len(rows) == 2
    export(tmp_path / "out", {"geometry": "unavailable"}, [], rows)
    assert (tmp_path / "out/stage_gate.png").exists()
    assert (tmp_path / "out/stage_residual_ratio.png").exists()
    assert (tmp_path / "out/sample_metrics.csv").exists()
    assert json.loads((tmp_path / "out/summary.json").read_text())["geometry"] == "unavailable"


def test_actual_noise_probe_and_cleanup():
    from gr00t.experiment.sidecar_evaluation import initial_noise_probe

    encoder = torch.nn.Linear(3, 3)
    captures = []
    for _ in range(3):
        with replay_rng(31), initial_noise_probe(encoder) as noise:
            actual = torch.randn(1, 4, 3)
            encoder(actual)
            encoder(torch.randn(1, 4, 3))
        assert len(noise) == 1
        assert torch.equal(noise[0], actual)
        captures.append(noise[0])
    assert all(torch.equal(captures[0], x) for x in captures)
    assert not encoder._forward_pre_hooks

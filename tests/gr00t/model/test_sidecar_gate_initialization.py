"""CPU-only coverage for configurable Sidecar scalar gate initialization."""

from pathlib import Path

from gr00t.configs.hydra_config import build_gr00t_config
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.configs.sidecar_config import validate_checkpoint_variant, validate_gate_init
from gr00t.model.extension.multilevel_local_sidecar import LocalJointActionStage
from gr00t.model.extension.point_action_sidecar import PointActionStage
from hydra import compose, initialize_config_dir
import pytest
import torch


def _init(mode, **values):
    return {"mode": mode, **values}


@pytest.mark.parametrize(
    ("mode", "settings", "expected"),
    [
        ("zero", {}, 0.0),
        ("one", {}, 1.0),
        ("constant", {"value": 0.02}, 0.02),
    ],
)
def test_exact_gate_initializers(mode, settings, expected):
    stage = PointActionStage(8, 8, 2, action_gate_init=_init(mode, **settings), gate_init_seed=42)
    torch.testing.assert_close(stage.action_gate.detach(), torch.tensor(expected), rtol=0, atol=0)
    assert stage.action_gate.requires_grad


def test_random_gate_initializers_are_bounded_reproducible_and_stage_independent():
    uniform = _init("uniform", low=0.01, high=0.05)
    first = [
        PointActionStage(
            8, 8, 2, action_gate_init=uniform, gate_init_seed=17, stage_index=index
        ).action_gate.item()
        for index in range(4)
    ]
    second = [
        PointActionStage(
            8, 8, 2, action_gate_init=uniform, gate_init_seed=17, stage_index=index
        ).action_gate.item()
        for index in range(4)
    ]
    assert first == second
    assert all(0.01 <= value < 0.05 for value in first)
    assert len(set(first)) == len(first)

    normal = _init("normal", mean=0.02, std=0.01)
    normal_values = [
        PointActionStage(
            8, 8, 2, action_gate_init=normal, gate_init_seed=17, stage_index=index
        ).action_gate.item()
        for index in range(2)
    ]
    assert torch.isfinite(torch.tensor(normal_values)).all()
    assert normal_values[0] != normal_values[1]


def test_random_gate_sampling_does_not_change_other_parameter_rng_order():
    torch.manual_seed(123)
    zero = PointActionStage(8, 8, 2, action_gate_init=_init("zero"))
    torch.manual_seed(123)
    random = PointActionStage(8, 8, 2, action_gate_init=_init("uniform", low=0.01, high=0.05))
    for name, value in zero.state_dict().items():
        if not name.endswith("gate"):
            torch.testing.assert_close(value, random.state_dict()[name], rtol=0, atol=0)


def test_positive_action_gate_unblocks_action_branch_gradient():
    def gradients(value):
        torch.manual_seed(9)
        stage = PointActionStage(
            8,
            8,
            2,
            action_gate_init=_init("constant", value=value),
        )
        action = torch.randn(2, 3, 8)
        points = torch.randn(2, 4, 8)
        mask = torch.ones(2, 4, dtype=torch.bool)
        stage(action, points, mask)[0].square().mean().backward()
        return stage.output_projection.weight.grad, stage.action_gate.grad

    zero_branch, zero_gate = gradients(0.0)
    positive_branch, positive_gate = gradients(0.02)
    assert zero_branch is not None and zero_branch.count_nonzero() == 0
    assert zero_gate is not None and zero_gate.abs() > 0
    assert positive_branch is not None and positive_branch.abs().sum() > 0
    assert positive_gate is not None and positive_gate.abs() > 0


def test_bf16_diagnostic_distinguishes_gated_from_applied_residual():
    stage = PointActionStage(
        8,
        8,
        2,
        action_gate_init=_init("constant", value=0.02),
    )
    action = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    points = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    mask = torch.ones(2, 4, dtype=torch.bool)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        stage(action, points, mask, observer=lambda *args: None)
    diagnostics = stage.last_diagnostics
    assert diagnostics["action_pre_gate_update_norm"] > 0
    assert diagnostics["action_gated_residual_norm"] > 0
    assert diagnostics["action_actual_residual_norm"] >= 0
    assert diagnostics["action_bf16_loss_norm"] is not None
    assert diagnostics["action_bf16_loss_norm"] >= 0


def test_checkpoint_state_wins_over_constructor_initialization():
    trained = PointActionStage(
        8, 8, 2, action_gate_init=_init("constant", value=0.02), gate_init_seed=1
    )
    with torch.no_grad():
        trained.action_gate.fill_(0.375)
        trained.point_gate.fill_(0.625)
    restored = PointActionStage(
        8,
        8,
        2,
        point_gate_init=_init("zero"),
        action_gate_init=_init("one"),
        gate_init_seed=999,
    )
    restored.load_state_dict(trained.state_dict(), strict=True)
    assert restored.action_gate.item() == pytest.approx(0.375)
    assert restored.point_gate.item() == pytest.approx(0.625)


def test_multilevel_local_action_initialization_and_no_point_gate():
    stage = LocalJointActionStage(
        8,
        6,
        8,
        2,
        action_gate_init=_init("constant", value=0.02),
        gate_init_seed=4,
        stage_index=2,
    )
    assert stage.action_gate.item() == pytest.approx(0.02)
    assert not hasattr(stage, "point_gate")


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"mode": "bad"}, "mode"),
        ({"mode": "constant", "value": float("nan")}, "finite"),
        ({"mode": "uniform", "low": 1.0, "high": 1.0}, "less than"),
        ({"mode": "normal", "mean": 0.0, "std": 0.0}, "positive"),
    ],
)
def test_invalid_gate_initialization_is_rejected(settings, message):
    with pytest.raises(ValueError, match=message):
        validate_gate_init(
            settings,
            default=lambda: _init("zero", value=0, low=0, high=1, mean=0, std=1),
            name="gate",
        )


@pytest.mark.parametrize("seed", [-1, True, 2**63])
def test_invalid_gate_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="seed"):
        PointActionStage(8, 8, 2, gate_init_seed=seed)


def test_hf_config_and_hydra_override_roundtrip(tmp_path):
    config = Gr00tN1d7Config(
        sidecar_action_gate_init=_init(
            "uniform", value=0.02, low=0.01, high=0.05, mean=0.02, std=0.01
        ),
        sidecar_gate_init_seed=73,
        sidecar_gate_mode="adaptive",
        sidecar_adaptive_gate_hidden_dim=96,
    )
    config.save_pretrained(tmp_path)
    restored = Gr00tN1d7Config.from_pretrained(tmp_path)
    assert restored.sidecar_action_gate_init["mode"] == "uniform"
    assert restored.sidecar_gate_init_seed == 73
    assert restored.sidecar_gate_mode == "adaptive"
    assert restored.sidecar_adaptive_gate_hidden_dim == 96

    root = Path(__file__).resolve().parents[3]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.3"):
        hydra = compose(
            config_name="experiment/wah_multilevel_local_frozen",
            overrides=[
                "model.sidecar_action_gate_init.mode=constant",
                "model.sidecar_action_gate_init.value=0.02",
                "model.sidecar_gate_init_seed=73",
            ],
        )
    built = build_gr00t_config(hydra)
    assert built.model.sidecar_action_gate_init["mode"] == "constant"
    assert built.model.sidecar_action_gate_init["value"] == pytest.approx(0.02)
    assert built.model.sidecar_gate_init_seed == 73


def test_multilevel_hydra_rejects_nondefault_point_gate():
    root = Path(__file__).resolve().parents[3]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.3"):
        hydra = compose(
            config_name="experiment/wah_multilevel_local_frozen",
            overrides=["model.sidecar_point_gate_init.mode=zero"],
        )
    with pytest.raises(ValueError, match="has no point gate"):
        build_gr00t_config(hydra)

    with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.3"):
        hydra = compose(
            config_name="experiment/wah_multilevel_local_frozen",
            overrides=["model.sidecar_adaptive_gate_init_point=0.7"],
        )
    with pytest.raises(ValueError, match="has no point gate"):
        build_gr00t_config(hydra)


def test_adaptive_gate_shape_range_mask_sensitivity_and_gradients():
    torch.manual_seed(11)
    stage = PointActionStage(
        8,
        8,
        2,
        gate_mode="adaptive",
        adaptive_gate_hidden_dim=12,
        adaptive_gate_init_action=0.02,
        adaptive_gate_init_point=0.8,
    )
    action = torch.randn(2, 3, 8, requires_grad=True)
    points = torch.randn(2, 5, 8, requires_grad=True)
    point_mask = torch.tensor([[True, True, False, False, False], [True] * 5])
    action_mask = torch.tensor([[True, True, False], [True] * 3])
    output, _ = stage(action, points, point_mask, action_mask)
    action_gate = stage.last_gate_values["action"]
    point_gate = stage.last_gate_values["point"]
    assert action_gate.shape == point_gate.shape == (2, 1, 1)
    assert ((0 < action_gate) & (action_gate < 1)).all()
    assert ((0 < point_gate) & (point_gate < 1)).all()
    assert action_gate[0].item() != action_gate[1].item()
    output.square().mean().backward()
    assert stage.action_gate_network.output.weight.grad.abs().sum() > 0
    assert stage.output_projection.weight.grad.abs().sum() > 0
    assert stage.action_reads_point.attention.in_proj_weight.grad.abs().sum() > 0

    changed = points.detach().clone()
    changed[0, 2:] = 10_000
    baseline = stage(action.detach(), points.detach(), point_mask, action_mask)[0]
    masked = stage(action.detach(), changed, point_mask, action_mask)[0]
    torch.testing.assert_close(baseline, masked, rtol=0, atol=0)


def test_adaptive_gate_seed_reproducibility_and_state_dict_roundtrip():
    def make(seed, stage):
        torch.manual_seed(123)
        return PointActionStage(
            8,
            8,
            2,
            gate_mode="adaptive",
            gate_init_seed=seed,
            stage_index=stage,
        )

    first, same, other_stage = make(9, 0), make(9, 0), make(9, 1)
    torch.testing.assert_close(
        first.action_gate_network.output.weight,
        same.action_gate_network.output.weight,
        rtol=0,
        atol=0,
    )
    assert not torch.equal(
        first.action_gate_network.output.weight,
        other_stage.action_gate_network.output.weight,
    )
    with torch.no_grad():
        first.action_gate_network.output.bias.fill_(-1.25)
    restored = make(999, 3)
    restored.load_state_dict(first.state_dict(), strict=True)
    torch.testing.assert_close(
        restored.action_gate_network.output.bias,
        torch.full_like(restored.action_gate_network.output.bias, -1.25),
    )


def test_adaptive_gate_network_uses_an_isolated_rng_stream():
    from gr00t.model.extension.point_action_sidecar import AdaptiveGate

    torch.manual_seed(1234)
    expected = torch.randn(6)
    torch.manual_seed(1234)
    AdaptiveGate(8, 12, 0.02, seed=7, stage=1, kind="action")
    actual = torch.randn(6)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_adaptive_effective_disable_is_noop_and_restores():
    stage = PointActionStage(8, 8, 2, gate_mode="adaptive")
    action = torch.randn(2, 3, 8)
    points = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4, dtype=torch.bool)
    assert not torch.equal(stage(action, points, mask)[0], action)
    stage.set_injection_enabled("action", False)
    torch.testing.assert_close(stage(action, points, mask)[0], action, rtol=0, atol=0)
    stage.set_injection_enabled("action", True)
    assert not torch.equal(stage(action, points, mask)[0], action)


def test_multilevel_adaptive_action_gate_and_temporal_context():
    stage = LocalJointActionStage(
        8,
        6,
        8,
        2,
        patch_size=3,
        gate_mode="adaptive",
        temporal_context_dim=8,
    )
    action = torch.randn(2, 3, 8, requires_grad=True)
    packed = {
        "feat": torch.randn(7, 6, requires_grad=True),
        "coord": torch.randn(7, 3),
        "offset": torch.tensor([4, 7]),
        "serialized_order": torch.arange(7)[None],
        "temporal_context": torch.randn(2, 5, 8, requires_grad=True),
        "temporal_context_mask": torch.tensor([[False, True, True, True, True], [True] * 5]),
    }
    output, _ = stage(action, packed, action_mask=torch.ones(2, 3, dtype=torch.bool))
    assert stage.last_gate_values["action"].shape == (2, 1, 1)
    output.square().mean().backward()
    assert stage.action_gate_network.output.weight.grad.abs().sum() > 0
    assert stage.output_projection.weight.grad.abs().sum() > 0
    assert packed["temporal_context"].grad.abs().sum() > 0
    assert not hasattr(stage, "point_gate") and not hasattr(stage, "point_gate_network")


def test_checkpoint_gate_mode_mismatch_is_explicit():
    from types import SimpleNamespace

    common = dict(dit_type="sidecar_conditioned_dit", sidecar_variant="legacy")
    checkpoint = SimpleNamespace(**common, sidecar_gate_mode="scalar")
    requested = SimpleNamespace(**common, sidecar_gate_mode="adaptive")
    with pytest.raises(ValueError, match="conversion must be explicit"):
        validate_checkpoint_variant(checkpoint, requested)


@pytest.mark.parametrize(
    ("name", "variant", "temporal", "adaptive"),
    [
        ("point_adaptive_sidecar_ditfull", "legacy", 0, True),
        ("temporal_tactile_multilevel_ditfull", "multilevel_local", 5, False),
        ("temporal_tactile_adaptive_sidecar_ditfull", "legacy", 5, True),
        ("point_adaptive_multilevel_ditfull", "multilevel_local", 0, True),
        ("temporal_tactile_adaptive_multilevel_ditfull", "multilevel_local", 5, True),
    ],
)
def test_adaptive_experiment_hydra_resolution(name, variant, temporal, adaptive):
    root = Path(__file__).resolve().parents[3]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.3"):
        hydra = compose(config_name=f"experiment/{name}")
    built = build_gr00t_config(hydra)
    assert built.model.sidecar_variant == variant
    assert built.model.temporal_window_size == temporal
    assert built.model.sidecar_gate_mode == ("adaptive" if adaptive else "scalar")
    # The checkpoint-owned model capacity remains 40; the training data horizon is 16.
    assert hydra.data.modalities.action.horizon == 16
    assert built.model.state_dropout_prob == pytest.approx(0.2)
    assert built.model.tune_projector is True
    assert built.model.dit_training_mode == "full"
    assert built.model.use_tactile_conditioning is bool(temporal)
    if temporal:
        assert hydra.data.modalities.tactile.delta_indices == [-5, -4, -3, -2, -1, 0]

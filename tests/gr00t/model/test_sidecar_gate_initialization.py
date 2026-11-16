"""CPU-only coverage for configurable Sidecar scalar gate initialization."""

from pathlib import Path

from gr00t.configs.hydra_config import build_gr00t_config
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.configs.sidecar_config import validate_gate_init
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
    )
    config.save_pretrained(tmp_path)
    restored = Gr00tN1d7Config.from_pretrained(tmp_path)
    assert restored.sidecar_action_gate_init["mode"] == "uniform"
    assert restored.sidecar_gate_init_seed == 73

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

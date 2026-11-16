"""CPU diagnostics contracts with tiny mocked models; never load pretrained weights."""

from copy import deepcopy
import csv
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
from gr00t.experiment.wah_diagnostics import (
    cartesian_response,
    interrupted_by_sigterm,
    load_scene_metadata,
    main,
    parser,
    validate_args,
)
from gr00t.experiment.wah_grounding import (
    compatible_donor,
    geometry_metrics,
    masked_action_metrics,
    noise_identity,
    observed_residual,
    paired_inference,
    sidecar_intervention,
    stage_mapping,
)
from gr00t.experiment.wah_reporting import (
    DiagnosticDashboard,
    DiagnosticReport,
    DiagnosticWandb,
    aggregate_stages,
)
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn


class Pool(nn.Module):
    def forward(self, tokens, mask=None):
        return tokens, mask


class Stage(nn.Module):
    def __init__(self, local=False):
        super().__init__()
        self.action_gate = nn.Parameter(torch.tensor(0.2))
        if not local:
            self.point_gate = nn.Parameter(torch.tensor(0.7))

    def forward(
        self, action, points, point_mask=None, action_mask=None, observer=None, stage=0, **kwargs
    ):
        values = points["feat"] if isinstance(points, dict) else points
        if hasattr(self, "point_gate"):
            delta = self.point_gate * action.mean() * torch.ones_like(values)
            if observer:
                observer("sidecar_point", stage, values, delta)
            points = values + delta
            values = points
        delta = self.action_gate * values.mean() * torch.ones_like(action)
        if action_mask is not None:
            delta = delta * action_mask[..., None]
        if observer:
            observer("sidecar_action", stage, action, delta)
        return action + delta, points


class Dit(nn.Module):
    def __init__(self, local=False):
        super().__init__()
        self.sidecar_variant = "multilevel_local" if local else "legacy"
        self.sidecar_point_stage_names = ["dec1", "dec0"] if local else []
        self.stage_indices = [1, 4]
        self.transformer_blocks = nn.ModuleList([nn.Identity() for _ in range(5)])
        self.sidecar_stages = nn.ModuleList([Stage(local), Stage(local)])
        self.sidecar_pool = Pool()
        self.tactile_fusion_mode = "surprise"
        self.contact_inputs = []

    def forward(self, action, points, point_mask, tactile, state, action_mask):
        # Simulates point-dependent ContactSurprise and independent input MoE.
        raw = points["dec1"]["feat"] if isinstance(points, dict) else points
        self.contact_inputs.append((raw.detach().clone(), tactile.clone(), state.clone()))
        action = action + raw.mean() * 0.03 + tactile.mean() * 0.02 + state.mean() * 0.01
        if self.sidecar_variant == "legacy":
            points, point_mask = self.sidecar_pool(points, point_mask)
        for index, stage in enumerate(self.sidecar_stages):
            level = (
                points[self.sidecar_point_stage_names[index]]
                if isinstance(points, dict)
                else points
            )
            action, updated = stage(
                action,
                level,
                point_mask,
                action_mask,
                getattr(self, "_diagnostic_observer", None),
                index,
            )
            if not isinstance(points, dict):
                points = updated
        return action * action_mask[..., None]


class Head(nn.Module):
    def __init__(self, local=False):
        super().__init__()
        self.model = Dit(local)
        self.action_encoder = nn.Identity()
        self.num_inference_timesteps = 2

    def _encode_multimodal_inputs(self, inputs):
        points = inputs["points"][..., :4]
        if self.model.sidecar_variant == "multilevel_local":
            points = {name: {"feat": points} for name in self.model.sidecar_point_stage_names}
            return points, None, inputs["tactile"]
        return points, torch.ones(points.shape[:2], dtype=torch.bool), inputs["tactile"]


class Model(nn.Module):
    def __init__(self, local=False, fail_at=None):
        super().__init__()
        self.action_head = Head(local)
        self.config = SimpleNamespace(contact_input_mode="moe", temporal_window_size=None)
        self.calls, self.fail_at = 0, fail_at

    def prepare_input(self, inputs):
        return {}, inputs

    def get_action(self, inputs):
        self.calls += 1
        head = self.action_head
        action = torch.randn(inputs["action_mask"].shape) * inputs["action_mask"]
        points, point_mask, tactile = head._encode_multimodal_inputs(inputs)
        for _ in range(head.num_inference_timesteps):
            action = head.action_encoder(action)
            action = head.model(
                action,
                points,
                point_mask,
                tactile,
                inputs["state"],
                inputs["action_mask"].bool().any(-1),
            )
            action = action * inputs["action_mask"]
            if self.calls == self.fail_at:
                raise KeyboardInterrupt()
        return {"action_pred": action}


class Processor:
    def eval(self):
        pass

    def __call__(self, messages):
        observation = messages[0]["content"]
        assert not observation.actions
        return dict(
            points=torch.from_numpy(observation.pointclouds["scene"][0].copy()),
            state=torch.from_numpy(observation.states["joint"].copy()),
            rgb=torch.from_numpy(observation.images["head"][0].copy()),
            language=torch.tensor([len(observation.text)]),
            tactile=torch.from_numpy(observation.tactile["rgb"][0].copy()).float(),
            action_mask=torch.tensor(
                [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
            ),
        )

    def decode_action(self, actions, embodiment, state):
        return {"joint": actions[:, :2, :2] * 2 + state["joint"]}


class Policy:
    def __init__(self, local=False, fail_at=None):
        self.model = Model(local, fail_at)
        self.processor = Processor()
        self.embodiment_tag = SimpleNamespace(value="new_embodiment")
        self.checkpoint_identity = "sha256:mock-checkpoint"

    def collate_fn(self, features):
        return {"inputs": {k: torch.stack([f[k] for f in features]) for k in features[0]}}

    def get_modality_config(self):
        return {
            "video": ModalityConfig([0], ["head"]),
            "language": ModalityConfig([0], ["task"]),
            "state": ModalityConfig([0], ["joint"]),
            "action": ModalityConfig([0, 1], ["joint"]),
            "pointcloud": ModalityConfig([0], ["scene"]),
            "tactile": ModalityConfig([0], ["rgb"]),
        }


class Loader:
    # Noncontiguous episode IDs ensure callers distinguish IDs from loader indices.
    episodes_metadata = [dict(episode_index=i, tasks=["lift the can"], length=4) for i in (3, 9)]

    def __getitem__(self, index):
        return pd.DataFrame(
            {
                "video.head": [np.zeros((2, 2, 3), dtype=np.uint8)] * 4,
                "state.joint": [np.array([0.2, 0.4], dtype=np.float32)] * 4,
                "language.task": ["lift the can"] * 4,
                "pointcloud.scene": [np.full((5, 6), 1 + index, dtype=np.float32)] * 4,
                "tactile.rgb": [np.full((2, 2, 3), 0.5, dtype=np.float32)] * 4,
            }
        )


@pytest.fixture
def cli_inputs(tmp_path, monkeypatch):
    # Keep the production preprocessing helper, but avoid loading transformers for mocks.
    monkeypatch.setitem(
        sys.modules,
        "gr00t.policy.gr00t_policy",
        SimpleNamespace(_prepare_policy_model_inputs=lambda inputs, model: inputs),
    )
    checkpoint, dataset = tmp_path / "checkpoint", tmp_path / "dataset"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"mock-not-real-weights")
    (dataset / "meta").mkdir(parents=True)
    for name in ("info.json", "episodes.jsonl", "tasks.jsonl", "modality.json", "stats.json"):
        (dataset / "meta" / name).write_text("{}")
    metadata = {
        str(i): dict(
            task="lift_can",
            scene=f"scene-{i}",
            sensor="calibration-v1",
            embodiment="new_embodiment",
            coordinate_frame="robot_base",
            object_id="can",
            object_xyz=[float(i), 0, 0],
            object_xyz_units="m",
            object_xyz_frame="robot_base",
            object_xyz_source="fixture ground truth",
        )
        for i in (3, 9)
    }
    (dataset / "meta" / "wah_diagnostics.json").write_text(json.dumps({"episodes": metadata}))
    return [
        "--checkpoint",
        str(checkpoint),
        "--dataset-path",
        str(dataset),
        "--output-dir",
        str(tmp_path / "report"),
        "--device",
        "cpu",
        "--num-samples",
        "2",
    ]


@pytest.mark.parametrize("local", [False, True])
def test_isolated_swap_gates_observers_and_exception_cleanup(local):
    policy = Policy(local)
    dit = policy.model.action_head.model
    before = deepcopy(dit.state_dict())
    x = dict(
        action=torch.ones(1, 3, 4),
        points=torch.ones(1, 5, 4),
        point_mask=torch.ones(1, 5, dtype=torch.bool),
        tactile=torch.ones(1, 2, 4),
        state=torch.ones(1, 1, 4),
        action_mask=torch.tensor([[True, True, False]]),
    )
    donor = torch.full((1, 5, 4), 8.0)
    if local:
        x.update(
            points={name: {"feat": x["points"]} for name in dit.sidecar_point_stage_names},
            point_mask=None,
        )
        donor = {name: {"feat": donor} for name in dit.sidecar_point_stage_names}
    rows = []
    with sidecar_intervention(dit, rows=rows):
        original = dit(**x)
    with sidecar_intervention(dit, swapped_points=(donor, x["point_mask"]), rows=[]):
        swapped = dit(**x)
    with sidecar_intervention(dit, zero_stages=[0, 1], rows=[]):
        disabled = dit(**x)
    assert not torch.equal(original, swapped)
    assert not torch.equal(original, disabled)
    for contact in dit.contact_inputs[1:]:
        for a, b in zip(contact, dit.contact_inputs[0]):
            assert torch.equal(
                a, b
            )  # Original point-dependent contact and MoE sensor inputs retained.
    assert {r["block_index"] for r in rows} == {1, 4}
    assert all(r["residual_ratio"] > 0 for r in rows if r["kind"] == "sidecar_action")
    with pytest.raises(RuntimeError), sidecar_intervention(dit, zero_stages=[1], rows=[]):
        assert dit.sidecar_stages[1].action_gate == 0
        raise RuntimeError("interrupted forward")
    assert not hasattr(dit, "_diagnostic_observer")
    assert not dit.sidecar_pool._forward_pre_hooks
    assert all(not stage._forward_pre_hooks for stage in dit.sidecar_stages)
    for key, value in dit.state_dict().items():
        assert torch.equal(before[key], value)
    assert (
        stage_mapping(dit)[0]["point_gate"] is None
        if local
        else stage_mapping(dit)[0]["point_gate"] > 0
    )


def test_masked_actions_and_undefined_residuals():
    a = np.zeros((1, 2, 3))
    b = np.array([[[2.0, 999.0, 999.0], [999.0, 999.0, 999.0]]])
    mask = np.array([[[1, 0, 0], [0, 0, 0]]])
    result = masked_action_metrics(a, b, mask)
    assert result["mae"] == result["rmse"] == 2
    assert result["valid_elements"] == 1
    assert result["per_dimension_mae"] == [2, None, None]
    assert result["per_timestep_mae"] == [2, None]
    with pytest.raises(ValueError):
        masked_action_metrics(a, b, mask * 0)
    with pytest.raises(ValueError):
        masked_action_metrics(a, b, mask * 0.5)
    result = observed_residual(torch.ones(1, 2, 3), torch.zeros(1, 2, 3))
    assert result["residual_ratio"] == 0
    assert result["cosine_similarity"] is None
    result = observed_residual(torch.zeros(1, 2, 3), torch.ones(1, 2, 3))
    assert result["residual_ratio"] is None
    result = observed_residual(
        torch.ones(1, 2, 3), torch.ones(1, 2, 3), torch.tensor([[True, False]])
    )
    assert result["residual_ratio"] == pytest.approx(1)
    assert result["cosine_similarity"] == pytest.approx(1)


def test_noise_replay_is_exact_and_rng_restored():
    policy = Policy()
    inputs = {
        "inputs": dict(
            points=torch.ones(1, 5, 6),
            tactile=torch.ones(1, 2, 4),
            state=torch.ones(1, 1, 2),
            action_mask=torch.ones(1, 3, 4),
        )
    }
    rng = torch.random.get_rng_state()
    _, noise, _ = paired_inference(policy, inputs, seed=32)
    _, other, _ = paired_inference(policy, inputs, seed=32, zero_stages=[0, 1], reference=noise)
    assert torch.equal(noise, other)
    assert torch.equal(rng, torch.random.get_rng_state())
    with pytest.raises(RuntimeError, match="noise differs"):
        paired_inference(policy, inputs, seed=33, reference=noise)
    bf16 = noise.to(torch.bfloat16)
    assert noise_identity(bf16) == noise_identity(bf16.clone())
    assert noise_identity(bf16) != noise_identity(bf16.float())


@pytest.mark.parametrize("experiment", ["point_ablation", "gate_diagnostics"])
@pytest.mark.parametrize("local", [False, True])
def test_full_cli_cpu_outputs(cli_inputs, experiment, local):
    policy = Policy(local)
    weights = deepcopy(policy.model.state_dict())
    checkpoint = Path(cli_inputs[1])
    checkpoint_before = {p.name: p.read_bytes() for p in checkpoint.iterdir()}
    assert (
        main(
            [*cli_inputs, "--experiment", experiment],
            policy_factory=lambda *a, **k: policy,
            loader_factory=lambda *a: Loader(),
        )
        == 0
    )
    output = Path(cli_inputs[5])
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert summary["samples_completed"] == 2
    assert summary["initial_noise_verified"]
    assert summary["config"]["checkpoint_identity"] == "sha256:mock-checkpoint"
    assert summary["config"]["sensor_contract"]["pointcloud"]["modality_keys"] == ["scene"]
    data = json.loads((output / "results.json").read_text())
    normalized = [r for r in data["per_sample"] if r["space"] == "normalized"]
    assert len(normalized) == 6
    assert all(r["mae"] > 0 for r in normalized)
    assert all(r["valid_elements"] == 4 for r in normalized)
    for row in normalized:
        physical = next(
            r
            for r in data["per_sample"]
            if r["sample"] == row["sample"]
            and r["comparison"] == row["comparison"]
            and r["space"] == "decoded:joint"
        )
        assert physical["mae"] == pytest.approx(row["mae"] * 2)
    assert [r["block_index"] for r in summary["gates"]] == [1, 4]
    assert all(r["n_samples"] == 2 and r["n_forward_calls"] == 4 for r in summary["stage_summary"])
    assert (output / "action_sensitivity.png").is_file()
    if experiment == "point_ablation":
        assert (output / "geometry_sensitivity.png").is_file()
        assert len(data["geometry"]) == 2
        assert data["geometry"][0]["object_displacement_m"] == 6
        assert data["geometry"][0]["cartesian_response"]["status"] == "skipped"
    else:
        zeros = [
            r
            for r in data["per_stage"]
            if r["condition"] == "zero_all" and r["kind"] == "sidecar_action"
        ]
        assert zeros and all(
            r["residual_norm"] == 0 and r["cosine_similarity"] is None for r in zeros
        )
    for filename in ("summary.csv", "per_sample.csv", "stage_metrics.csv", "stage_summary.csv"):
        with (output / filename).open() as stream:
            assert list(csv.DictReader(stream))
    assert checkpoint_before == {p.name: p.read_bytes() for p in checkpoint.iterdir()}
    for key, value in policy.model.state_dict().items():
        assert torch.equal(weights[key], value)


def test_missing_scene_metadata_is_na(cli_inputs):
    path = Path(cli_inputs[3]) / "meta" / "wah_diagnostics.json"
    path.write_text('{"episodes": {}}')
    assert (
        main(
            [*cli_inputs, "--experiment", "point_ablation"],
            policy_factory=lambda *a, **k: Policy(),
            loader_factory=lambda *a: Loader(),
        )
        == 0
    )
    output = Path(cli_inputs[5])
    report = json.loads((output / "results.json").read_text())
    skipped = [r for r in report["per_sample"] if r["status"] == "skipped"]
    assert len(skipped) == 4 and all(r["mae"] is None for r in skipped)
    assert not (output / "geometry_sensitivity.png").exists()
    summary = json.loads((output / "summary.json").read_text())
    assert all(
        r["mean"] is None and r["n"] == 0
        for r in summary["action_summary"]
        if "swapped" in r["comparison"]
    )


def test_interruption_preserves_completed_conditions_and_restores(cli_inputs):
    policy = Policy(fail_at=3)
    before = deepcopy(policy.model.state_dict())
    assert (
        main(
            [*cli_inputs, "--experiment", "gate_diagnostics"],
            policy_factory=lambda *a, **k: policy,
            loader_factory=lambda *a: Loader(),
        )
        == 130
    )
    summary = json.loads((Path(cli_inputs[5]) / "summary.json").read_text())
    assert summary["status"] == "interrupted" and summary["samples_completed"] == 0
    assert summary["conditions_completed"] == 2
    assert summary["action_summary"]
    assert summary["errors"][0]["condition"] == "zero_stage_1"
    for key, value in policy.model.state_dict().items():
        assert torch.equal(before[key], value)
    assert not hasattr(policy.model.action_head.model, "_diagnostic_observer")


def test_exception_retains_error_artifacts(cli_inputs):
    def fail(*a, **k):
        raise ValueError("checkpoint unavailable")

    assert main([*cli_inputs, "--experiment", "gate_diagnostics"], policy_factory=fail) == 1
    summary = json.loads((Path(cli_inputs[5]) / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert "checkpoint unavailable" in summary["errors"][0]["reason"]
    assert (Path(cli_inputs[5]) / "per_sample.csv").exists()


def test_provenance_scene_identity_and_geometry():
    a = dict(
        task="lift_can", embodiment="robot", coordinate_frame="base", sensor="calib", scene="same"
    )
    b = dict(a)
    assert compatible_donor(3, {"3": a, "9": b}, [3, 9])[0] is None
    b["scene"] = "different"
    assert compatible_donor(3, {"3": a, "9": b}, [3, 9])[0] == 9
    assert geometry_metrics(a, b, 0, 0, 0.1)["status"] == "skipped"
    a.update(
        object_id="can",
        object_xyz=[0, 0, 0],
        object_xyz_frame="base",
        object_xyz_units="m",
        object_xyz_source="GT",
    )
    b.update(a, scene="different", object_xyz=[0, 3, 4])
    result = geometry_metrics(a, b, 0, 0, 0.5)
    assert result["object_displacement_m"] == 5
    assert result["normalized_mae_per_m"] == 0.1
    b["object_xyz_units"] = "cm"
    assert geometry_metrics(a, b, 0, 0, 0.5)["status"] == "skipped"


def test_stage_sample_weighting_avoids_flow_pseudoreplication():
    rows = [
        dict(
            task="lift_can",
            condition="original",
            stage=0,
            kind="sidecar_action",
            block_index=4,
            block_number=5,
            sample=sample,
            residual_ratio=value,
        )
        for sample, value in [(0, 1), (0, 1), (0, 1), (1, 3)]
    ]
    result = aggregate_stages(rows)[0]
    assert result["residual_ratio_mean"] == 2  # Each sample has equal weight, not each flow call.
    assert result["residual_ratio_std"] == 1
    assert result["n_samples"] == result["residual_ratio_n"] == 2
    assert result["point_gate_mean"] is None


def test_non_tty_progress_is_bounded_and_has_no_escape_codes(tmp_path):
    stream = io.StringIO()
    config = dict(
        experiment="gate_diagnostics", task="lift_can", checkpoint="mock", num_samples=100
    )
    with DiagnosticDashboard(config, stream=stream, tty=False) as dashboard:
        for sample in range(100):
            for condition in ("original", "zero_stage_0", "zero_all"):
                dashboard.update(sample, condition)
        dashboard.update(100, "complete")
    text = stream.getvalue()
    assert "checkpoint=mock" in text and "ETA=" in text and "condition=" in text
    assert "\x1b" not in text
    assert len(text.splitlines()) <= 13


def test_rich_dashboard_cleanup_on_exception(monkeypatch):
    monkeypatch.setenv("TERM", "xterm")
    stream = io.StringIO()
    config = dict(experiment="gate_diagnostics", task="lift_can", checkpoint="mock", num_samples=1)
    dashboard = DiagnosticDashboard(config, stream=stream, tty=True)
    with pytest.raises(RuntimeError), dashboard:
        dashboard.update(0, "original")
        raise RuntimeError("test cleanup")
    assert dashboard.live is None


def test_sigterm_handler_restored():
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(InterruptedError), interrupted_by_sigterm():
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    assert signal.getsignal(signal.SIGTERM) is before


def test_validation_rejects_output_over_input_and_raw_zero3(cli_inputs):
    args = parser().parse_args([*cli_inputs, "--experiment", "gate_diagnostics"])
    args.output_dir = args.checkpoint / "report"
    with pytest.raises(ValueError, match="overlap"):
        validate_args(args)
    args.output_dir = Path(cli_inputs[5])
    (args.checkpoint / "model.safetensors").rename(args.checkpoint / "model_states.pt")
    with pytest.raises(ValueError, match="ZeRO-3"):
        validate_args(args)
    policy = Policy()
    policy.model.action_head.model.sidecar_stages[0].action_gate.ds_id = 1
    with pytest.raises(ValueError, match="ZeRO-3"):
        stage_mapping(policy.model.action_head.model)


def test_help_never_imports_torch_or_training():
    root = Path(__file__).resolve().parents[3]
    code = (
        "import sys; from gr00t.experiment.wah_diagnostics import parser; "
        "parser().format_help(); assert 'torch' not in sys.modules; "
        "assert 'gr00t.experiment.experiment' not in sys.modules"
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "PYTEST_CURRENT_TEST",
            "GROOT_HF_LOCAL_FIRST",
            "GROOT_PATCH_MISTRAL",
            "GROOT_SKIP_HF_MODEL_WEIGHTS",
        }
    }
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_wandb_disabled_and_connection_failure_do_not_block(cli_inputs, monkeypatch):
    calls = []

    def fail(**kwargs):
        calls.append(kwargs)
        raise ConnectionError("offline network")

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fail, Settings=lambda **k: k))
    assert (
        main(
            [*cli_inputs, "--experiment", "gate_diagnostics", "--wandb"],
            policy_factory=lambda *a, **k: Policy(),
            loader_factory=lambda *a: Loader(),
        )
        == 0
    )
    summary = json.loads((Path(cli_inputs[5]) / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert summary["wandb"]["status"] == "failed"
    assert calls[0]["settings"]["disable_git"] and calls[0]["settings"]["disable_code"]
    assert calls[0]["job_type"] == "wah-diagnostics" and calls[0]["resume"] == "never"
    report = DiagnosticReport(Path(cli_inputs[5]).parent / "disabled-report", {})
    args = SimpleNamespace(wandb=True, wandb_mode="disabled")
    DiagnosticWandb(args, report).start()
    assert len(calls) == 1


def test_wandb_logs_axes_tables_and_local_artifact(tmp_path, monkeypatch):
    logged, definitions, artifacts, finishes, initialization = [], [], [], [], []

    class Config(dict):
        def update(self, values, **kwargs):
            super().update(values)

    run = SimpleNamespace(
        id="mockrun",
        url=None,
        config=Config(),
        summary={},
        define_metric=lambda *a, **k: definitions.append((a, k)),
        log=logged.append,
        log_artifact=artifacts.append,
        finish=lambda **k: finishes.append(k),
    )

    class Artifact:
        def __init__(self, *args, **kwargs):
            self.files = []

        def add_file(self, path, **kwargs):
            self.files.append(Path(path).name)

    def init(**kwargs):
        initialization.append(kwargs)
        return run

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            init=init,
            Settings=lambda **k: k,
            Table=lambda **k: k,
            Image=lambda x: x,
            Artifact=Artifact,
        ),
    )
    report = DiagnosticReport(
        tmp_path, dict(task="lift_can", checkpoint_identity="mock", num_samples=1)
    )
    report.summary["status"] = "complete"
    report.summary["samples_completed"] = 1
    report.samples.append(
        dict(
            sample=0,
            task="lift_can",
            comparison="original__vs__zero_all",
            space="normalized",
            mae=0.2,
            rmse=0.3,
        )
    )
    report.stages.append(
        dict(
            sample=0,
            task="lift_can",
            condition="original",
            kind="sidecar_action",
            stage=0,
            block_index=4,
            block_number=5,
            residual_ratio=0.2,
            action_gate=0.1,
            point_gate=None,
            cosine_similarity=0.4,
        )
    )
    report.save()
    args = SimpleNamespace(
        wandb=True,
        wandb_mode="offline",
        wandb_project="wah-univtac",
        wandb_group=None,
        wandb_name=None,
        experiment="gate_diagnostics",
        task="lift_can",
    )
    tracker = DiagnosticWandb(args, report)
    tracker.start()
    tracker.sample(0, report.samples)
    tracker.finish()
    assert initialization[0]["mode"] == "offline"
    assert initialization[0]["group"].startswith("diagnostics-")
    assert any(d[1].get("step_metric") == "diagnostic_sample" for d in definitions)
    assert any(d[1].get("step_metric") == "diagnostic_stage" for d in definitions)
    assert all("training_step" not in str(item) for item in logged + definitions)
    assert any("diagnostics/stages" in item and "diagnostics/per_task" in item for item in logged)
    assert {"summary.json", "summary.csv", "per_sample.csv", "results.json"}.issubset(
        artifacts[0].files
    )
    assert finishes == [{"exit_code": 0}]


@pytest.mark.parametrize("local", [False, True])
def test_real_small_sidecar_forward_contract(local):
    from gr00t.model.extension.sidecar_dit import SidecarConditionedDiT

    model = SidecarConditionedDiT(
        num_attention_heads=2,
        attention_head_dim=4,
        output_dim=8,
        num_layers=2,
        dropout=0.0,
        final_dropout=False,
        positional_embeddings=None,
        interleave_self_attention=True,
        cross_attention_dim=6,
        sidecar_num_stages=2,
        sidecar_stage_indices=[0, 1],
        sidecar_hidden_dim=8,
        sidecar_num_heads=2,
        sidecar_num_point_tokens=3,
        contact_hidden_dim=8,
        contact_num_tokens=2,
        tactile_fusion_mode="none" if local else "surprise",
        sidecar_variant="multilevel_local" if local else "legacy",
        sidecar_point_stage_names=["enc1", "dec0"] if local else None,
        sidecar_point_stage_dims=[6, 4] if local else None,
        sidecar_local_patch_size=2,
    ).eval()
    with torch.no_grad():
        for stage in model.sidecar_stages:
            stage.action_gate.fill_(0.4)
    inputs = dict(
        hidden_states=torch.randn(1, 4, 8),
        encoder_hidden_states=torch.randn(1, 6, 6),
        timestep=torch.tensor([1]),
        image_mask=torch.tensor([[0, 0, 0, 1, 1, 1]]).bool(),
        backbone_attention_mask=torch.ones(1, 6).bool(),
        hidden_attention_mask=torch.tensor([[1, 1, 1, 0]]).bool(),
        robot_state=torch.randn(1, 1, 8),
        tactile_tokens=None if local else torch.randn(1, 2, 8),
    )

    def level(width):
        return dict(
            feat=torch.randn(3, width),
            coord=torch.randn(3, 3),
            offset=torch.tensor([3]),
            serialized_order=torch.arange(3)[None],
        )

    if local:
        inputs["point_tokens"] = dict(enc1=level(6), dec0=level(4))
        donor = dict(enc1=level(6), dec0=level(4)), None
    else:
        inputs["point_tokens"] = torch.randn(1, 5, 8)
        inputs["point_attention_mask"] = torch.ones(1, 5).bool()
        donor = torch.randn(1, 5, 8), torch.ones(1, 5).bool()
    contact_inputs = []
    handle = model.contact_adapter.register_forward_pre_hook(
        lambda module, args: contact_inputs.append(args[0].clone())
    )
    try:
        with torch.inference_mode():
            expected = model(**deepcopy(inputs))
            rows = []
            with sidecar_intervention(model, rows=rows):
                actual = model(**deepcopy(inputs))
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            with sidecar_intervention(model, swapped_points=donor, rows=[]):
                swapped = model(**deepcopy(inputs))
            assert not torch.equal(actual, swapped)
            with sidecar_intervention(model, zero_stages=[0, 1], rows=[]):
                disabled = model(**deepcopy(inputs))
            assert not torch.equal(actual, disabled)
            assert {r["stage"] for r in rows if r["kind"] == "sidecar_action"} == {0, 1}
            if not local:
                assert len(contact_inputs) == 4
                assert all(torch.equal(contact_inputs[0], p) for p in contact_inputs[1:])
    finally:
        handle.remove()
    assert all(float(s.action_gate) == pytest.approx(0.4) for s in model.sidecar_stages)


def test_cartesian_response_requires_verified_eef_layout():
    config = ModalityConfig(
        [0, 1],
        ["eef"],
        action_configs=[
            ActionConfig(ActionRepresentation.RELATIVE, ActionType.EEF, ActionFormat.XYZ_ROTVEC)
        ],
    )
    a, b = np.zeros((1, 2, 6)), np.zeros((1, 2, 6))
    b[..., 0] = 0.1
    b[:, 1, 0] = 900  # Invalid future timestep must not enter physical response.
    mask = np.zeros((1, 2, 6), bool)
    mask[:, 0] = True
    results = {"original": (a, {"eef": a}), "swapped": (b, {"eef": b})}
    contract = dict(field="eef", units="m", coordinate_frame="base", source="verified config")
    source = dict(object_xyz_frame="base")
    result = cartesian_response(results, mask, config, contract, source, source)
    assert result["status"] == "available"
    assert result["mean_action_displacement_norm_m"] == pytest.approx(0.1)
    config.action_configs[0].type = ActionType.NON_EEF
    assert (
        cartesian_response(results, mask, config, contract, source, source)["status"] == "skipped"
    )
    assert cartesian_response(results, mask, config, {}, source, source)["status"] == "skipped"


def test_scene_metadata_rejects_nonfinite_json(cli_inputs):
    path = Path(cli_inputs[3]) / "meta" / "wah_diagnostics.json"
    path.write_text('{"episodes": {"3": {"object_xyz": [NaN, 0, 0]}}}')
    args = parser().parse_args([*cli_inputs, "--experiment", "point_ablation"])
    with pytest.raises(ValueError, match="Nonfinite JSON"):
        load_scene_metadata(args)

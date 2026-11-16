"""CPU mocks exercise actual Sidecar/flow/HF paths without pretrained downloads."""

from copy import deepcopy
from functools import partial
from types import SimpleNamespace

from gr00t.model.extension.multilevel_local_sidecar import (
    LocalJointActionStage,
    resolve_point_stages,
    snapshot_level,
    validate_checkpoint_variant,
    validate_level,
)
from gr00t.model.extension.point_encoder import ConcertoPointEncoder
from gr00t.model.extension.sidecar_dit import SidecarConditionedDiT
from gr00t.model.modules.dit import AlternateVLDiT
import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


ARCH = dict(in_channels=9, enc_mode=True, enc_channels=[4, 6, 8, 10, 12], enc_depths=[1] * 5)
CFG = dict(
    num_attention_heads=2,
    attention_head_dim=4,
    output_dim=8,
    num_layers=4,
    dropout=0.0,
    final_dropout=False,
    positional_embeddings=None,
    interleave_self_attention=True,
    cross_attention_dim=6,
)


class Point(dict):
    __getattr__ = dict.__getitem__


class MockStage(nn.Module):
    def __init__(self, before, after, downsample):
        super().__init__()
        self.projection = nn.Linear(before, after)
        self.downsample = downsample

    def forward(self, point):
        # Deliberately mutate the same Point to test snapshot safety.
        if self.downsample:
            indices, counts, start = [], [], 0
            for end in point.offset.tolist():
                selected = torch.arange(start, end, 2, device=point.feat.device)
                indices.append(selected)
                counts.append(len(selected))
                start = end
            index = torch.cat(indices)
            point["feat"] = point.feat[index]
            point["coord"] = point.coord[index]
            point["grid_coord"] = point.grid_coord[index]
            point["offset"] = torch.tensor(counts, device=index.device).cumsum(0)
        point["feat"] = self.projection(point.feat)
        point["serialized_order"] = torch.arange(len(point.feat), device=point.feat.device)[None]
        return point


class MockConcerto(nn.Module):
    def __init__(self, **architecture):
        super().__init__()
        self.calls = 0
        self.enc = nn.Sequential()
        before = 9
        for i, after in enumerate(architecture["enc_channels"]):
            self.enc.add_module(f"enc{i}", MockStage(before, after, i > 0))
            before = after

    def forward(self, inputs):
        self.calls += 1
        return self.enc(Point(inputs))


def encoder(architecture=ARCH, dim=8):
    return ConcertoPointEncoder(
        backbone=MockConcerto(**architecture),
        backbone_output_dim=architecture["enc_channels"][-1],
        architecture_config=architecture,
        point_dim=dim,
        load_pretrained=False,
    )


def level(counts=(5, 3), dim=6):
    n = sum(counts)
    return dict(
        feat=torch.randn(n, dim, requires_grad=True),
        coord=torch.randn(n, 3),
        offset=torch.tensor(counts).cumsum(0),
        grid_coord=torch.arange(n)[:, None].expand(-1, 3),
        serialized_order=torch.arange(n)[None],
    )


def model(variant="multilevel_local", **extra):
    args = dict(
        sidecar_hidden_dim=8,
        sidecar_num_heads=2,
        sidecar_num_point_tokens=3,
        contact_hidden_dim=8,
        contact_num_tokens=2,
        tactile_fusion_mode="none",
        sidecar_variant=variant,
    )
    if variant == "multilevel_local":
        args.update(
            sidecar_point_stage_names=["enc1", "enc2", "enc3", "enc4"],
            sidecar_point_stage_dims=[6, 8, 10, 12],
            sidecar_local_patch_size=3,
            sidecar_window_chunk_size=2,
        )
    args.update(extra)
    return SidecarConditionedDiT(**CFG, **args).eval()


def inputs():
    return dict(
        hidden_states=torch.randn(2, 5, 8),
        encoder_hidden_states=torch.randn(2, 6, 6),
        timestep=torch.tensor([3, 7]),
        image_mask=torch.tensor([[0, 0, 0, 1, 1, 1]] * 2).bool(),
        backbone_attention_mask=torch.ones(2, 6).bool(),
        hidden_attention_mask=torch.tensor([[1, 1, 1, 1, 0]] * 2).bool(),
    )


def test_extraction_gradients_and_mutable_point_snapshot():
    enc = encoder()
    raw = torch.randn(2, 33, 6)
    mask = torch.ones(2, 33).bool()
    mask[1, 25:] = False
    levels = enc.forward_multilevel(raw, mask, stage_names=["enc1", "enc2", "enc3", "enc4"])
    assert enc.backbone.calls == 1
    assert [levels[n]["feat"].shape[-1] for n in levels] == [6, 8, 10, 12]
    assert levels["enc1"]["offset"].tolist() == [17, 30]
    assert levels["enc4"]["offset"].tolist() == [3, 5]
    for name, dim in zip(levels, [6, 8, 10, 12]):
        validate_level(levels[name], dim, 2)
        assert len(levels[name]["coord"]) == len(levels[name]["feat"])
    sum(item["feat"].sum() for item in levels.values()).backward()
    assert enc.backbone.enc.enc0.projection.weight.grad.abs().sum() > 0
    assert all(not m._forward_hooks for m in enc.backbone.enc.children())
    legacy, valid = enc(raw, mask, return_mask=True)
    assert legacy.shape == (2, 3, 8) and valid.sum().item() == 5


def test_snapshot_inplace_mutation_retains_grad():
    p = level()
    original = p["feat"]
    saved = snapshot_level(p, 6)
    p["feat"] = torch.zeros(8, 1)
    p["coord"].add_(100)
    assert saved["feat"].shape == (8, 6)
    assert not torch.equal(saved["coord"], p["coord"])
    saved["feat"].sum().backward()
    assert torch.equal(original.grad, torch.ones_like(original))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_noop_gradient_padding_batch_isolation_and_chunking(dtype):
    stage = LocalJointActionStage(8, 6, 8, 2, patch_size=3, window_chunk_size=1).to(dtype)
    data = level()
    action = torch.randn(2, 4, 8).to(dtype).requires_grad_()
    mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
    out, _ = stage(action, data, action_mask=mask)
    assert torch.equal(action, out)
    out.float().square().mean().backward()
    assert stage.action_gate.grad.abs() > 0
    stage.zero_grad()
    with torch.no_grad():
        stage.action_gate.fill_(0.7)
    out, _ = stage(action, data, action_mask=mask)
    assert out.shape == action.shape
    assert torch.equal(out[0, -1], action[0, -1])
    out.float().square().mean().backward()
    assert stage.qkv.weight.grad.abs().sum() > 0
    assert data["feat"].grad.abs().sum() > 0
    changed = deepcopy({k: v.detach() for k, v in data.items()})
    changed["feat"][5:] += 100
    changed["coord"][5:] += 100
    other_action = action.detach().clone()
    other_action[1] += 100
    altered, _ = stage(other_action, changed, action_mask=mask)
    torch.testing.assert_close(out[0], altered[0], rtol=0, atol=0)
    # Padded actions cannot act as keys for valid action queries.
    padded = action.detach().clone()
    padded[0, -1] += 1000
    masked_out, _ = stage(padded, data, action_mask=mask)
    torch.testing.assert_close(out[0, :3], masked_out[0, :3], rtol=0, atol=0)
    stage.window_chunk_size = 8
    chunked, _ = stage(action, data, action_mask=mask)
    torch.testing.assert_close(
        out,
        chunked,
        atol=0.01 if dtype != torch.float32 else 1e-6,
        rtol=0.01 if dtype != torch.float32 else 1e-6,
    )
    assert stage._z3_leaf


def test_replicated_windows_and_unweighted_aggregation():
    stage = LocalJointActionStage(8, 6, 8, 2, patch_size=3, window_chunk_size=8)
    data, action = level((5,)), torch.randn(1, 4, 8)
    with torch.no_grad():
        stage.action_gate.fill_(1.0)
        stage.output_projection.weight.copy_(torch.eye(8))
        stage.output_projection.bias.zero_()
    seen = []

    def joint(actions, points, valid):
        seen.append((actions.clone(), valid.clone()))
        # Different known attention-branch updates per window.
        return torch.stack([torch.ones_like(actions[0]), torch.full_like(actions[0], 3)])

    stage._joint = joint
    out, _ = stage(action, data)
    torch.testing.assert_close(out, action + 2)
    assert torch.equal(seen[0][0][0], seen[0][0][1])
    assert seen[0][1].tolist() == [[True] * 7, [True] * 6 + [False]]


def test_stage_mapping_current_action_latents_and_checkpointing():
    m, x = model(), inputs()
    x["point_tokens"] = {
        n: level(dim=d) for n, d in zip(m.sidecar_point_stage_names, [6, 8, 10, 12])
    }
    baseline = AlternateVLDiT(**CFG).eval()
    baseline.load_state_dict(
        {k: v for k, v in m.state_dict().items() if k in baseline.state_dict()}, strict=True
    )
    ref = {k: v for k, v in x.items() if k != "point_tokens"}
    torch.testing.assert_close(m(**x), baseline(**ref), rtol=0, atol=0)
    expected, observed = [], []
    original = m._run_original_gr00t_block

    def block(*args):
        out = original(*args)
        expected.append((out[:, 1:] * x["hidden_attention_mask"][:, 1:, None]).detach().clone())
        return out

    m._run_original_gr00t_block = block
    handles = []
    for module in m.sidecar_stages:
        with torch.no_grad():
            module.action_gate.fill_(0.3)
        handles.append(
            module.register_forward_pre_hook(
                lambda module, args: observed.append(args[0].detach().clone())
            )
        )
    output = m(**x)
    assert len(observed) == len(expected) == 4
    for before, actual in zip(expected, observed):
        torch.testing.assert_close(before, actual)
    for h in handles:
        h.remove()
    m._run_original_gr00t_block = original
    m.gradient_checkpointing = True
    m._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
    checkpointed = m(**x)
    torch.testing.assert_close(output, checkpointed)
    checkpointed.square().mean().backward()
    assert all(s.qkv.weight.grad.abs().sum() > 0 for s in m.sidecar_stages)
    assert m.stage_indices == [0, 1, 2, 3]


def test_invalid_mapping_and_serialization():
    assert resolve_point_stages(ARCH, None, 4) == (["enc1", "enc2", "enc3", "enc4"], [6, 8, 10, 12])
    for names in [["missing"], ["enc1", "enc1"]]:
        with pytest.raises(ValueError):
            resolve_point_stages(ARCH, names, len(names))
    bad = level()
    bad["serialized_order"][0, [0, 5]] = bad["serialized_order"][0, [5, 0]]
    with pytest.raises(ValueError, match="batch boundary"):
        validate_level(bad, 6, 2)
    with pytest.raises(ValueError):
        model(sidecar_point_stage_dims=[6])


def test_variant_compatibility():
    legacy = SimpleNamespace(dit_type="sidecar_conditioned_dit")
    local = SimpleNamespace(
        dit_type="sidecar_conditioned_dit",
        sidecar_variant="multilevel_local",
        sidecar_point_stage_names=["enc1"],
        sidecar_local_patch_size=8,
        sidecar_window_chunk_size=2,
    )
    validate_checkpoint_variant(SimpleNamespace(dit_type="alternate_vl_dit"), local)
    validate_checkpoint_variant(legacy, legacy)
    validate_checkpoint_variant(local, local)
    for a, b in [(legacy, local), (local, legacy)]:
        with pytest.raises(ValueError, match="variant"):
            validate_checkpoint_variant(a, b)
    altered = deepcopy(local)
    altered.sidecar_point_stage_names = ["enc2"]
    with pytest.raises(ValueError, match="configuration"):
        validate_checkpoint_variant(local, altered)


def head_config(**overrides):
    from tests.gr00t.model.test_action_head import _small_config

    args = dict(
        dit_type="sidecar_conditioned_dit",
        sidecar_variant="multilevel_local",
        sidecar_num_stages=2,
        sidecar_point_stage_names=["enc1", "enc4"],
        sidecar_hidden_dim=16,
        sidecar_num_heads=2,
        sidecar_local_patch_size=3,
        sidecar_window_chunk_size=2,
        point_encoder_cfg="concerto_small",
        point_input_dim=6,
        use_point_conditioning=True,
        use_tactile_conditioning=False,
        point_encoder_model_config=deepcopy(ARCH),
        point_encoder_pretrained_load_on_init=False,
        tactile_fusion_mode="none",
        contact_prediction_loss_weight=0.0,
        contact_injection_enabled=False,
        tune_diffusion_model=False,
        freeze_pretrained_dit=True,
        tune_point_encoder=True,
        tune_multimodal_adapter=True,
    )
    args.update(overrides)
    return _small_config(**args)


def mock_builder(name, **kwargs):
    return encoder(kwargs.get("architecture_config") or ARCH, kwargs["point_dim"])


@pytest.mark.parametrize("mode", ["frozen", "lora", "full"])
def test_flow_forward_inference_reuse_save_reload(monkeypatch, tmp_path, mode):
    from gr00t.experiment.sidecar_evaluation import gate_rows, intervention
    import gr00t.model.gr00t_n1d7.gr00t_n1d7 as module
    from tests.gr00t.model.test_action_head import _make_action_input, _make_backbone_output

    monkeypatch.setattr(module, "build_point_encoder", mock_builder)
    cfg = head_config(
        dit_training_mode=mode,
        tune_diffusion_model=mode == "full",
        freeze_pretrained_dit=mode != "full",
    )
    head = module.Gr00tN1d7ActionHead(cfg)
    assert not any(p.requires_grad for p in head.point_encoder.projection.parameters())
    assert any(p.requires_grad for p in head.model.sidecar_stages.parameters())
    action, backbone = _make_action_input(cfg), _make_backbone_output(cfg)
    action["points"] = torch.randn(2, 33, 6)
    with torch.no_grad():
        for stage in head.model.sidecar_stages:
            stage.action_gate.fill_(0.2)
    result = head(deepcopy(backbone), action)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert all(s.qkv.weight.grad.abs().sum() > 0 for s in head.model.sidecar_stages)
    if mode == "full":
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in head.model.transformer_blocks.parameters()
        )
    elif mode == "lora":
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for n, p in head.model.named_parameters()
            if ".lora_B." in n
        )
    else:
        assert not any(p.requires_grad for p in head.model.transformer_blocks.parameters())
    action.pop("action")
    head.eval()
    before = head.point_encoder.backbone.calls
    rows = []
    with intervention(head.model, rows=rows), torch.no_grad():
        torch.manual_seed(19)
        prediction = head.get_action(deepcopy(backbone), action).action_pred
    assert head.point_encoder.backbone.calls == before + 1
    assert len(rows) == cfg.num_inference_timesteps * 2
    assert rows[0]["point_stage"] == "enc1" and rows[0]["window_counts"]
    assert gate_rows(head.model)[0]["point_gate"] is None
    assert prediction.shape == (2, cfg.action_horizon, cfg.max_action_dim)
    cfg.save_pretrained(tmp_path)
    torch.save(head.state_dict(), tmp_path / "head.pt")
    restored_config = type(cfg).from_pretrained(tmp_path)
    restored = module.Gr00tN1d7ActionHead(restored_config).eval()
    restored.load_state_dict(torch.load(tmp_path / "head.pt", weights_only=True), strict=True)
    torch.manual_seed(19)
    with torch.no_grad():
        repeated = restored.get_action(backbone, action).action_pred
    torch.testing.assert_close(prediction, repeated, rtol=0, atol=0)
    assert restored_config.sidecar_variant == "multilevel_local"
    assert restored_config.sidecar_point_stage_names == ["enc1", "enc4"]


def test_hf_model_roundtrip_and_variant_override_rejection(monkeypatch, tmp_path):
    import gr00t.model.gr00t_n1d7.gr00t_n1d7 as module
    import gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 as processing

    monkeypatch.setenv("GROOT_SKIP_HF_MODEL_WEIGHTS", "0")
    from tests.gr00t.model.test_action_head import _make_action_input, _make_backbone_output
    from transformers import AutoModel

    class Backbone(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))

        def prepare_input(self, inputs):
            return {
                k: inputs[k] for k in ["backbone_features", "backbone_attention_mask", "image_mask"]
            }

        def forward(self, inputs):
            return {**inputs, "backbone_features": inputs["backbone_features"] * self.scale}

    monkeypatch.setattr(module, "get_backbone_cls", lambda cfg: Backbone)
    monkeypatch.setattr(module, "build_point_encoder", mock_builder)
    monkeypatch.setattr(processing, "Gr00tN1d7DataCollator", lambda **kw: None)
    cfg = head_config()
    cfg.sidecar_action_gate_init = {
        "mode": "constant",
        "value": 0.02,
        "low": 0.01,
        "high": 0.05,
        "mean": 0.02,
        "std": 0.01,
    }
    original = module.Gr00tN1d7(cfg).eval()
    assert all(
        stage.action_gate.item() == pytest.approx(0.02)
        for stage in original.action_head.model.sidecar_stages
    )
    with torch.no_grad():
        for index, stage in enumerate(original.action_head.model.sidecar_stages):
            stage.action_gate.fill_(0.25 + index)
    original.save_pretrained(tmp_path)
    restored, info = AutoModel.from_pretrained(tmp_path, output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    restored.eval()
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0, msg=name)
    assert [stage.action_gate.item() for stage in restored.action_head.model.sidecar_stages] == [
        0.25 + index for index in range(len(restored.action_head.model.sidecar_stages))
    ]
    action = dict(_make_action_input(cfg))
    action.pop("action")
    action["points"] = torch.randn(2, 33, 6)
    inputs = {**action, **dict(_make_backbone_output(cfg))}
    with torch.no_grad():
        torch.manual_seed(3)
        a = original.get_action(deepcopy(inputs)).action_pred
        torch.manual_seed(3)
        b = restored.get_action(deepcopy(inputs)).action_pred
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match="variant"):
        AutoModel.from_pretrained(tmp_path, sidecar_variant="legacy")


@pytest.mark.parametrize("variant", ["legacy", "multilevel_local"])
@pytest.mark.parametrize("value", [0.0, 0.02])
def test_hf_missing_adapter_initialization(monkeypatch, tmp_path, variant, value):
    # HF no-init contexts must not replace zero scalar gates with uninitialized values.
    from gr00t.model.gr00t_n1d7.pretrained_loading import initialize_missing_sidecar

    cfg = head_config(sidecar_variant=variant)
    settings = {
        "mode": "zero" if value == 0 else "constant",
        "value": value,
        "low": 0.01,
        "high": 0.05,
        "mean": 0.02,
        "std": 0.01,
    }
    cfg.sidecar_action_gate_init = settings
    dit = model(variant, sidecar_action_gate_init=settings)
    if variant == "multilevel_local":
        cfg.sidecar_point_stage_names = dit.sidecar_point_stage_names
        cfg.sidecar_hidden_dim = 8
        cfg.sidecar_num_heads = 2
    else:
        cfg.sidecar_hidden_dim = 8
        cfg.sidecar_num_heads = 2
        cfg.sidecar_num_point_tokens = 3
    cfg.contact_hidden_dim = 8
    cfg.contact_num_tokens = 2
    fake = SimpleNamespace(
        config=cfg, action_head=SimpleNamespace(model=dit), named_parameters=lambda: []
    )
    keys = [
        f"action_head.model.sidecar_stages.{name}"
        for name, _ in dit.sidecar_stages.named_parameters()
    ]
    with torch.no_grad():
        for stage in dit.sidecar_stages:
            stage.action_gate.fill_(99)
    initialize_missing_sidecar(fake, keys)
    assert all(s.action_gate.item() == pytest.approx(value) for s in dit.sidecar_stages)


@pytest.mark.parametrize("mode", ["frozen", "lora", "full"])
def test_hydra_experiment_and_preflight(mode):
    from pathlib import Path

    from gr00t.configs.hydra_config import build_gr00t_config, preflight
    from hydra import compose, initialize_config_dir

    root = Path(__file__).resolve().parents[3]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.3"):
        cfg = compose(config_name=f"experiment/wah_multilevel_local_{mode}")
        built = build_gr00t_config(cfg)
        assert built.model.sidecar_variant == "multilevel_local"
        assert built.model.dit_training_mode == mode
        assert built.model.tune_diffusion_model == (mode == "full")
        assert preflight(cfg, built)["sidecar"]["point_stages"] == ["enc1", "enc2", "enc3", "enc4"]


@pytest.mark.parametrize(
    "overrides",
    [
        dict(temporal_window_size=5),
        dict(contact_input_mode="moe_soft"),
        dict(tactile_fusion_mode="surprise"),
        dict(sidecar_point_stage_names=["bad"]),
        dict(sidecar_local_patch_size=0),
        dict(point_encoder_cfg="pointnet2"),
    ],
)
def test_unsupported_combinations_fail_before_loading(overrides):
    from gr00t.model.extension.multilevel_local_sidecar import validate_variant

    with pytest.raises(ValueError):
        validate_variant(head_config(**overrides))


def test_training_diagnostics_local_metadata(monkeypatch):
    from gr00t.experiment.multimodal_diagnostics import MultimodalDiagnostics
    import gr00t.model.gr00t_n1d7.gr00t_n1d7 as module
    from tests.gr00t.model.test_action_head import _make_action_input, _make_backbone_output

    monkeypatch.setattr(module, "build_point_encoder", mock_builder)
    cfg = head_config()
    head = module.Gr00tN1d7ActionHead(cfg)
    trainer = SimpleNamespace(
        model=SimpleNamespace(action_head=head),
        state=SimpleNamespace(global_step=0),
        optimizer=SimpleNamespace(),
    )
    monitor = MultimodalDiagnostics(trainer, interval=1, level="per_layer")
    head.model._diagnostic_observer = monitor.observe
    action = _make_action_input(cfg)
    action["points"] = torch.randn(2, 33, 6)
    head(_make_backbone_output(cfg), action)["loss"].backward()
    monitor.pre_step()
    assert monitor.record["sidecar_stage_mapping"][0]["point_stage"] == "enc1"
    assert monitor.record["sidecar_action_gate_init_mode_stage_0"] == "zero"
    assert monitor.record["sidecar/action_pre_gate_update_norm"] > 0
    assert monitor.record["sidecar/action_gated_residual_norm"] == 0
    assert monitor.record["sidecar/action_actual_residual_norm"] == 0
    assert monitor.record["sidecar_stage_mapping"][1]["dit_block"] == 1
    assert monitor.record["sidecar_stage_mapping"][0]["action_gate"] == 0.0
    assert monitor.record["sidecar_action_gate_stage_0"] == 0.0
    assert monitor.record["sidecar_point_gate_mean_abs"] is None
    assert monitor.record["sidecar_local/window_counts_stage_0"] > 0


def test_nvidia_base_bootstrap_keeps_pretrained_dit(monkeypatch, tmp_path):
    from gr00t.model.gr00t_n1d7.dit_integrity import validate_loading_info
    import gr00t.model.gr00t_n1d7.gr00t_n1d7 as module
    from gr00t.model.gr00t_n1d7.pretrained_loading import initialize_missing_sidecar
    import gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 as processing
    from gr00t.model.gr00t_n1d7.sensor_loading import SIDECAR_ADAPTER_PREFIXES
    from tests.gr00t.model.test_action_head import _small_config
    from transformers import AutoModel

    class Backbone(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2))

    monkeypatch.setenv("GROOT_SKIP_HF_MODEL_WEIGHTS", "0")
    monkeypatch.setattr(module, "get_backbone_cls", lambda cfg: Backbone)
    monkeypatch.setattr(module, "build_point_encoder", mock_builder)
    monkeypatch.setattr(processing, "Gr00tN1d7DataCollator", lambda **kw: None)
    original = module.Gr00tN1d7(_small_config(dit_type="alternate_vl_dit"))
    original.save_pretrained(tmp_path)
    requested = head_config(point_encoder_deferred_bootstrap=True)
    overrides = {
        key: getattr(requested, key)
        for key in [
            "dit_type",
            "sidecar_variant",
            "sidecar_num_stages",
            "sidecar_point_stage_names",
            "sidecar_hidden_dim",
            "sidecar_num_heads",
            "sidecar_local_patch_size",
            "sidecar_window_chunk_size",
            "point_encoder_model_config",
            "point_encoder_deferred_bootstrap",
            "point_encoder_cfg",
            "point_input_dim",
            "use_point_conditioning",
            "use_tactile_conditioning",
            "tactile_fusion_mode",
            "contact_prediction_loss_weight",
            "contact_injection_enabled",
            "tune_diffusion_model",
            "freeze_pretrained_dit",
        ]
    }
    local, info = AutoModel.from_pretrained(tmp_path, output_loading_info=True, **overrides)
    assert validate_loading_info(info, SIDECAR_ADAPTER_PREFIXES)["status"] == "PASS"
    initialize_missing_sidecar(local, info["missing_keys"])
    for name, value in original.action_head.model.state_dict().items():
        torch.testing.assert_close(
            value, local.action_head.model.state_dict()[name], rtol=0, atol=0
        )
    assert all(s.action_gate.item() == 0.0 for s in local.action_head.model.sidecar_stages)
    local.action_head.bootstrap_deferred_sensor_backbones()
    assert local.action_head.point_encoder is not None
    assert local.config.sidecar_point_stage_names == ["enc1", "enc4"]


def test_checkpoint_metadata_and_non_encoder_mode(tmp_path):
    architecture = {**ARCH, "enc_mode": False, "dec_channels": [4, 6, 8, 10]}
    path = tmp_path / "concerto.pth"
    torch.save(dict(config=architecture, state_dict={}), path)
    assert (
        ConcertoPointEncoder.read_architecture_config(
            model_name="concerto_small", checkpoint_path=path
        )
        == architecture
    )
    enc = encoder(architecture)
    data = torch.randn(1, 17, 6)
    levels = enc.forward_multilevel(data, stage_names=["enc1", "enc4"])
    assert list(levels) == ["enc1", "enc4"]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Optional GPU-only one-step smoke")
def test_cuda_one_step_smoke():
    stage = (
        LocalJointActionStage(8, 6, 8, 2, patch_size=4, window_chunk_size=2)
        .cuda()
        .to(torch.bfloat16)
    )
    data = {k: v.detach().cuda() for k, v in level().items()}
    action = torch.randn(2, 4, 8, device="cuda", dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(stage.parameters(), lr=0.01)
    # First zero-gate update trains the scalar gate; then attention sees gradients.
    stage(action, data)[0].float().square().mean().backward()
    optimizer.step()
    optimizer.zero_grad()
    stage(action, data)[0].float().square().mean().backward()
    assert stage.qkv.weight.grad.abs().sum() > 0


def test_cpu_autocast_flow_with_checkpointing(monkeypatch):
    import gr00t.model.gr00t_n1d7.gr00t_n1d7 as module
    from tests.gr00t.model.test_action_head import _make_action_input, _make_backbone_output

    monkeypatch.setattr(module, "build_point_encoder", mock_builder)
    cfg = head_config()
    head = module.Gr00tN1d7ActionHead(cfg)
    head.model.gradient_checkpointing = True
    head.model._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
    with torch.no_grad():
        for stage in head.model.sidecar_stages:
            stage.action_gate.fill_(0.2)
    action = _make_action_input(cfg)
    action["points"] = torch.randn(2, 33, 6)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = head(_make_backbone_output(cfg), action)
    assert result["loss"].dtype == torch.float32
    result["loss"].backward()
    assert all(s.qkv.weight.grad.abs().sum() > 0 for s in head.model.sidecar_stages)


def test_local_optimizer_ownership(monkeypatch, tmp_path):
    import gr00t.model.gr00t_n1d7.gr00t_n1d7 as module
    from tests.gr00t.experiment.test_optimizer_parameter_groups import (
        _learning_rate_by_parameter_id,
        _make_trainer,
        _ToySplitModel,
    )

    monkeypatch.setattr(module, "build_point_encoder", mock_builder)
    toy = _ToySplitModel()
    toy.action_head = module.Gr00tN1d7ActionHead(head_config())
    trainer = _make_trainer(tmp_path, toy)
    trainer.create_optimizer()
    rates = _learning_rate_by_parameter_id(trainer.optimizer)
    for parameter in toy.action_head.model.sidecar_stages.parameters():
        assert rates[id(parameter)] == 1e-3
    ids = [id(p) for group in trainer.optimizer.param_groups for p in group["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in toy.parameters() if p.requires_grad}


def test_offline_local_mapping_export(tmp_path):
    import json

    from gr00t.experiment.sidecar_evaluation import export, offline_rows

    source = tmp_path / "multimodal_diagnostics.jsonl"
    source.write_text(
        json.dumps(
            dict(
                global_step=1,
                sidecar_stage_mapping=[
                    dict(
                        stage=0,
                        point_stage="enc1",
                        dit_block=3,
                        action_gate=0.1,
                        action_injection_ratio=0.02,
                        point_counts=[5, 3],
                        window_counts=[2, 1],
                        window_point_counts=[[3, 2], [3]],
                    )
                ],
            )
        )
        + "\n"
    )
    rows = offline_rows(source)
    assert rows[0]["point_stage"] == "enc1" and rows[0]["block_number"] == 4
    export(tmp_path / "out", dict(geometry=dict(status="unavailable", reason="No GT/FK")), [], rows)
    assert (tmp_path / "out/stage_gate.png").exists()
    assert (tmp_path / "out/stage_residual_ratio.png").exists()


def test_checkpointed_fully_frozen_inputs_still_train_gate(monkeypatch):
    import gr00t.model.gr00t_n1d7.gr00t_n1d7 as module
    from tests.gr00t.model.test_action_head import _make_action_input, _make_backbone_output

    monkeypatch.setattr(module, "build_point_encoder", mock_builder)
    cfg = head_config(tune_projector=False, tune_vlln=False, tune_point_encoder=False)
    head = module.Gr00tN1d7ActionHead(cfg)
    head.model.gradient_checkpointing = True
    head.model._gradient_checkpointing_func = partial(checkpoint, use_reentrant=True)
    action = _make_action_input(cfg)
    action["points"] = torch.randn(2, 33, 6)
    head(_make_backbone_output(cfg), action)["loss"].backward()
    assert all(
        s.action_gate.grad is not None and s.action_gate.grad.abs() > 0
        for s in head.model.sidecar_stages
    )
    assert not any(p.requires_grad for p in head.point_encoder.parameters())


def test_historical_hydra_resume_defaults(tmp_path):
    from gr00t.configs.hydra_config import build_gr00t_config
    from gr00t.experiment.hydra_launch import resolve_resume
    from omegaconf import OmegaConf
    from tests.gr00t.configs.test_hydra_config import checkpoint, composed

    run = tmp_path / "run"
    cfg = composed(overrides=[f"output_dir={run}"])
    build_gr00t_config(cfg)
    historical = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    for key in [
        "sidecar_variant",
        "sidecar_point_stage_names",
        "sidecar_local_patch_size",
        "sidecar_window_chunk_size",
    ]:
        historical.model.pop(key, None)
    (run / ".hydra").mkdir(parents=True)
    OmegaConf.save(historical, run / ".hydra/config.yaml")
    saved = checkpoint(run / "checkpoint-100", ds=True)
    OmegaConf.update(cfg, "model.sidecar_variant", "legacy", force_add=True)
    cfg.resume.enabled = True
    cfg.resume.auto_find_latest = True
    assert resolve_resume(cfg, build_gr00t_config(cfg)) == saved


def test_direct_stage_rejects_cross_batch_window():
    data = level()
    data["serialized_order"][0, [0, 5]] = data["serialized_order"][0, [5, 0]]
    with pytest.raises(ValueError, match="batch boundary"):
        LocalJointActionStage(8, 6, 8, 2)(torch.randn(2, 4, 8), data)

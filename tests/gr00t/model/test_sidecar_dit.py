"""Deterministic CPU tests; no claim about real pretrained checkpoints."""

from functools import partial
import inspect

from gr00t.model.extension.contact_surprise import ContactSurprise
from gr00t.model.extension.point_action_sidecar import TokenPool, stage_indices
from gr00t.model.extension.sidecar_dit import SidecarConditionedDiT
from gr00t.model.modules.dit import AlternateVLDiT
import pytest
import torch
from torch.utils.checkpoint import checkpoint


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


def model(mode="surprise"):
    return SidecarConditionedDiT(
        **CFG,
        sidecar_hidden_dim=8,
        sidecar_num_heads=2,
        sidecar_num_point_tokens=3,
        contact_hidden_dim=8,
        contact_num_tokens=2,
        tactile_fusion_mode=mode,
    ).eval()


def inputs():
    return dict(
        hidden_states=torch.randn(2, 5, 8),
        encoder_hidden_states=torch.randn(2, 6, 6),
        timestep=torch.tensor([3, 9]),
        image_mask=torch.tensor([[0, 0, 0, 1, 1, 1]] * 2).bool(),
        backbone_attention_mask=torch.ones(2, 6).bool(),
        point_tokens=torch.randn(2, 7, 8),
        point_attention_mask=torch.tensor([[1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]).bool(),
        tactile_tokens=torch.randn(2, 4, 8),
        robot_state=torch.randn(2, 1, 8),
        hidden_attention_mask=torch.tensor([[1, 1, 1, 1, 0]] * 2).bool(),
    )


def test_boundaries():
    assert stage_indices(32) == [7, 15, 23, 31]
    assert stage_indices(7) == [1, 3, 5, 6]
    assert stage_indices(2) == [0, 1]
    assert stage_indices(8, indices=[0, 5]) == [0, 5]
    with pytest.raises(ValueError):
        stage_indices(8, indices=[3, 3])


@pytest.mark.parametrize("mode", ["none", "direct", "contact_gate", "surprise", "hybrid"])
def test_parity_modes_and_response(mode):
    torch.manual_seed(8)
    base = AlternateVLDiT(**CFG).eval()
    m = model(mode)
    report = m.load_state_dict(base.state_dict(), strict=False)
    assert not report.unexpected_keys
    assert all(k.startswith(("sidecar_", "contact_")) for k in report.missing_keys)
    x = inputs()
    ref = {
        k: v
        for k, v in x.items()
        if k not in {"point_tokens", "point_attention_mask", "tactile_tokens", "robot_state"}
    }
    torch.testing.assert_close(m(**x), base(**ref), atol=0, rtol=0)
    for k, v in base.state_dict().items():
        assert torch.equal(v, m.state_dict()[k])
    for stage in m.sidecar_stages:
        stage.action_gate.data.fill_(0.3)
    for stage in m.contact_injections:
        stage.scale.data.fill_(0.3)
    y = m(**x)
    assert torch.isfinite(y).all()
    x["point_tokens"] = x["point_tokens"] + torch.randn_like(x["point_tokens"])
    assert not torch.equal(y, m(**x))
    assert torch.equal(y[:, -1], torch.zeros_like(y[:, -1]))


def test_mask_invariance():
    m = model()
    x = inputs()
    for s in m.sidecar_stages:
        s.action_gate.data.fill_(0.4)
    a = m(**x)
    x["point_tokens"][~x["point_attention_mask"]] = 10000
    torch.testing.assert_close(a, m(**x), atol=1e-6, rtol=1e-6)
    pool = TokenPool(8, 8, 3, 2)
    y, valid = pool(torch.randn(2, 4, 8), torch.zeros(2, 4).bool())
    assert torch.isfinite(y).all() and not valid.any() and y.count_nonzero() == 0


def test_gradients_frozen_and_checkpointing():
    m = model().train()
    x = inputs()
    x["point_tokens"].requires_grad_()
    m.gradient_checkpointing = True
    m._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
    original = {
        k: v.clone()
        for k, v in m.state_dict().items()
        if not k.startswith(("sidecar_", "contact_"))
    }
    optimizer = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=0.01)
    m(**x).square().mean().backward()
    assert any(s.action_gate.grad.abs() > 0 for s in m.sidecar_stages)
    optimizer.step()
    optimizer.zero_grad()
    (m(**x).square().mean() + m.contact_prediction_loss).backward()
    assert x["point_tokens"].grad.abs().sum() > 0
    assert m.sidecar_stages[0].point_reads_action.attention.in_proj_weight.grad.abs().sum() > 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in m.contact_adapter.predictor.parameters()
    )
    assert all(p.grad is None for p in m.contact_adapter.projector.parameters())
    optimizer.step()
    for k, v in original.items():
        assert torch.equal(v, m.state_dict()[k])
    assert all(p.grad is None for p in m.transformer_blocks.parameters())


def test_contact_alignment_detach_and_no_action_api():
    c = ContactSurprise(8, 8, 2, 2)
    assert set(inspect.signature(c.forward).parameters) == {
        "points",
        "state",
        "tactile",
        "point_mask",
        "tactile_mask",
    }
    tactile = torch.randn(2, 5, 8, requires_grad=True)
    observed, pred, surprise, gate, valid, loss = c(
        torch.randn(2, 3, 8), torch.randn(2, 1, 8), tactile
    )
    assert observed.shape == pred.shape == surprise.shape == (2, 2, 8)
    torch.testing.assert_close(surprise, observed - pred)
    assert ((gate >= 0) & (gate <= 1)).all()
    loss.backward()
    assert tactile.grad is None
    assert c.predictor[-1].weight.grad.abs().sum() > 0


def test_serialization_and_repeat_inference(tmp_path):
    m = model()
    x = inputs()
    for s in m.sidecar_stages:
        s.action_gate.data.fill_(0.2)
    opt = torch.optim.AdamW(m.parameters())
    (m(**x).square().mean() + m.contact_prediction_loss).backward()
    opt.step()
    path = tmp_path / "resume.pt"
    torch.save({"model": m.state_dict(), "optimizer": opt.state_dict()}, path)
    n = model()
    saved = torch.load(path, weights_only=True)
    n.load_state_dict(saved["model"], strict=True)
    new_opt = torch.optim.AdamW(n.parameters())
    new_opt.load_state_dict(saved["optimizer"])
    torch.testing.assert_close(m(**x), n(**x), atol=0, rtol=0)
    x2 = {**x, "hidden_states": torch.randn_like(x["hidden_states"])}
    m(**x2)
    torch.testing.assert_close(m(**x), n(**x), atol=0, rtol=0)


def test_bfloat16():
    m = model().to(torch.bfloat16)
    x = inputs()
    x = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in x.items()}
    base = AlternateVLDiT(**CFG).eval().to(torch.bfloat16)
    base.load_state_dict({key: m.state_dict()[key] for key in base.state_dict()}, strict=True)
    original_inputs = {
        key: value
        for key, value in x.items()
        if key not in {"point_tokens", "point_attention_mask", "tactile_tokens", "robot_state"}
    }
    y = m(**x)
    torch.testing.assert_close(y, base(**original_inputs), rtol=0, atol=0)
    (y.float().square().mean() + m.contact_prediction_loss).backward()
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("warmup,mode", [(False, "none"), (False, "surprise"), (True, "surprise")])
def test_action_head_flow_and_rollout(warmup, mode, tmp_path):
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
    from tests.gr00t.model.test_action_head import (
        _make_action_input,
        _make_backbone_output,
        _small_multimodal_config,
    )

    cfg = _small_multimodal_config(
        dit_type="sidecar_conditioned_dit",
        sidecar_hidden_dim=16,
        sidecar_num_heads=2,
        sidecar_num_point_tokens=3,
        contact_hidden_dim=16,
        contact_num_tokens=2,
        contact_predictor_warmup=warmup,
        tactile_fusion_mode=mode,
        tune_projector=False,
        tune_diffusion_model=False,
        freeze_pretrained_dit=True,
    )
    head = Gr00tN1d7ActionHead(cfg)
    action = _make_action_input(cfg)
    action["points"] = torch.randn(2, 16, 3)
    action["tactile"] = torch.randn(2, 2, 3, 32, 32)
    backbone = _make_backbone_output(cfg)
    result = head(backbone, action)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    if mode != "none":
        assert head.model.contact_adapter.predictor[-1].weight.grad.abs().sum() > 0
    assert not any(p.requires_grad for p in head.model.transformer_blocks.parameters())
    assert not any(p.requires_grad for p in head.tactile_encoder.parameters())
    if warmup:
        assert all(not p.requires_grad for p in head.model.sidecar_stages.parameters())
    else:
        assert any(s.action_gate.grad.abs() > 0 for s in head.model.sidecar_stages)
    if mode == "none":
        head.zero_grad()
        for stage in head.model.sidecar_stages:
            stage.action_gate.data.fill_(0.2)
        action.pop("tactile")
        head(backbone, action)["loss"].backward()
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in head.point_encoder.parameters()
        )
    head.eval()
    if mode != "none":
        predictions = []
        hook = head.model.contact_adapter.register_forward_hook(
            lambda module, args, out: predictions.append(out[1].detach().clone())
        )
        head(backbone, action)
        action["action"] = torch.randn_like(action["action"]) * 100
        head(backbone, action)
        hook.remove()
        torch.testing.assert_close(predictions[0], predictions[1], rtol=0, atol=0)
    action.pop("action")
    with torch.no_grad():
        output = head.get_action(backbone, action)
    assert torch.isfinite(output.action_pred).all()
    cfg.save_pretrained(tmp_path)
    restored = type(cfg).from_pretrained(tmp_path)
    reloaded = Gr00tN1d7ActionHead(restored)
    reloaded.load_state_dict(head.state_dict(), strict=True)


def test_evolving_points_and_contact_scale_gradients():
    m = model()
    x = inputs()
    seen = []
    handle = m.sidecar_stages[0].register_forward_hook(
        lambda module, args, out: seen.append((args[1].detach(), out[1].detach()))
    )
    m(**x).square().mean().backward()
    handle.remove()
    assert not torch.equal(*seen[0])
    assert any(s.scale.grad.abs() > 0 for s in m.contact_injections)
    assert all(
        s.attention.attention.in_proj_weight.grad.count_nonzero() == 0 for s in m.contact_injections
    )
    for s in m.contact_injections:
        s.scale.data.fill_(0.3)
    before = m(**x)
    x["tactile_tokens"] = torch.randn_like(x["tactile_tokens"])
    assert not torch.equal(before, m(**x))
    m.zero_grad()
    m(**x).square().mean().backward()
    assert m.contact_injections[0].attention.attention.in_proj_weight.grad.abs().sum() > 0
    assert m.contact_adapter.gate[0].weight.grad.abs().sum() > 0


def test_optional_diagnostics_do_not_change_predictions():
    from types import SimpleNamespace

    from gr00t.experiment.multimodal_diagnostics import MultimodalDiagnostics

    m = model()
    x = inputs()
    optimizer = torch.optim.AdamW(m.parameters())
    trainer = SimpleNamespace(
        model=SimpleNamespace(action_head=SimpleNamespace(model=m, point_encoder=None)),
        optimizer=optimizer,
        state=SimpleNamespace(global_step=0),
    )
    monitor = MultimodalDiagnostics(trainer, interval=1)
    before = m(**x)
    m._diagnostic_observer = monitor.observe
    after = m(**x)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    (after.square().mean() + m.contact_prediction_loss).backward()
    monitor.pre_step()
    assert monitor.record["sidecar_action_gate_grad_norm"] > 0
    assert monitor.record["sidecar_action_gate_init_mode_stage_0"] == "zero"
    assert monitor.record["sidecar_action_gate_initial_stage_0"] == 0
    assert monitor.record["sidecar_action_gate_grad_finite_stage_0"] is True
    assert monitor.record["sidecar/action_pre_gate_update_norm"] > 0
    assert monitor.record["sidecar/action_gated_residual_norm"] == 0
    assert monitor.record["sidecar/action_actual_residual_norm"] == 0
    assert monitor.record["contact_prediction_loss"] >= 0
    assert "sidecar_point_residual_ratio_stage_0" in monitor.record
    monitor.post_step()


def test_missing_base_weight_is_not_an_adapter():
    from gr00t.model.gr00t_n1d7.sensor_loading import (
        SIDECAR_ADAPTER_PREFIXES,
        unexpected_missing_keys,
    )

    base = "action_head.model.transformer_blocks.0.attn1.to_q.weight"
    adapter = "action_head.model.sidecar_stages.0.action_gate"
    assert unexpected_missing_keys([base, adapter], SIDECAR_ADAPTER_PREFIXES) == [base]


def test_checkpointing_with_frozen_backbone_in_eval_mode():
    m = model()
    m.gradient_checkpointing = True
    calls = []

    def recompute(function, *args):
        calls.append(True)
        return checkpoint(function, *args, use_reentrant=False)

    m._gradient_checkpointing_func = recompute
    for stage in m.sidecar_stages:
        stage.action_gate.data.fill_(0.2)
    m(**inputs()).square().mean().backward()
    assert len(calls) == len(m.transformer_blocks)
    assert m.sidecar_stages[0].action_reads_point.attention.in_proj_weight.grad.abs().sum() > 0


def test_unfrozen_teacher_still_has_detached_prediction_target():
    c = ContactSurprise(8, 8, 2, 2, freeze_teacher=False)
    tactile = torch.randn(2, 5, 8, requires_grad=True)
    out = c(torch.randn(2, 3, 8), torch.randn(2, 1, 8), tactile)
    out[-1].backward(retain_graph=True)
    assert tactile.grad is None
    assert all(p.grad is None for p in c.projector.parameters())
    out[0].square().mean().backward()
    assert tactile.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in c.projector.parameters())

"""Parameter storage must remain suitable for DeepSpeed's byte broadcast."""

from types import SimpleNamespace

from gr00t.experiment.parameter_layout import audit_parameter_layout
from gr00t.model.modules.qwen3_backbone import Qwen3Backbone
import pytest
import torch
from torch import nn


@pytest.mark.parametrize("trainable", [True, False])
def test_patch_embed_preserves_values_identity_and_ties(trainable):
    conv = nn.Conv3d(3, 8, (2, 2, 2), dtype=torch.bfloat16)
    conv.requires_grad_(trainable)
    visual = nn.ModuleList([conv])
    visual.register_parameter("tied_weight", conv.weight)
    owner = SimpleNamespace(model=SimpleNamespace(visual=visual))
    weight = conv.weight
    pointer = weight.untyped_storage().data_ptr()
    values = weight.detach().clone()
    Qwen3Backbone._apply_vision_patch_embed_channels_last(owner)
    Qwen3Backbone._apply_vision_patch_embed_channels_last(owner)
    assert conv.weight is weight is visual.tied_weight
    assert weight.untyped_storage().data_ptr() == pointer
    assert torch.equal(weight, values)
    assert weight.requires_grad == trainable
    assert weight.is_contiguous() and weight.stride(-1) == 1
    weight.view(torch.uint8)  # The exact initialization operation in DeepSpeed 0.19.9.
    assert len(conv._forward_pre_hooks) == 1
    x = torch.randn(1, 3, 4, 4, 4, dtype=torch.bfloat16)
    prepared = next(iter(conv._forward_pre_hooks.values()))(conv, (x,))[0]
    assert prepared.is_contiguous(memory_format=torch.channels_last_3d)
    torch.testing.assert_close(conv(prepared), torch.nn.functional.conv3d(x, values, conv.bias))


def test_audit_reports_layout_views_and_aliases_without_mutation():
    model = nn.Module()
    source = torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4)
    model.p = nn.Parameter(source.permute(0, 2, 1))
    model.tied = model.p
    model.offset = nn.Parameter(source.flatten()[1:5])
    before = model.p.detach().clone()
    rows = {row["parameter_name"]: row for row in audit_parameter_layout(model)}
    assert rows["p"]["stride"][-1] == 4
    assert rows["p"]["shared_storage"] == ["p", "tied", "offset"]
    assert rows["offset"]["storage_offset"] == 1
    assert torch.equal(model.p, before)


def test_audit_detects_nonunit_stride_even_when_contiguous():
    model = nn.Module()
    model.p = nn.Parameter(torch.empty_strided((2, 1), (1, 3), dtype=torch.bfloat16))
    assert model.p.is_contiguous()
    assert audit_parameter_layout(model)[0]["stride"] == (1, 3)
    with pytest.raises(RuntimeError, match="stride"):
        model.p.view(torch.uint8)


def test_local_checkpoint_preserves_gradients_without_replaying_module_hooks():
    from copy import deepcopy
    from unittest.mock import patch

    from gr00t.model.extension.multilevel_local_sidecar import LocalJointActionStage

    torch.manual_seed(17)
    plain = LocalJointActionStage(8, 6, 8, 2, patch_size=4, window_chunk_size=2)
    with torch.no_grad():
        plain.action_gate.fill_(0.3)  # Exercise every branch, beyond zero-gate initialization.
    checked = deepcopy(plain)
    calls = []
    checked.register_forward_pre_hook(lambda *args: calls.append("gather"))
    checked.register_forward_hook(lambda *args: calls.append("release"))
    action = torch.randn(2, 3, 8, requires_grad=True)
    feat = torch.randn(14, 6, requires_grad=True)
    level = dict(
        feat=feat,
        coord=torch.randn(14, 3),
        offset=torch.tensor([6, 14]),
        serialized_order=torch.arange(14)[None],
    )
    reference, _ = plain(action, level)
    expected = torch.autograd.grad(reference.square().sum(), [action, feat, *plain.parameters()])
    with patch.object(checked, "_forward", wraps=checked._forward) as recompute:
        actual, _ = checked(action, level, checkpoint_activations=True)
        actual_grad = torch.autograd.grad(
            actual.square().sum(), [action, feat, *checked.parameters()]
        )
        assert recompute.call_count == 2
    torch.testing.assert_close(actual, reference)
    for value, ref in zip(actual_grad, expected):
        assert torch.isfinite(value).all()
        torch.testing.assert_close(value, ref)
    # ZeRO hooks belong outside the recompute region, including its post-forward release.
    assert calls == ["gather", "release"]

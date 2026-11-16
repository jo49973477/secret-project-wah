"""Small real Qwen3-VL checkpoint tests; no downloads or GPU required."""

import json
from pathlib import Path

from gr00t.model.gr00t_n1d7.pretrained_loading import (
    backbone_is_covered,
    checkpoint_backbone_shapes,
    finalize_pretrained_model,
    initialize_missing_sidecar,
)
from gr00t.model.modules.qwen3_backbone import Qwen3Backbone
import pytest
from safetensors.torch import save_file
import torch
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
)


class TinyConfig(PretrainedConfig):
    model_type = "test_loading"

    def __init__(self, vlm_path="", sidecar=False, load_bf16=False, tune_llm=False, **kwargs):
        super().__init__(**kwargs)
        self.vlm_path = vlm_path
        self.sidecar = sidecar
        self.load_bf16 = load_bf16
        self.tune_llm = tune_llm
        self.sidecar_hidden_dim = 8
        self.sidecar_num_point_tokens = 3
        self.sidecar_num_heads = 2
        self.contact_hidden_dim = 8
        self.contact_num_tokens = 2
        self.freeze_contact_teacher = True


class TinyModel(PreTrainedModel):
    config_class = TinyConfig

    def __init__(self, config, checkpoint_backbone_shapes=None, checkpoint_bf16_shapes=None):
        super().__init__(config)
        self.backbone = Qwen3Backbone(
            model_name=config.vlm_path,
            select_layer=1,
            load_bf16=config.load_bf16,
            tune_llm=config.tune_llm,
            trainable_params_fp32=True,
            checkpoint_backbone_shapes=checkpoint_backbone_shapes,
        )
        from gr00t.model.modules.dit import AlternateVLDiT
        from tests.gr00t.model.test_sidecar_dit import CFG, model

        self.action_head = torch.nn.Module()
        self.action_head.model = model() if config.sidecar else AlternateVLDiT(**CFG)
        self.post_init()
        if checkpoint_bf16_shapes:
            from gr00t.model.gr00t_n1d7.pretrained_loading import prepare_bf16_checkpoint_parameters

            prepare_bf16_checkpoint_parameters(self, checkpoint_bf16_shapes)


@pytest.fixture(autouse=True)
def real_weights(monkeypatch):
    monkeypatch.setenv("GROOT_SKIP_HF_MODEL_WEIGHTS", "0")


@pytest.fixture
def checkpoint(tmp_path):
    config = Qwen3VLConfig(
        text_config=dict(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=32,
            rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1]},
        ),
        vision_config=dict(
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            out_hidden_size=16,
            deepstack_visual_indexes=[],
        ),
        tie_word_embeddings=False,
    )
    vlm = tmp_path / "vlm"
    Qwen3VLForConditionalGeneration(config).save_pretrained(vlm)
    outer = tmp_path / "outer"
    wrapper = TinyModel(TinyConfig(vlm_path=str(vlm)))
    # The complete checkpoint must win over the external VLM checkpoint.
    with torch.no_grad():
        for parameter in wrapper.backbone.parameters():
            parameter.add_(0.125)
    wrapper.save_pretrained(outer)
    return outer


@pytest.mark.parametrize("load_bf16,tune_llm", [(False, False), (True, False), (True, True)])
def test_real_hf_nested_load_parity(checkpoint, monkeypatch, load_bf16, tune_llm):
    legacy = TinyModel.from_pretrained(checkpoint, load_bf16=load_bf16, tune_llm=tune_llm)
    shapes = checkpoint_backbone_shapes(str(checkpoint), {"local_files_only": True})

    def forbidden(*args, **kwargs):
        raise AssertionError("external pretrained weights must not be loaded")

    monkeypatch.setattr(Qwen3VLForConditionalGeneration, "from_pretrained", forbidden)
    optimized = TinyModel.from_pretrained(
        checkpoint,
        checkpoint_backbone_shapes=shapes,
        load_bf16=load_bf16,
        tune_llm=tune_llm,
    )
    finalize_pretrained_model(optimized)
    for name, value in legacy.state_dict().items():
        assert value.dtype == optimized.state_dict()[name].dtype, name
        assert torch.equal(value, optimized.state_dict()[name]), name
    from safetensors import safe_open

    with safe_open(checkpoint / "model.safetensors", framework="pt") as reader:
        for name, tensor in optimized.state_dict().items():
            assert torch.equal(tensor, reader.get_tensor(name).to(tensor.dtype)), name
    legacy_buffers = dict(legacy.named_buffers())
    for name, buffer in optimized.named_buffers():
        if "inv_freq" in name:
            assert torch.equal(buffer, legacy_buffers[name]), name
    for module in optimized.backbone.model.visual.modules():
        if isinstance(module, torch.nn.Conv3d):
            assert module.weight.is_contiguous()
            assert (
                module.weight.to(torch.bfloat16).view(torch.uint8).numel()
                == module.weight.numel() * 2
            )
            assert len(module._forward_pre_hooks) == 1
    for a, b in zip(legacy.parameters(), optimized.parameters()):
        assert a.requires_grad == b.requires_grad
    assert not any(t.is_meta for t in list(optimized.parameters()) + list(optimized.buffers()))
    # Run actual text attention and backpropagation through the loaded model.
    optimized.backbone.model.requires_grad_(True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=load_bf16):
        result = optimized.backbone.model(input_ids=torch.tensor([[1, 2, 3]]))
        loss = result.logits.float().square().mean()
    loss.backward()
    assert any(p.grad is not None for p in optimized.parameters())


def test_incomplete_coverage_falls_back(checkpoint, monkeypatch):
    shapes = checkpoint_backbone_shapes(str(checkpoint), {})
    shapes.pop(next(iter(shapes)))
    original = Qwen3VLForConditionalGeneration.from_pretrained
    calls = []

    def tracked(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(Qwen3VLForConditionalGeneration, "from_pretrained", tracked)
    model = TinyModel.from_pretrained(checkpoint, checkpoint_backbone_shapes=shapes)
    finalize_pretrained_model(model)
    assert len(calls) == 1


def test_headers_and_shape_validation(tmp_path):
    model = torch.nn.Linear(3, 2)
    save_file({"backbone.model.weight": model.weight.detach()}, tmp_path / "one.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "backbone.model.weight": "one.safetensors",
                    "backbone.model.bias": "one.safetensors",
                }
            }
        )
    )
    shapes = checkpoint_backbone_shapes(str(tmp_path), {})
    assert shapes == {"weight": (2, 3)}
    assert not backbone_is_covered(model, shapes)
    assert not backbone_is_covered(model, {"weight": (3, 2), "bias": (2,)})
    assert backbone_is_covered(model, {"weight": (2, 3), "bias": (2,)})


@pytest.mark.parametrize("optimized", [False, True])
def test_new_sidecar_initialization_and_dit_integrity(checkpoint, optimized):
    from gr00t.model.gr00t_n1d7.dit_integrity import compare_shared_tensors, validate_loading_info
    from gr00t.model.gr00t_n1d7.sensor_loading import SIDECAR_ADAPTER_PREFIXES
    from tests.gr00t.model.test_sidecar_dit import inputs

    reference = TinyModel.from_pretrained(checkpoint)
    shapes = checkpoint_backbone_shapes(str(checkpoint), {}) if optimized else None
    loaded, info = TinyModel.from_pretrained(
        checkpoint,
        sidecar=True,
        checkpoint_backbone_shapes=shapes,
        output_loading_info=True,
    )
    assert validate_loading_info(info, SIDECAR_ADAPTER_PREFIXES)["status"] == "PASS"
    initialize_missing_sidecar(loaded, info["missing_keys"])
    finalize_pretrained_model(loaded)
    assert compare_shared_tensors(reference.state_dict(), loaded.state_dict())["status"] == "PASS"
    dit = loaded.action_head.model
    for stage in dit.sidecar_stages:
        assert stage.action_gate.item() == 0 and stage.action_gate.requires_grad
        assert stage.point_gate.item() == 1
    assert all(stage.scale.item() == 0 for stage in dit.contact_injections)
    assert not any(p.requires_grad for p in dit.transformer_blocks.parameters())
    assert not any(p.requires_grad for p in dit.contact_adapter.projector.parameters())
    assert dit.sidecar_pool.queries.isfinite().all() and dit.sidecar_pool.queries.std() > 0
    dit(**inputs()).square().mean().backward()
    assert all(stage.action_gate.grad is not None for stage in dit.sidecar_stages)


def test_sharded_checkpoint_parity(checkpoint, tmp_path):
    reference = TinyModel.from_pretrained(checkpoint)
    sharded = tmp_path / "sharded"
    reference.save_pretrained(sharded, max_shard_size="8KB")
    assert (sharded / "model.safetensors.index.json").exists()
    shapes = checkpoint_backbone_shapes(str(sharded), {})
    loaded = TinyModel.from_pretrained(sharded, checkpoint_backbone_shapes=shapes)
    finalize_pretrained_model(loaded)
    assert all(
        torch.equal(tensor, loaded.state_dict()[name])
        for name, tensor in reference.state_dict().items()
    )


def test_memory_diagnostics(monkeypatch, caplog):
    import logging
    import os

    from gr00t.utils.model_load_diagnostics import log_model_load_memory

    monkeypatch.setenv("GROOT_DEBUG_MODEL_LOAD_MEMORY", "1")
    monkeypatch.setenv("RANK", "7")
    with caplog.at_level(logging.INFO):
        log_model_load_memory("test stage")
    assert f"rank=7 pid={os.getpid()}" in caplog.text
    assert "RSS=" in caplog.text and "peak=" in caplog.text


def test_memory_comparison(checkpoint):
    """Isolated processes: report peaks without asserting a tiny-model saving."""
    import os
    import subprocess
    import sys

    results = {}
    for mode in ("legacy", "optimized"):
        output = subprocess.check_output(
            [sys.executable, __file__, str(checkpoint), mode],
            env={
                **os.environ,
                "GROOT_DEBUG_MODEL_LOAD_MEMORY": "1",
                "PYTHONPATH": str(Path(__file__).resolve().parents[3])
                + os.pathsep
                + os.environ.get("PYTHONPATH", ""),
            },
            timeout=60,
            text=True,
        )
        results[mode] = json.loads(output.strip().splitlines()[-1])
    print("Isolated tiny-Qwen RSS comparison (KiB):", json.dumps(results))
    assert all(result["peak_kib"] > 0 for result in results.values())


if __name__ == "__main__":
    import logging
    import sys

    from gr00t.utils.model_load_diagnostics import log_model_load_memory, model_load_memory_kib

    logging.basicConfig(level=logging.INFO)
    path, mode = sys.argv[1:]
    log_model_load_memory("before model construction")
    shapes = checkpoint_backbone_shapes(path, {}) if mode == "optimized" else None
    loaded = TinyModel.from_pretrained(path, checkpoint_backbone_shapes=shapes)
    finalize_pretrained_model(loaded)
    log_model_load_memory("after GR00T checkpoint loading (tiny fixture)")
    print(json.dumps(model_load_memory_kib()))


def test_actual_shard_timings_include_materialization(checkpoint, tmp_path, monkeypatch):
    from gr00t.utils.model_load_diagnostics import trace_checkpoint_loading

    reference = TinyModel.from_pretrained(checkpoint)
    sharded = tmp_path / "traced-shards"
    reference.save_pretrained(sharded, max_shard_size="8KB")
    monkeypatch.setenv("GROOT_DEBUG_MODEL_LOAD_IO", "1")
    monkeypatch.setenv("GROOT_MODEL_LOAD_TRACE_DIR", str(tmp_path / "trace"))
    shapes = checkpoint_backbone_shapes(str(sharded), {})
    with trace_checkpoint_loading():
        loaded = TinyModel.from_pretrained(sharded, checkpoint_backbone_shapes=shapes)
    finalize_pretrained_model(loaded)
    events = [
        json.loads(line)
        for path in (tmp_path / "trace").glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    shards = [
        event
        for event in events
        if event["event"] == "exit" and event["phase"] == "transformers.load_shard_file"
    ]
    expected = set(
        json.loads((sharded / "model.safetensors.index.json").read_text())["weight_map"].values()
    )
    assert {Path(event["checkpoint"]).name for event in shards} == expected
    assert all(event["file_size_bytes"] > 0 and event["elapsed_seconds"] >= 0 for event in shards)
    assert all(
        torch.equal(tensor, loaded.state_dict()[name])
        for name, tensor in reference.state_dict().items()
    )


def test_source_bf16_storage_preserves_pretrained_values_sidecar_and_rng(checkpoint, tmp_path):
    from gr00t.model.gr00t_n1d7.dit_integrity import compare_shared_tensors
    from gr00t.model.gr00t_n1d7.sensor_loading import SIDECAR_ADAPTER_PREFIXES

    source = TinyModel.from_pretrained(checkpoint)
    source.action_head.model.bfloat16()
    # Mixed checkpoints must retain source FP32 values, not round them to BF16.
    source.action_head.model.proj_out_2.float()
    with torch.no_grad():
        source.action_head.model.proj_out_2.weight.fill_(0.1234567)
        source.action_head.model.proj_out_2.bias.fill_(0.314159)
    bf16_checkpoint = tmp_path / "source-bf16"
    source.save_pretrained(bf16_checkpoint, max_shard_size="8KB")
    bf16_shapes = {}
    shapes = checkpoint_backbone_shapes(str(bf16_checkpoint), {}, bf16_parameter_shapes=bf16_shapes)
    assert bf16_shapes and all(name.startswith("action_head.") for name in bf16_shapes)

    def load(metadata):
        torch.manual_seed(71)
        loaded, info = TinyModel.from_pretrained(
            bf16_checkpoint,
            sidecar=True,
            checkpoint_backbone_shapes=shapes,
            checkpoint_bf16_shapes=metadata,
            output_loading_info=True,
        )
        initialize_missing_sidecar(loaded, info["missing_keys"])
        finalize_pretrained_model(loaded)
        return loaded, torch.random.get_rng_state()

    legacy, legacy_rng = load(None)
    optimized, optimized_rng = load(bf16_shapes)
    assert torch.equal(legacy_rng, optimized_rng)
    assert compare_shared_tensors(source.state_dict(), optimized.state_dict())["status"] == "PASS"
    for name, parameter in optimized.named_parameters():
        expected = dict(legacy.named_parameters())[name]
        assert parameter.requires_grad == expected.requires_grad
        if name in bf16_shapes:
            assert parameter.dtype == torch.bfloat16
            assert torch.equal(parameter, source.state_dict()[name])
        elif name.startswith(SIDECAR_ADAPTER_PREFIXES):
            assert parameter.dtype == expected.dtype == torch.float32
            assert torch.equal(parameter, expected), name
        else:
            assert parameter.dtype == expected.dtype
            assert torch.equal(parameter, expected), name


def test_bf16_preparation_skips_real_parameters_buffers_and_active_zero3(monkeypatch):
    from gr00t.model.gr00t_n1d7.pretrained_loading import prepare_bf16_checkpoint_parameters
    import transformers.integrations

    root = torch.nn.Module()
    root.action_head = torch.nn.Module()
    with torch.device("meta"):
        root.action_head.layer = torch.nn.Linear(3, 2)
        root.action_head.layer.register_buffer("runtime_buffer", torch.ones(2))
    shapes = {"action_head.layer.weight": (2, 3), "action_head.layer.bias": (99,)}
    assert prepare_bf16_checkpoint_parameters(root, shapes) == 1
    assert root.action_head.layer.weight.dtype == torch.bfloat16
    assert (
        root.action_head.layer.bias.dtype
        == root.action_head.layer.runtime_buffer.dtype
        == torch.float32
    )
    real = torch.nn.Module()
    real.action_head = torch.nn.Linear(3, 2)
    original = real.action_head.weight.clone()
    assert prepare_bf16_checkpoint_parameters(real, {"action_head.weight": (2, 3)}) == 0
    assert torch.equal(real.action_head.weight, original)
    monkeypatch.setattr(transformers.integrations, "is_deepspeed_zero3_enabled", lambda: True)
    assert prepare_bf16_checkpoint_parameters(root, {"action_head.layer.bias": (2,)}) == 0
    assert root.action_head.layer.bias.dtype == torch.float32

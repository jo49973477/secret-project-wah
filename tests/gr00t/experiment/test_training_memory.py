"""CPU coverage for one-shot startup memory and DeepSpeed state diagnostics."""

from types import SimpleNamespace

from gr00t.experiment import trainer as trainer_module
from gr00t.experiment.training_memory import log_deepspeed_engine_state
import pytest
import torch


def _fake_deepspeed_optimizer(*, optimizer_offload=True, param_offload=False, subgroup=20_000_000):
    cpu_adam = type("DeepSpeedCPUAdam", (), {})()
    stage3 = SimpleNamespace(
        offload_optimizer=optimizer_offload,
        offload_param=param_offload,
        sub_group_size=subgroup,
        optimizer=cpu_adam,
        fp32_partitioned_groups_flat=[torch.zeros(1)],
        parameter_offload=SimpleNamespace(offload_device="cpu" if param_offload else None),
    )
    return SimpleNamespace(optimizer=stage3)


def test_initialized_deepspeed_profile_is_verified() -> None:
    log_deepspeed_engine_state(
        _fake_deepspeed_optimizer(),
        expected_optimizer_offload=True,
        expected_param_offload=False,
        expected_sub_group_size=20_000_000,
    )


def test_initialized_deepspeed_profile_mismatch_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="optimizer_offload expected=True actual=False"):
        log_deepspeed_engine_state(
            _fake_deepspeed_optimizer(optimizer_offload=False),
            expected_optimizer_offload=True,
            expected_param_offload=False,
            expected_sub_group_size=20_000_000,
        )


def test_deepspeed_optimizer_step_snapshots_are_one_shot(monkeypatch) -> None:
    phases = []
    monkeypatch.setattr(trainer_module, "log_memory_snapshot", phases.append)
    calls = []

    class _ZeroOptimizer:
        def step(self):
            calls.append("step")
            return 17

    zero_optimizer = _ZeroOptimizer()
    model = SimpleNamespace(optimizer=zero_optimizer)
    trainer = object.__new__(trainer_module.Gr00tTrainer)

    trainer._install_first_deepspeed_optimizer_step_diagnostics(model)
    assert zero_optimizer.step() == 17
    assert zero_optimizer.step() == 17

    assert calls == ["step", "step"]
    assert phases == [
        "after_first_backward",
        "before_first_optimizer_step",
        "after_first_optimizer_step",
    ]


def test_gradient_audit_survives_first_step_memory_wrapper(monkeypatch):
    calls = []
    monkeypatch.setattr(trainer_module, "log_memory_snapshot", lambda *args: None)
    optimizer = SimpleNamespace(step=lambda: calls.append("optimizer"))
    engine = SimpleNamespace(optimizer=optimizer)

    class Audit:
        installed = False

        def install(self, model):
            if self.installed:
                return
            self.installed = True
            original = model.optimizer.step

            def audited():
                calls.append("audit")
                original()

            model.optimizer.step = audited

        def record_loss(self, loss):
            assert torch.isfinite(loss)

    def training_step(self, model, *args):
        model.optimizer.step()
        return torch.tensor(1.0)

    monkeypatch.setattr(trainer_module.Trainer, "training_step", training_step)
    trainer = object.__new__(trainer_module.Gr00tTrainer)
    trainer.is_deepspeed_enabled = True
    trainer.multimodal_monitor = None
    trainer.zero3_gradient_audit = Audit()
    trainer.training_step(engine, {})
    trainer.training_step(engine, {})
    assert calls == ["audit", "optimizer", "audit", "optimizer"]

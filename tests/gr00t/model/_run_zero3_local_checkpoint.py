"""Two-rank BF16 gradient parity and accumulation regression for a ZeRO leaf."""

from copy import deepcopy
import os
import tempfile

import deepspeed
from gr00t.experiment.zero3_debug import Zero3GradientAudit
from gr00t.model.extension.multilevel_local_sidecar import LocalJointActionStage
import torch
from torch import nn
import torch.distributed as dist


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(8, 8)
        self.stage = LocalJointActionStage(8, 6, 8, 2, 4, 2)
        with torch.no_grad():
            self.stage.action_gate.fill_(0.3)

    def forward(self, action, level, checkpointed=True):
        return self.stage(self.input(action), level, checkpoint_activations=checkpointed)[0]


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    deepspeed.init_distributed("nccl")
    torch.manual_seed(19)
    model = Model().cuda().bfloat16()
    reference = deepcopy(model)
    audit_dir = tempfile.TemporaryDirectory(prefix="wah-zero3-grad-")
    audit = Zero3GradientAudit(model, audit_dir.name)
    expected_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        config={
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 2,
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": 3,
                "stage3_param_persistence_threshold": 0,
                "overlap_comm": False,
            },
            "zero_allow_untested_optimizer": True,
        },
    )
    from deepspeed.utils import safe_get_full_grad

    audit.install(engine)

    # Two optimizer steps, each accumulating two distinct microbatches.
    for step in range(2):
        expected_optimizer.zero_grad()
        for micro in range(2):
            torch.manual_seed(100 + 2 * step + micro)
            action = torch.randn(1, 3, 8, device="cuda", dtype=torch.bfloat16)
            level = dict(
                feat=torch.randn(9, 6, device="cuda", dtype=torch.bfloat16),
                coord=torch.randn(9, 3, device="cuda"),
                offset=torch.tensor([9], device="cuda"),
                serialized_order=torch.arange(9, device="cuda")[None],
            )
            expected = reference(action, level, checkpointed=False)
            actual = engine(action, level)
            torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
            (expected.float().square().mean() / 2).backward()
            engine.backward(actual.float().square().mean())
            if engine.is_gradient_accumulation_boundary():
                for (name, parameter), (_, ref) in zip(
                    model.named_parameters(), reference.named_parameters()
                ):
                    gradient = safe_get_full_grad(parameter)
                    assert ref.grad is not None and gradient is not None, name
                    assert torch.isfinite(gradient).all(), name
                    torch.testing.assert_close(
                        gradient.float(), ref.grad.float(), atol=0.005, rtol=0.08, msg=name
                    )
            engine.step()
        expected_optimizer.step()
    print(
        f"ZERO3_LOCAL_CHECKPOINT_PARITY rank={dist.get_rank()} optimizer_steps=2 accumulation=2 PASS",
        flush=True,
    )
    dist.barrier()
    dist.destroy_process_group()
    audit_dir.cleanup()


if __name__ == "__main__":
    main()

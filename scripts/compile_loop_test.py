"""End-to-end test of TrainingLoop.train() with a compiled whole-step.

Exercises the real driver: base train() -> compiled_train_step branch ->
compiled forward+losses+backward+optimizer, with global_step/pkbar/writer
bookkeeping, plus SaveLoadModel save/load round-trip through the compiled
forward.

Usage: venv/bin/python scripts/compile_loop_test.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Lib"))

from hrothgar.utils import SaveLoadModel, TrainingLoop

torch.manual_seed(0)
torch.set_float32_matmul_precision("high")


class FakeArgs:
    tag = "compile-loop-test"
    model_path = "/tmp/compile_loop_test.pth"
    allow_dirty = True
    target_steps = 3


class FakeModel(SaveLoadModel):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(8, 8)

    def forward(self, x):
        return self.net(x)


class FakeLoader:
    def __init__(self):
        self.batches = [torch.rand(4, 8) for _ in range(4)]

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


class FakeLoop(TrainingLoop):
    def post_init(self, train_args):
        self.model = FakeModel().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.train_loader = FakeLoader()
        self.num_epochs = 1
        self.target_steps = train_args.target_steps
        self.validation_direction = "lower"
        # Mirror the style_extraction wiring.
        self.model.forward = torch.compile(self.model.forward)
        self.compiled_train_step = torch.compile(self._compiled_train_step)
        self.step_count = 0

    def _compute_losses(self, batch):
        out = self.model(batch.to(self.device))
        loss = out.square().mean() + self.model.net.weight.square().mean() * 1e-2
        return loss, {"loss": loss.detach()}

    def train_step(self, batch):
        return self._compute_losses(batch)

    def _compiled_train_step(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        total, terms = self._compute_losses(batch)
        total.backward()
        self.optimizer.step()
        return total.detach(), {k: v.detach() for k, v in terms.items()}

    def post_train_step(self):
        self.step_count += 1
        if self.global_step % 2 == 0 and self.global_step > 0:
            # Exercise save/load through the compiled-forward model.
            self.model.save(self.model_path)
            fresh = FakeModel().to(self.device)
            fresh.load(self.model_path, device=self.device, strict=False)
            assert torch.allclose(
                fresh.net.weight, self.model.net.weight
            ), "checkpoint round-trip mismatch"


loop = FakeLoop(FakeArgs())
loop.train()
assert (
    loop.global_step == FakeArgs.target_steps
), f"expected {FakeArgs.target_steps} steps, got {loop.global_step}"
print(
    f"OK: {loop.global_step} steps through the compiled whole-step; save/load round-trip intact"
)

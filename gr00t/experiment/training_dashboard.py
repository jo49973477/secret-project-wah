"""Rank-zero Trainer rendering, with a coordinated Rich display or clean text."""

from __future__ import annotations

import atexit
import logging
import math
import sys
import time
from typing import Any

from transformers.trainer_callback import ProgressCallback, TrainerCallback


class QuietProgressCallback(ProgressCallback):
    """Keep HF's tqdm train/eval bars while removing dictionary printing at source."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        return control

    def close(self):
        for name in ("training_bar", "prediction_bar"):
            bar = getattr(self, name, None)
            if bar is not None:
                bar.close()
                setattr(self, name, None)


class TrainingDashboardCallback(TrainerCallback):
    """Display already logged CPU scalars; never touch model tensors."""

    LOSS = {
        "loss/total": "Total",
        "loss/action": "Action (contribution)",
        "loss/contact_weighted": "Contact (weighted)",
        "loss/contact_raw": "Contact (raw)",
        "loss/zero_prediction_mse": "Zero Baseline",
        "loss/action_to_zero_ratio": "Action / Zero",
    }
    GRADIENT = {
        "grad_status/sample_step": "Module sample step",
        "grad_norm/pre_clip": "Global Pre",
        "grad_norm/post_clip": "Global Post",
        **{
            f"grad_norm/{name}": name.replace("_", " ").title()
            for name in (
                "dit",
                "lora",
                "sidecar",
                "temporal",
                "moe",
                "point_encoder",
                "tactile_encoder",
                "contact_teacher",
                "contact_predictor",
                "contact_injection",
                "gates",
            )
        },
    }
    ACTION = {
        "action/predicted_velocity_std": "Predicted Std",
        "action/target_velocity_std": "Target Std",
        "action/valid_action_ratio": "Valid Ratio",
        "action/valid_dimension_count": "Valid Dimensions",
        "action/valid_timestep_ratio": "Valid Timesteps",
    }
    TRAINING = {
        "samples_seen": "Samples Seen",
        "effective_epoch": "Effective Epoch",
        "learning_rate": "Learning Rate",
        "clip_limit": "Gradient Clip",
        "precision": "Precision",
        "distributed": "Distributed",
        "system/gpu_memory_allocated": "GPU GiB",
        "system/steps_per_second": "Steps/s",
    }
    DIAGNOSTICS = {
        "contact/observed_variance": "Teacher Variance",
        "contact/observed_norm": "Teacher Norm",
        "contact/predicted_variance": "Predictor Variance",
        "contact/surprise_norm": "Surprise Norm",
        "contact/gate_mean": "Contact Gate",
        "gate/action_value_mean": "Action Gate Mean",
        "gate/action_value_std": "Action Gate Std",
        "gate/action_grad_norm": "Action Gate Grad",
        "gate/action_update_norm": "Action Pre-Gate Update",
        "gate/point_value_mean": "Point Gate Mean",
        "gate/point_value_std": "Point Gate Std",
        "gate/point_grad_norm": "Point Gate Grad",
        "gate/point_update_norm": "Point Pre-Gate Update",
        "sidecar/action_residual_ratio": "Action Residual",
        "sidecar/point_residual_ratio": "Point Residual",
        "sidecar/action_pre_gate_update_norm": "Action Pre-Gate Norm",
        "sidecar/action_gated_residual_norm": "Action Gated Norm",
        "sidecar/action_actual_residual_norm": "Action Applied Norm",
        "sidecar/action_bf16_loss_norm": "Action BF16 Loss",
    }

    def __init__(
        self,
        enabled: bool | None = None,
        refresh_seconds: float = 1.0,
        diagnostics_enabled: bool = False,
        *,
        experiment_name: str | None = None,
        screen: bool = True,
    ):
        if not math.isfinite(refresh_seconds) or refresh_seconds <= 0:
            raise ValueError("refresh_seconds must be finite and positive")
        self.enabled = sys.stdout.isatty() and enabled is not False
        self.experiment_name = experiment_name
        self.screen = screen
        self.diagnostics_enabled = diagnostics_enabled
        self.refresh_seconds = refresh_seconds
        self._live = None
        self._progress = None
        self._task = None
        self._eval_task = None
        self._metrics: dict[str, Any] = {}
        self._started = 0.0
        self._last_render = 0.0
        self._initial_step = 0
        self._training = False
        self._redirected_handlers = []

    @staticmethod
    def _is_main(args: Any) -> bool:
        return getattr(args, "process_index", 0) == 0

    def on_train_begin(self, args, state, control, **kwargs):
        self._training = True
        self._started = time.monotonic()
        self._initial_step = state.global_step
        self._metrics.clear()
        self._status = "Running"
        self._checkpoint = "None"
        deepspeed = getattr(args, "deepspeed", None)
        stage = (
            deepspeed.get("zero_optimization", {}).get("stage")
            if isinstance(deepspeed, dict)
            else None
        )
        distribution = (
            f"ZeRO-{stage}" if stage is not None else "DeepSpeed" if deepspeed else "DDP/local"
        )
        self._metrics.update(
            clip_limit=getattr(args, "max_grad_norm", None),
            precision="BF16"
            if getattr(args, "bf16", False)
            else "FP16"
            if getattr(args, "fp16", False)
            else "FP32",
            distributed=f"{distribution} / {getattr(args, 'world_size', 1)} GPUs",
        )
        if not self._is_main(args) or not self.enabled:
            return control
        if self._live is None:
            self._start_live()
        total = max(int(getattr(state, "max_steps", 0) or 0), 1)
        if self._task is None:
            self._task = self._progress.add_task(
                "Training", total=total, completed=state.global_step, speed="0.00 step/s"
            )
        else:
            self._progress.reset(self._task, total=total, completed=state.global_step)
        self._refresh(force=True)
        return control

    def _start_live(self):
        from rich.console import Console, Group
        from rich.live import Live
        from rich.panel import Panel
        from rich.progress import (
            BarColumn,
            Progress,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )
        from rich.table import Table

        self._Table, self._Panel, self._Group = Table, Panel, Group
        self._console = Console(file=sys.stdout)
        self._progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TextColumn("{task.completed:.0f}/{task.total:.0f}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[speed]}", style="dim"),
            console=self._console,
            auto_refresh=False,
        )
        self._wide_columns = self._progress.columns
        self._live = Live(
            self,
            console=self._console,
            refresh_per_second=1 / self.refresh_seconds,
            screen=self.screen,
            transient=not self.screen,
            vertical_overflow="crop",
            redirect_stdout=True,
            redirect_stderr=True,
        )
        old_streams = (sys.stdout, sys.stderr)
        self._live.start()
        # Existing StreamHandlers keep a reference to the old stream; simply
        # replacing sys.stderr would let them write through the live display.
        loggers = [logging.getLogger()] + [
            logger
            for logger in logging.Logger.manager.loggerDict.values()
            if isinstance(logger, logging.Logger)
        ]
        seen = set()
        for logger in loggers:
            for handler in logger.handlers:
                if id(handler) in seen or not isinstance(handler, logging.StreamHandler):
                    continue
                seen.add(id(handler))
                for old, redirected in zip(old_streams, (sys.stdout, sys.stderr)):
                    if handler.stream is old:
                        handler.setStream(redirected)
                        self._redirected_handlers.append((handler, old, redirected))
                        break
        atexit.register(self._stop_live)

    def _stop_live(self):
        if self._live is not None:
            try:
                self._live.stop()
            finally:
                for handler, original, redirected in self._redirected_handlers:
                    if handler.stream is redirected:
                        handler.setStream(original)
                self._redirected_handlers.clear()
                self._live = None
                self._progress = None
                self._task = self._eval_task = None
                atexit.unregister(self._stop_live)

    @staticmethod
    def _format(value):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return "N/A"
        return f"{value:.6g}"

    def _panel(self, title, labels):
        table = self._Table.grid(padding=(0, 2))
        table.add_column(style="cyan")
        table.add_column(justify="right")
        for key, label in labels.items():
            value = self._metrics.get(key)
            text = value if isinstance(value, str) else self._format(value)
            if key == "samples_seen" and isinstance(value, (int, float)):
                text = f"{int(value):,}"
            elif key == "effective_epoch" and isinstance(value, (int, float)):
                text = f"{value:.2f}"
            if key.startswith("grad_norm/") and self._metrics.get(
                f"grad_status/{key.split('/')[-1]}/frozen"
            ):
                text = "N/A (Frozen)"
            table.add_row(label, text)
        return self._Panel(table, title=title)

    def __rich_console__(self, console, options):
        # Live invokes this on refresh, so resize/elapsed time also work while saving.
        yield self._render()

    def _render(self):
        import os

        from rich.table import Table
        from rich.text import Text

        width, height = self._console.size
        self._progress.columns = (
            self._wide_columns
            if width >= 80
            else tuple(self._wide_columns[i] for i in (1, 2, 3, 5))
        )
        panels = [
            Text(
                f"Florence | {self.experiment_name or os.environ.get('GR00T_EXPERIMENT_NAME', 'Training')}",
                style="bold cyan",
                no_wrap=True,
                overflow="ellipsis",
            ),
            self._progress,
        ]
        if self._training:
            panels.append(
                Text(
                    f"{self._status} | Checkpoint: {self._checkpoint}",
                    no_wrap=True,
                    overflow="ellipsis",
                )
            )
            if width < 80 or height < 20:
                # A compact view fits narrow SSH panes without stacking four panels.
                essential = {
                    "loss/total": "LOSS Total",
                    "loss/action": "Action",
                    "samples_seen": "Samples Seen",
                    "effective_epoch": "Effective Epoch",
                    "loss/contact_weighted": "Contact",
                    "loss/zero_prediction_mse": "Zero baseline",
                    "grad_norm/pre_clip": "GRAD Pre",
                    "grad_norm/post_clip": "Post",
                    "action/predicted_velocity_std": "ACTION Pred std",
                    "action/target_velocity_std": "Target std",
                    "action/valid_action_ratio": "Valid ratio",
                    "learning_rate": "OPTIMIZER LR",
                }
                labels = dict(list(essential.items())[: max(1, height - 6)])
                panels.append(self._panel("Metrics", labels))
            else:
                compact = height < 38
                loss = dict(list(self.LOSS.items())[:4]) if compact else self.LOSS
                gradient = (
                    {
                        key: self.GRADIENT[key]
                        for key in (
                            "grad_norm/pre_clip",
                            "grad_norm/post_clip",
                            "grad_norm/dit",
                            "grad_norm/sidecar",
                        )
                    }
                    if compact
                    else self.GRADIENT
                )
                action = dict(list(self.ACTION.items())[:3]) if compact else self.ACTION
                optimizer = self.TRAINING
                for row in (
                    (("LOSS", loss), ("GRADIENT", gradient)),
                    (("ACTION", action), ("OPTIMIZER", optimizer)),
                ):
                    table = Table.grid(expand=True, padding=(0, 1))
                    table.add_column(ratio=1)
                    table.add_column(ratio=1)
                    table.add_row(*(self._panel(title, labels) for title, labels in row))
                    panels.append(table)
                if self.diagnostics_enabled and height >= 38:
                    panels.append(self._panel("Multimodal Diagnostics", self.DIAGNOSTICS))
        evaluation = {
            key: key.removeprefix("eval_").replace("_", " ").title()
            for key in self._metrics
            if key.startswith("eval_")
        }
        if evaluation:
            panels.append(self._panel("Evaluation Metrics", evaluation))
        return self._Group(*panels)

    def _refresh(self, force=False):
        if self._live is not None and (
            force or time.monotonic() - self._last_render >= self.refresh_seconds
        ):
            self._live.update(self, refresh=True)
            self._last_render = time.monotonic()

    def on_step_begin(self, args, state, control, **kwargs):
        self._status = "Running"
        return control

    def on_save(self, args, state, control, **kwargs):
        self._checkpoint = str(state.global_step)
        self._status = "Running"
        self._refresh(force=True)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if getattr(control, "should_save", False):
            self._status = "Saving checkpoint"

        if self._progress is not None and self._task is not None:
            elapsed = max(time.monotonic() - self._started, 0.001)
            self._progress.update(
                self._task,
                completed=state.global_step,
                speed=f"{(state.global_step - self._initial_step) / elapsed:.2f} step/s",
            )
            self._refresh(force=True)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        self._status = "Running" if self._training else "Idle"
        if not self._is_main(args):
            return control
        self._metrics.update(
            {k: v for k, v in logs.items() if isinstance(v, (float, int)) or v is None}
        )
        if self._live is not None:
            self._refresh()
        else:
            elapsed = max(time.monotonic() - self._started, 0.001)
            completed = max(state.global_step - self._initial_step, 0)
            max_steps = int(getattr(state, "max_steps", state.global_step))
            remaining = max(max_steps - state.global_step, 0)
            eta = elapsed * remaining / completed if completed else None
            fields = (
                ("loss", "loss"),
                ("action", "loss/action"),
                ("total", "loss/total"),
                ("lr", "learning_rate"),
                ("grad", "grad_norm/pre_clip"),
                ("gpu_gib", "system/gpu_memory_allocated"),
                ("steps/s", "system/steps_per_second"),
            )
            summary = " ".join(
                f"{label}={self._format(logs[key])}"
                for label, key in fields
                if key in logs and logs[key] is not None
            )
            if summary:
                eta_text = f"{eta:.0f}s" if eta is not None else "?"
                exposure = ""
                if logs.get("samples_seen") is not None:
                    exposure += f" Samples Seen: {int(logs['samples_seen']):,}"
                if logs.get("effective_epoch") is not None:
                    exposure += f" Effective Epoch: {logs['effective_epoch']:.2f}"
                print(
                    f"Step {state.global_step}/{max_steps}{exposure} ETA {eta_text} {summary}",
                    flush=True,
                )
        return control

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if not self._is_main(args) or not self.enabled:
            return control
        if self._live is None:
            self._start_live()
        if self._eval_task is None:
            try:
                total = len(eval_dataloader)
            except TypeError:
                total = 0
            self._eval_task = self._progress.add_task("Evaluation", total=max(total, 1), speed="")
        self._progress.advance(self._eval_task)
        self._refresh()
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self._live is not None:
            if metrics:
                self._metrics.update(metrics)
            if self._eval_task is not None:
                self._progress.remove_task(self._eval_task)
                self._eval_task = None
            self._refresh(force=True)
            if not self._training:
                self._stop_live()
        return control

    def on_predict(self, args, state, control, metrics=None, **kwargs):
        return self.on_evaluate(args, state, control, metrics=metrics, **kwargs)

    def on_train_end(self, args, state, control, **kwargs):
        self._status = "Completed"
        self._refresh(force=True)
        self._stop_live()
        self._training = False
        return control

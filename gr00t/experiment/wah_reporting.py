"""Local-first reporting for inference diagnostics, independent of Trainer/W&B state."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import sys
import time
import uuid

from gr00t.experiment.wah_grounding import finite_stats


def aggregate_actions(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["comparison"], row["space"])].append(row)
    result = []
    for (task, comparison, space), group in sorted(grouped.items()):
        for metric in ("mae", "rmse", "l2"):
            stats = finite_stats([r.get(metric) for r in group])
            result.append(
                dict(
                    section="action",
                    task=task,
                    comparison=comparison,
                    space=space,
                    metric=metric,
                    **stats,
                    n_missing=len(group) - stats["n"],
                )
            )
    return result


def aggregate_stages(rows):
    """Average flow calls within frames first; n counts frames, which may be correlated."""
    metrics = (
        "point_gate",
        "action_gate",
        "feature_norm",
        "residual_norm",
        "residual_ratio",
        "cosine_similarity",
    )
    samples = defaultdict(list)
    for row in rows:
        key = (row["task"], row["condition"], row["stage"], row["kind"], row["sample"])
        samples[key].append(row)
    groups = defaultdict(list)
    for key, group in samples.items():
        sample = {
            metric: finite_stats([r.get(metric) for r in group])["mean"] for metric in metrics
        }
        sample.update(n_calls=len(group), source=group[0])
        groups[key[:-1]].append(sample)
    result = []
    for (task, condition, stage, kind), group in sorted(groups.items()):
        first = group[0]["source"]
        row = dict(
            task=task,
            condition=condition,
            stage=stage,
            kind=kind,
            block_index=first["block_index"],
            block_number=first["block_number"],
            point_stage=first.get("point_stage"),
            n_samples=len(group),
            n_forward_calls=sum(g["n_calls"] for g in group),
        )
        for metric in metrics:
            stats = finite_stats([g[metric] for g in group])
            row.update({f"{metric}_{key}": value for key, value in stats.items()})
            row[f"{metric}_n_missing"] = len(group) - stats["n"]
        result.append(row)
    return result


def _json(value):
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _atomic_text(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _csv(path, rows, default_fields):
    fields = list(dict.fromkeys([*default_fields, *(k for row in rows for k in row)]))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v, ensure_ascii=False, allow_nan=False)
                    if isinstance(v, (list, dict))
                    else "N/A"
                    if v is None
                    else v
                    for k, v in row.items()
                }
            )
    temporary.replace(path)


class DiagnosticReport:
    def __init__(self, output, config):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.summary = dict(
            schema_version=1,
            status="initializing",
            config=config,
            samples_completed=0,
            conditions_completed=0,
            errors=[],
            skipped_metrics=[],
            figures=[],
            interpretation=(
                "Action differences measure sensitivity, not successful grounding or "
                "task success. Decoded native joint units are not Cartesian coordinates."
            ),
            statistics=(
                "Population std (ddof=0); stages average flow calls within each sample first. "
                "Samples are evaluation frames and may be correlated within episodes."
            ),
        )
        self.samples, self.stages, self.geometry = [], [], []
        self.save()

    def issue(self, metric, reason, *, error=False, **context):
        entry = dict(metric=metric, reason=str(reason), **context)
        target = self.summary["errors" if error else "skipped_metrics"]
        if entry not in target:
            target.append(entry)

    def save(self):
        self.summary["action_summary"] = aggregate_actions(self.samples)
        self.summary["stage_summary"] = aggregate_stages(self.stages)
        geometry_summary = []
        for metric in (
            "object_displacement_m",
            "normalized_action_mae",
            "normalized_mae_per_m",
            "geometry_alignment_cosine",
        ):
            stats = finite_stats([r.get(metric) for r in self.geometry])
            geometry_summary.append(
                dict(
                    section="geometry",
                    task=self.summary["config"].get("task"),
                    metric=metric,
                    **stats,
                    n_missing=len(self.geometry) - stats["n"],
                )
            )
        self.summary["geometry_summary"] = geometry_summary
        summary_rows = [*self.summary["action_summary"], *geometry_summary]
        for result in self.summary.get("closed_loop_outcomes", []):
            summary_rows.append(
                dict(
                    section="closed_loop",
                    condition=result["condition"],
                    task="all",
                    **result["overall"],
                )
            )
            summary_rows.extend(
                dict(section="closed_loop", condition=result["condition"], **task)
                for task in result["tasks"]
            )
        for stage in self.summary["stage_summary"]:
            for metric in (
                "point_gate",
                "action_gate",
                "feature_norm",
                "residual_norm",
                "residual_ratio",
                "cosine_similarity",
            ):
                summary_rows.append(
                    dict(
                        section="stage",
                        metric=metric,
                        **{
                            k: stage[k]
                            for k in ("task", "condition", "stage", "kind", "block_index")
                        },
                        **{k: stage[f"{metric}_{k}"] for k in ("mean", "std", "n", "n_missing")},
                    )
                )
        for kind in ("errors", "skipped_metrics"):
            summary_rows.extend(
                dict(section=kind, status="error" if kind == "errors" else "skipped", **entry)
                for entry in self.summary[kind]
            )
        _csv(self.output / "summary.csv", summary_rows, ["section", "metric", "mean", "std", "n"])
        _csv(
            self.output / "per_sample.csv",
            self.samples,
            [
                "sample",
                "task",
                "episode",
                "step",
                "seed",
                "comparison",
                "space",
                "status",
                "mae",
                "rmse",
            ],
        )
        _csv(
            self.output / "stage_metrics.csv",
            self.stages,
            ["sample", "task", "condition", "stage", "block_index", "kind", "forward_call"],
        )
        _csv(
            self.output / "stage_summary.csv",
            self.summary["stage_summary"],
            ["task", "condition", "stage", "block_index", "kind", "n_samples", "n_forward_calls"],
        )
        _csv(self.output / "geometry.csv", self.geometry, ["sample", "status", "reason"])
        _atomic_text(
            self.output / "results.json",
            _json(dict(per_sample=self.samples, per_stage=self.stages, geometry=self.geometry)),
        )
        _atomic_text(self.output / "summary.json", _json(self.summary))

    def figures(self):
        """Each plot is optional; numeric results survive plotting failures."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            self.issue("figures", exc, error=True)
            return

        def save(name, draw):
            fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
            try:
                draw(ax)
                fig.savefig(self.output / name, dpi=150)
                self.summary["figures"].append(name)
            except Exception as exc:
                self.issue(name, exc, error=True)
            finally:
                plt.close(fig)

        action = [
            r
            for r in self.summary["action_summary"]
            if r["space"] == "normalized" and r["metric"] == "mae" and r["n"]
        ]
        if action:

            def action_plot(ax):
                ax.barh(
                    [r["comparison"] for r in action],
                    [r["mean"] for r in action],
                    xerr=[r["std"] for r in action],
                    color="#26999b",
                    capsize=3,
                )
                ax.set(
                    xlabel="Masked normalized action MAE (mean ± population std)",
                    title="Paired action sensitivity — not grounding correctness",
                )

            save("action_sensitivity.png", action_plot)
        else:
            self.issue("action_sensitivity.png", "No available paired action metrics")
        stages = [
            r
            for r in self.summary["stage_summary"]
            if r["condition"] == "original" and r["kind"] == "sidecar_action"
        ]
        for metric in ("action_gate", "point_gate", "residual_ratio", "cosine_similarity"):
            valid = [r for r in stages if r[f"{metric}_n"]]
            if not valid:
                self.issue(f"stage_{metric}.png", "No defined baseline stage measurements")
                continue

            def stage_plot(ax, metric=metric, valid=valid):
                ax.bar(
                    [f"{r['stage']} (block {r['block_index']})" for r in valid],
                    [r[f"{metric}_mean"] for r in valid],
                    yerr=[r[f"{metric}_std"] for r in valid],
                    color="#386ab3",
                    capsize=3,
                )
                ax.set(
                    xlabel="Stage (zero-based actual DiT block)",
                    ylabel=metric,
                    title=f"{metric}: per-sample mean over flow calls, then mean ± std",
                )

            save(f"stage_{metric}.png", stage_plot)
        geometry = [r for r in self.geometry if r.get("status") == "available"]
        if geometry:

            def geometry_plot(ax):
                ax.scatter(
                    [r["object_displacement_m"] for r in geometry],
                    [r["normalized_action_mae"] for r in geometry],
                    color="#ac527b",
                )
                ax.set(
                    xlabel="Metadata object displacement (m)",
                    ylabel="Normalized action MAE",
                    title="Cloud-swap sensitivity — no correctness claim",
                )

            save("geometry_sensitivity.png", geometry_plot)
        else:
            self.issue("geometry_sensitivity.png", "No valid object XYZ scene pairs")


def display_value(value):
    return "N/A" if value is None else f"{value:.5g}"


class DiagnosticDashboard:
    """Rich Live/Panel/Table conventions used by the project's training dashboard.

    No Trainer import, no global logging configuration. Non-TTY writes at most
    about ten progress messages plus time-throttled heartbeat messages.
    """

    def __init__(self, config, stream=None, tty=None):
        self.config = config
        self.stream = sys.stdout if stream is None else stream
        terminal = self.stream.isatty() if tty is None else tty
        self.tty = terminal and os.environ.get("TERM") != "dumb"
        self.live = None
        self.started = time.monotonic()
        self.last_text = 0
        self.last_count = -1

    def __enter__(self):
        if self.tty:
            from rich.console import Console
            from rich.live import Live
            from rich.progress import (
                BarColumn,
                Progress,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )

            self.console = Console(file=self.stream, force_terminal=True)
            self.progress = Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                TextColumn("{task.completed:.0f}/{task.total:.0f} samples"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
                auto_refresh=False,
            )
            self.task_id = self.progress.add_task("Initializing", total=self.config["num_samples"])
            self.live = Live(
                self.render("loading checkpoint"),
                console=self.console,
                refresh_per_second=2,
                transient=True,
                screen=False,
            )
            try:
                self.live.start()
            except BaseException:
                self.close()
                raise
        else:
            self.stream.write(
                f"WAH {self.config['experiment']} | task={self.config['task']} | "
                f"checkpoint={self.config['checkpoint']}\n"
            )
            self.stream.flush()
        return self

    def render(self, condition):
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        table = Table.grid(padding=(0, 2))
        table.add_column(style="cyan")
        table.add_column()
        for label, value in (
            ("Task", self.config["task"]),
            ("Checkpoint", self.config["checkpoint"]),
            ("Condition", condition),
        ):
            table.add_row(label, Text(str(value)))
        return Panel(
            Group(table, self.progress),
            title=f"WAH diagnostics · {self.config['experiment']}",
            border_style="cyan",
        )

    def update(self, completed, condition):
        if self.live is not None:
            self.progress.update(self.task_id, completed=completed, description="Evaluating")
            self.live.update(self.render(condition), refresh=True)
        else:
            now = time.monotonic()
            interval = max(1, self.config["num_samples"] // 10)
            if (
                completed != self.last_count
                and (completed % interval == 0 or completed == self.config["num_samples"])
            ) or now - self.last_text >= 30:
                elapsed = now - self.started
                eta = (
                    elapsed / completed * (self.config["num_samples"] - completed)
                    if completed
                    else None
                )
                self.stream.write(
                    f"Samples {completed}/{self.config['num_samples']} | "
                    f"condition={condition} | elapsed={elapsed:.0f}s | "
                    f"ETA={display_value(eta)}s\n"
                )
                self.stream.flush()
                self.last_text, self.last_count = now, completed

    def close(self):
        if self.live is not None:
            try:
                self.live.stop()
            finally:
                self.live = None

    def __exit__(self, *exc):
        self.close()

    def _final_text(self, report):
        summary = report.summary
        lines = [
            f"WAH {summary['status']} | samples={summary['samples_completed']}/"
            f"{self.config['num_samples']} | output={report.output}"
        ]
        for row in summary["action_summary"]:
            if row["metric"] == "mae":
                lines.append(
                    f"{row['task']} | {row['comparison']} | {row['space']} | "
                    f"MAE={display_value(row['mean'])} +/- {display_value(row['std'])} | "
                    f"n={row['n']} missing={row['n_missing']}"
                )
        for gate in summary.get("gates", []):
            stage = next(
                (
                    r
                    for r in summary["stage_summary"]
                    if r["stage"] == gate["stage"]
                    and r["condition"] == "original"
                    and r["kind"] == "sidecar_action"
                ),
                {},
            )
            ablation = next(
                (
                    r
                    for r in summary["action_summary"]
                    if r["metric"] == "mae"
                    and r["space"] == "normalized"
                    and r["comparison"] == f"original__vs__zero_stage_{gate['stage']}"
                ),
                {},
            )
            lines.append(
                f"stage={gate['stage']} block={gate['block_index']} | "
                f"point_gate={display_value(gate['point_gate'])} "
                f"action_gate={display_value(gate['action_gate'])} | "
                f"residual_ratio={display_value(stage.get('residual_ratio_mean'))} +/- "
                f"{display_value(stage.get('residual_ratio_std'))} | "
                f"cosine={display_value(stage.get('cosine_similarity_mean'))} +/- "
                f"{display_value(stage.get('cosine_similarity_std'))} | "
                f"n={stage.get('n_samples', 0)} | "
                f"stage_ablation_MAE={display_value(ablation.get('mean'))} +/- "
                f"{display_value(ablation.get('std'))}"
            )
        if report.geometry:
            for row in summary["geometry_summary"]:
                lines.append(
                    f"geometry/{row['metric']}={display_value(row['mean'])} +/- "
                    f"{display_value(row['std'])} | n={row['n']} missing={row['n_missing']}"
                )
        lines.append(summary["interpretation"])
        for kind in ("errors", "skipped_metrics"):
            lines.extend(
                dict.fromkeys(
                    f"{'ERROR' if kind == 'errors' else 'N/A'} | {r['metric']}: {r['reason']}"
                    for r in summary[kind]
                )
            )
        self.stream.write("\n".join(lines) + "\n")
        self.stream.flush()

    def final(self, report):
        if not self.tty:
            self._final_text(report)
            return
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text

        console = Console(file=self.stream, force_terminal=self.tty, no_color=not self.tty)
        summary = report.summary
        console.print(
            Text(
                f"WAH {summary['status']} · {summary['samples_completed']}/"
                f"{self.config['num_samples']} samples · {report.output}",
                style="bold cyan",
            )
        )
        actions = Table(
            title="Paired action differences · mean ± std (population)", header_style="bold cyan"
        )
        for name in ("Task", "Comparison", "Space", "MAE", "Std", "n / missing"):
            actions.add_column(name)
        for row in summary["action_summary"]:
            if row["metric"] == "mae":
                actions.add_row(
                    row["task"],
                    row["comparison"],
                    row["space"],
                    display_value(row["mean"]),
                    display_value(row["std"]),
                    f"{row['n']} / {row['n_missing']}",
                )
        console.print(actions)
        if summary.get("gates"):
            table = Table(
                title="Baseline Sidecar stages · actual block indices (zero-based)",
                header_style="bold cyan",
            )
            for name in (
                "Stage / block",
                "Point gate",
                "Action gate",
                "||ΔA||/||A|| ± std",
                "cos(A, ΔA) ± std",
                "n",
                "Stage ablation MAE ± std",
            ):
                table.add_column(name)
            for gate in summary["gates"]:
                stage = next(
                    (
                        r
                        for r in summary["stage_summary"]
                        if r["condition"] == "original"
                        and r["kind"] == "sidecar_action"
                        and r["stage"] == gate["stage"]
                    ),
                    {},
                )
                ablation = next(
                    (
                        r
                        for r in summary["action_summary"]
                        if r["metric"] == "mae"
                        and r["space"] == "normalized"
                        and r["comparison"] == f"original__vs__zero_stage_{gate['stage']}"
                    ),
                    {},
                )

                def pair(metric):
                    return (
                        f"{display_value(stage.get(metric + '_mean'))} ± "
                        f"{display_value(stage.get(metric + '_std'))}"
                    )

                table.add_row(
                    f"{gate['stage']} / {gate['block_index']}",
                    display_value(gate["point_gate"]),
                    display_value(gate["action_gate"]),
                    pair("residual_ratio"),
                    pair("cosine_similarity"),
                    str(stage.get("n_samples", 0)),
                    f"{display_value(ablation.get('mean'))} ± {display_value(ablation.get('std'))}",
                )
            console.print(table)
        if report.geometry:
            table = Table(title="Scene-pair geometry sensitivity", header_style="bold cyan")
            for name in ("Metric", "Mean", "Std", "n / missing"):
                table.add_column(name)
            for row in summary["geometry_summary"]:
                table.add_row(
                    row["metric"],
                    display_value(row["mean"]),
                    display_value(row["std"]),
                    f"{row['n']} / {row['n_missing']}",
                )
            console.print(table)
        console.print(Text(summary["interpretation"], style="dim"))
        for title in ("errors", "skipped_metrics"):
            reasons = list(dict.fromkeys(f"{r['metric']}: {r['reason']}" for r in summary[title]))
            for reason in reasons[:12]:
                console.print(Text(f"{'ERROR' if title == 'errors' else 'N/A'} · {reason}"))
            if len(reasons) > 12:
                console.print(f"{len(reasons) - 12} more reasons in summary.json")


class DiagnosticWandb:
    """Opt-in, separate inference run; W&B failures never invalidate local results."""

    def __init__(self, args, report):
        self.args, self.report, self.run, self.module = args, report, None, None

    def start(self):
        if not self.args.wandb or self.args.wandb_mode == "disabled":
            self.report.summary["wandb"] = dict(status="disabled")
            return
        try:
            import wandb

            self.module = wandb
            # Explicitly disable Git discovery and code saving: neither is authorized.
            self.run = wandb.init(
                project=self.args.wandb_project,
                group=self.args.wandb_group or f"diagnostics-{self.args.experiment}",
                name=self.args.wandb_name or f"diagnostics-{self.args.experiment}-{self.args.task}",
                id=uuid.uuid4().hex,
                resume="never",
                reinit="create_new",
                job_type="wah-diagnostics",
                tags=["diagnostics", "inference", self.args.experiment],
                mode=self.args.wandb_mode,
                dir=str(self.report.output),
                config=self.report.summary["config"],
                settings=wandb.Settings(
                    disable_git=True, disable_code=True, init_timeout=15, quiet=True
                ),
            )
            self.run.define_metric("diagnostic_sample")
            self.run.define_metric("sample/*", step_metric="diagnostic_sample")
            self.run.define_metric("diagnostic_stage")
            self.run.define_metric("stage/*", step_metric="diagnostic_stage")
            self.report.summary["wandb"] = dict(
                status=self.args.wandb_mode, run_id=self.run.id, url=getattr(self.run, "url", None)
            )
        except Exception as exc:
            self.report.issue("wandb_initialization", exc, error=True)
            self.report.summary["wandb"] = dict(status="failed", reason=str(exc))

    def sample(self, sample, rows):
        if self.run is None:
            return
        try:
            payload = {"diagnostic_sample": sample}
            for row in rows:
                if row.get("mae") is not None:
                    payload[f"sample/{row['comparison']}/{row['space']}/mae"] = row["mae"]
            self.run.log(payload)
        except Exception as exc:
            self.report.issue("wandb_sample_log", exc, error=True)

    def finish(self):
        if self.run is None:
            return
        try:
            wb, summary = self.module, self.report.summary
            self.run.config.update(
                dict(summary["config"], samples_completed=summary["samples_completed"]),
                allow_val_change=True,
            )
            payload = {}
            for row in summary["action_summary"]:
                prefix = f"diagnostics/{row['comparison']}/{row['space']}/{row['metric']}"
                task_prefix = (
                    f"tasks/{row['task']}/{row['comparison']}/{row['space']}/{row['metric']}"
                )
                for name in ("mean", "std", "n", "n_missing"):
                    if row[name] is not None:
                        self.run.summary[f"{prefix}_{name}"] = row[name]
                        self.run.summary[f"{task_prefix}_{name}"] = row[name]

            def table(rows):
                columns = sorted({k for row in rows for k in row}) or ["status"]
                return wb.Table(
                    columns=columns,
                    data=[
                        [
                            json.dumps(row.get(k))
                            if isinstance(row.get(k), (dict, list))
                            else row.get(k)
                            for k in columns
                        ]
                        for row in rows
                    ],
                )

            payload["diagnostics/per_task"] = table(summary["action_summary"])
            payload["diagnostics/stages"] = table(summary["stage_summary"])
            payload["diagnostics/geometry"] = table(self.report.geometry)
            for row in summary["stage_summary"]:
                if row["condition"] != "original" or row["kind"] != "sidecar_action":
                    continue
                self.run.log(
                    dict(
                        diagnostic_stage=row["stage"],
                        **{
                            f"stage/{metric}": row[f"{metric}_mean"]
                            for metric in (
                                "action_gate",
                                "point_gate",
                                "residual_ratio",
                                "cosine_similarity",
                            )
                            if row[f"{metric}_mean"] is not None
                        },
                    )
                )
            for name in summary["figures"]:
                payload[f"figures/{Path(name).stem}"] = wb.Image(str(self.report.output / name))
            self.run.summary["diagnostics/status"] = summary["status"]
            self.run.summary["diagnostics/samples_completed"] = summary["samples_completed"]
            self.run.log(payload)
            # Capture local errors/status before packaging detailed machine-readable files.
            self.report.save()
            artifact = wb.Artifact(
                f"wah-diagnostics-{self.run.id}",
                type="wah-diagnostics",
                metadata={"experiment": self.args.experiment, "status": summary["status"]},
            )
            for path in sorted(self.report.output.iterdir()):
                if path.is_file() and path.suffix in {".json", ".csv", ".png"}:
                    artifact.add_file(str(path), name=path.name)
            self.run.log_artifact(artifact)
        except Exception as exc:
            self.report.issue("wandb_finalize", exc, error=True)
        finally:
            try:
                self.run.finish(exit_code=0 if self.report.summary["status"] == "complete" else 1)
            except Exception as exc:
                self.report.issue("wandb_finish", exc, error=True)
            self.run = None

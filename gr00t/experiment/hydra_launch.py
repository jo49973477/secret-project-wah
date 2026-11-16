"""Parent-only preflight, resume validation, snapshots and Torchrun launching."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from gr00t.configs.hydra_config import build_gr00t_config, preflight
from gr00t.configs.sidecar_config import default_action_gate_init, default_point_gate_init
from gr00t.experiment.dashboard_telemetry import ParentDashboard
from gr00t.experiment.trainctl_launch import LogRelay, check_trainctl, trainctl_run
from gr00t.utils.sensor_dependency_preflight import check_sensor_dependencies


def validate_checkpoint(path: Path) -> None:
    """Check the same state families as checkpoint_resume.sh without unpickling tensors."""
    if not path.is_dir() or not re.fullmatch(r"checkpoint-\d+", path.name):
        raise ValueError(f"Invalid training checkpoint directory: {path}")
    checks = {
        "trainer state": (path / "trainer_state.json").is_file(),
        "optimizer state": (path / "optimizer.pt").is_file() or any(path.rglob("*optim_states.pt")),
        "scheduler state": (path / "scheduler.pt").is_file() or any(path.rglob("*model_states.pt")),
        "RNG state": any(path.glob("rng_state*.pth")),
        "model weights/partitions": any(path.rglob("*model_states.pt"))
        or any(path.glob("*.safetensors"))
        or any(path.glob("pytorch_model*.bin")),
    }
    missing = [key for key, present in checks.items() if not present]
    if missing:
        raise ValueError(
            f"Checkpoint {path} is not resumable; missing {', '.join(missing)}. Model-only checkpoints are fine-tuning sources, not full resumes."
        )
    state = json.loads((path / "trainer_state.json").read_text())
    if type(state.get("global_step")) is not int or state["global_step"] != int(
        path.name.split("-")[1]
    ):
        raise ValueError(f"Checkpoint {path}: trainer global_step must match directory step")


def find_latest(output: Path) -> Path:
    candidates = []
    for path in output.glob("checkpoint-*"):
        try:
            validate_checkpoint(path)
        except (ValueError, OSError):
            continue
        candidates.append(path)
    if not candidates:
        raise ValueError(f"No valid resumable checkpoint-* found in {output}")
    return max(candidates, key=lambda path: int(path.name.split("-")[1]))


def resolve_resume(cfg: DictConfig, config) -> Path | None:
    if not cfg.resume.enabled:
        return None
    output = Path(cfg.output_dir).resolve()
    checkpoint = (
        Path(cfg.resume.checkpoint_path).resolve()
        if cfg.resume.checkpoint_path
        else find_latest(output)
    )
    validate_checkpoint(checkpoint)
    if checkpoint.parent != output:
        raise ValueError(
            f"Resume must retain the original output directory; set output_dir={checkpoint.parent}"
        )
    # Hydra provenance is required: legacy checkpoint state alone cannot prove all compatibility.
    saved_path = output / ".hydra" / "config.yaml"
    if not saved_path.is_file():
        raise ValueError(
            f"Cannot verify resume compatibility: missing {saved_path}. Use the legacy launcher for legacy runs, or fine-tune from the model in a fresh run."
        )
    old = OmegaConf.to_container(OmegaConf.load(saved_path), resolve=True)
    new = OmegaConf.to_container(cfg, resolve=True)
    allowed_training_changes = {
        "max_steps",
        "logging_steps",
        "save_steps",
        "save_total_limit",
        "use_wandb",
        "wandb_project",
        "wandb_group",
        "telegram_on",
        "telegram_chat_id",
        "telegram_notify_start",
        "telegram_notify_save",
        "telegram_notify_finish",
        "telegram_notify_error",
        "discord_on",
        "discord_notify_start",
        "discord_notify_save",
        "discord_notify_finish",
        "discord_notify_error",
        "enable_training_dashboard",
        "dashboard_refresh_seconds",
        "enable_multimodal_diagnostics",
        "multimodal_diagnostics_interval",
        "multimodal_diagnostics_level",
    }
    differences = []
    for section in ("model", "data", "training", "hardware"):
        before, after = dict(old[section]), dict(new[section])
        if section == "model":
            defaults = {
                "sidecar_variant": "legacy",
                "sidecar_point_stage_names": None,
                "sidecar_local_patch_size": 64,
                "sidecar_window_chunk_size": 16,
                "sidecar_point_gate_init": default_point_gate_init(),
                "sidecar_action_gate_init": default_action_gate_init(),
                "sidecar_gate_init_seed": 42,
            }
            for key, value in defaults.items():
                before.setdefault(key, value)
                after.setdefault(key, value)
        ignored = (
            allowed_training_changes
            if section == "training"
            else {
                "master_port",
                "cuda_visible_devices",
                "dataloader_num_workers",
                "global_batch_size",
                "gradient_accumulation_steps",
            }
            if section == "hardware"
            else set()
        )
        for key in sorted(set(before) | set(after)):
            if key not in ignored and before.get(key) != after.get(key):
                differences.append(f"{section}.{key}")
    if differences:
        raise ValueError(
            f"Incompatible resume changes: {', '.join(differences)}. Optimizer/scheduler state restores saved LRs; LR, modality, architecture and optimizer structure changes require a fresh fine-tune."
        )
    if old["name"] != new["name"]:
        raise ValueError("Resume must use the original experiment name")
    cfg.run_id = old["run_id"]
    config.training.wandb_name = f"{cfg.name}-{cfg.run_id}"
    step = json.loads((checkpoint / "trainer_state.json").read_text())["global_step"]
    if (
        config.training.max_steps <= step
        or config.training.max_steps < old["training"]["max_steps"]
    ):
        raise ValueError(
            "Resume max_steps must exceed the saved step and must not shorten the original duration"
        )
    config.training.resume_from_checkpoint = str(checkpoint)
    # Preserve the legacy construction path from the original base model. Trainer
    # restores the checkpoint (including native ZeRO partitions) after construction.
    return checkpoint


def write_snapshots(
    cfg: DictConfig, config, checkpoint: Path | None = None, florence_audit: Path | None = None
) -> Path:
    output = Path(cfg.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    # Keep initial provenance immutable. Resumes get their own small manifest directory.
    snapshot = output / ".hydra"
    if checkpoint:
        snapshot = snapshot / f"resume_{datetime.now(timezone.utc):%Y%m%d_%H%M%S_%f}"
    snapshot.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, snapshot / "config.yaml", resolve=True)
    hydra = HydraConfig.get() if HydraConfig.initialized() else OmegaConf.create({})
    OmegaConf.save(hydra, snapshot / "hydra.yaml", resolve=False)
    overrides = list(hydra.overrides.task) if HydraConfig.initialized() else []
    OmegaConf.save(OmegaConf.create(overrides), snapshot / "overrides.yaml")
    config.save(snapshot / "gr00t.yaml")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", None
    metadata = {
        "git_commit": commit,
        "git_dirty": dirty,
        "experiment_name": cfg.name,
        "run_id": output.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "configuration_groups": OmegaConf.to_container(hydra.runtime.choices)
        if HydraConfig.initialized()
        else {},
        "overrides": overrides,
        "base_model": cfg.model.base_model_path,
        "datasets": [asdict(ds) for ds in config.data.datasets],
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "precision": "bf16" if config.training.bf16 else "fp16" if config.training.fp16 else "fp32",
        "resume_source": str(checkpoint) if checkpoint else None,
        **preflight(cfg, config),
    }
    if cfg.name in {
        "temporal_sidecar_ditfrozen-overfit5",
        "temporal_sidecar_ditlora-overfit5",
        "temporal_sidecar_ditfull-overfit5",
        "moe_sidecar_ditfrozen-overfit5",
        "moe_sidecar_ditlora-overfit5",
        "moe_sidecar_ditfull-overfit5",
        "temporal_sidecar_ditlora-full",
        "moe_sidecar_ditlora-full",
        "temporal_sidecar_ditfull-full",
        "moe_sidecar_ditfull-full",
    }:
        audit = json.loads(florence_audit.read_text()) if florence_audit else {}
        metadata.update(
            {
                "pretrained_dit_verified": False,
                "pretrained_dit_action_audit_passed": audit.get("checks", {}).get(
                    "actual_pretrained_dit"
                )
                == "PASSED",
                "pretrained_model_checkpoint_revision": audit.get(
                    "actual_pretrained_checkpoint", {}
                ).get("revision"),
                "dit_training_mode": config.model.dit_training_mode,
                "temporal_window_size": config.model.temporal_window_size,
                "dit_lora_rank": config.model.dit_lora_rank,
                "dit_lora_alpha": config.model.dit_lora_alpha,
            }
        )
    (snapshot / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return snapshot / "config.yaml"


def worker_command(cfg: DictConfig, resolved_path: Path) -> list[str]:
    worker = Path(__file__).resolve().parents[2] / "scripts" / "train_hydra_worker.py"
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc-per-node={cfg.hardware.num_gpus}",
        f"--master-port={cfg.hardware.master_port}",
        str(worker),
        str(resolved_path),
    ]


def validate_dataset_inputs(cfg: DictConfig) -> None:
    """Check local dataset prerequisites and an explicitly asserted independent subset."""
    expected = cfg.data.get("expected_episode_ids")
    for dataset in cfg.data.datasets:
        for directory in dataset.dataset_paths:
            path = Path(directory)
            if not path.is_dir():
                raise ValueError(
                    f"Dataset directory does not exist: {path}. Set data.datasets or FLORENCE_DATASET to an existing converted dataset"
                )
            for name in ("info.json", "modality.json"):
                if not (path / "meta" / name).is_file():
                    raise ValueError(f"Dataset is missing meta/{name}: {path}")
            if expected is not None:
                episodes = [
                    json.loads(line)
                    for line in (path / "meta/episodes.jsonl").read_text().splitlines()
                    if line.strip()
                ]
                actual = sorted(int(ep["episode_index"]) for ep in episodes)
                if actual != sorted(expected):
                    raise ValueError(
                        f"Dataset {path} must be an independent subset containing exactly episode IDs {list(expected)}, got {actual}. Use univtac_overfit_workflow.py prepare"
                    )


def launch(cfg: DictConfig) -> int:
    config = build_gr00t_config(cfg)
    # Dry-run does not even read checkpoint metadata; resume validation is a real-launch operation.
    if cfg.dry_run:
        print(OmegaConf.to_yaml(cfg, resolve=True), flush=True)
        print(json.dumps(preflight(cfg, config), indent=2), flush=True)
        return 0
    if "RANK" in os.environ:
        raise ValueError("Run train_hydra.py once as the parent; it launches Torchrun itself")
    if cfg.trainctl.enabled:
        check_trainctl()
        if os.environ.get("TRAINCTL_RUN_ID"):
            raise ValueError("Launch Hydra directly, outside trainctl, to avoid nested supervisors")
    check_sensor_dependencies(config.model)
    validate_dataset_inputs(cfg)
    checkpoint = resolve_resume(cfg, config)
    output = Path(cfg.output_dir).resolve()
    if not checkpoint and output.exists() and any(output.iterdir()):
        raise ValueError(
            f"Fresh run requires an empty output_dir: {output}; use resume.enabled=true or a new run_id"
        )
    florence_names = {
        "temporal_sidecar_ditfrozen-overfit5",
        "temporal_sidecar_ditlora-overfit5",
        "temporal_sidecar_ditfull-overfit5",
        "moe_sidecar_ditfrozen-overfit5",
        "moe_sidecar_ditlora-overfit5",
        "moe_sidecar_ditfull-overfit5",
        "temporal_sidecar_ditlora-full",
        "moe_sidecar_ditlora-full",
        "temporal_sidecar_ditfull-full",
        "moe_sidecar_ditfull-full",
    }
    florence_report_path = None
    if cfg.name in florence_names:
        from scripts.preflight_florence_ablations import run as run_florence_preflight

        florence_report_path = (
            Path("outputs/preflight") / f"{cfg.name}-{cfg.run_id}" / "florence_preflight.json"
        ).resolve()
        result = run_florence_preflight(cfg, config, florence_report_path)
        if result["status"] != "PASSED":
            raise RuntimeError(
                f"Florence preflight failed: {result.get('error')}; see {florence_report_path}"
            )
    # Serialize resolved literals so ranks cannot resolve environment/time independently.
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    cfg.output_dir = str(output)
    config.training.output_dir = str(output)
    if checkpoint:
        cfg.resume.checkpoint_path = str(checkpoint)
        cfg.resume.auto_find_latest = False
    print(OmegaConf.to_yaml(cfg, resolve=True), flush=True)
    print(json.dumps(preflight(cfg, config), indent=2), flush=True)
    path = write_snapshots(cfg, config, checkpoint, florence_report_path)
    if florence_report_path is not None:
        import shutil

        run_preflight = output / "preflight"
        run_preflight.mkdir(exist_ok=True)
        for source, destination in (
            (florence_report_path, run_preflight / "florence_preflight.json"),
            (
                florence_report_path.with_name("dit_verification.json"),
                run_preflight / "dit_verification.json",
            ),
            (
                florence_report_path.with_name("dit_verification.txt"),
                run_preflight / "dit_verification.txt",
            ),
        ):
            shutil.copy2(source, destination)
        for name in ("dit_verification.json", "dit_verification.txt"):
            shutil.copy2(run_preflight / name, Path("outputs/preflight") / name)
        florence_report_path = run_preflight / "florence_preflight.json"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = cfg.hardware.cuda_visible_devices
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    env.setdefault("LOGURU_LEVEL", "INFO")
    env["GR00T_HYDRA_CONFIG"] = str(path)
    env["GROOT_COMMIT_HASH"] = json.loads((output / "metadata.json").read_text())["git_commit"]
    if florence_report_path is not None:
        env["GR00T_FLORENCE_PREFLIGHT"] = str(florence_report_path)
    env.update(
        GR00T_EXPERIMENT_NAME=cfg.name,
        GR00T_OUTPUT_DIR=str(output),
        GR00T_TRAIN_LOG=str(output / "logs/train.log"),
    )
    command = worker_command(cfg, path)
    # A separate process group allows forwarding termination to supervisor, Torchrun and ranks.
    with (
        trainctl_run(cfg, output, command, env) as invocation,
        (output / "logs/train.log").open("a", buffering=1) as log,
    ):
        command, env = invocation
        relay = None
        with ParentDashboard(cfg, config, output, env) as dashboard:
            master = slave = None
            interactive = (
                sys.stdout.isatty() and config.training.enable_training_dashboard is not False
            )
            worker_pty = interactive and not cfg.trainctl.enabled
            if worker_pty:
                import pty

                from gr00t.experiment.console_logging import copy_terminal_size

                master, slave = pty.openpty()
                copy_terminal_size(sys.stdout.fileno(), slave)
                env["GR00T_LIVE_CONSOLE"] = "1"
            process = subprocess.Popen(
                command,
                env=env,
                stdout=slave if worker_pty else subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )

            if cfg.trainctl.enabled:
                relay = LogRelay(env["TRAINCTL_LOG_DIR"], log)
                relay.start()

            if slave is not None:
                os.close(slave)

            def forward(signum, frame):
                if worker_pty and signum == signal.SIGWINCH:
                    copy_terminal_size(sys.stdout.fileno(), master)
                    return
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signum)
                    except ProcessLookupError:
                        pass

            signals = (signal.SIGINT, signal.SIGTERM) + ((signal.SIGWINCH,) if worker_pty else ())
            previous = {sig: signal.signal(sig, forward) for sig in signals}
            try:
                if worker_pty:
                    import codecs
                    import errno

                    decoder = codecs.getincrementaldecoder("utf-8")("replace")
                    while True:
                        try:
                            chunk = os.read(master, 65536)
                        except OSError as exc:
                            if exc.errno != errno.EIO:
                                raise
                            break
                        if not chunk:
                            break
                        text = decoder.decode(chunk)
                        sys.stdout.write(text)
                        sys.stdout.flush()
                        log.write(text)
                    tail = decoder.decode(b"", final=True)
                    sys.stdout.write(tail)
                    log.write(tail)
                else:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                code = process.wait()
                dashboard.result = code
            finally:
                if master is not None:
                    os.close(master)
                    # Last resort if a worker was killed before Rich's finally cleanup.
                    sys.stdout.write("\x1b[?1049l\x1b[?25h")
                    sys.stdout.flush()
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait()
                if relay is not None:
                    relay.close()
    return code if code >= 0 else 128 - code

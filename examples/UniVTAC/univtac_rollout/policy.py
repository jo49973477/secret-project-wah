"""UniVTAC-to-GR00T deployment policy adapter."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .checkpoint import checkpoint_identity
from .client import MinimalPolicyClient
from .core import extract_action_chunk
from .multimodal import (
    ObservationHistory,
    add_scene_calibration,
    detect_policy,
    field,
    load_sensor_contract,
    validate_contract,
)
from .profiling import GLOBAL_PROFILER, RuntimeProfiler


def validate_rgb_checkpoint_contract(configs: dict[str, Any], execution_horizon: int) -> None:
    if detect_policy(configs) == "wah":
        raise ValueError(
            "This RGB rollout adapter cannot satisfy checkpoint modalities ['pointcloud', 'tactile']."
        )
    validate_contract(configs, execution_horizon, "rgb")


class Gr00tUniVTACPolicy:
    """Small adapter that leaves all preprocessing/action decoding on the GR00T server."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        execution_horizon: int = 1,
        timeout_ms: int = 300_000,
        checkpoint: str | None = None,
        expected_checkpoint_identity: str | None = None,
        policy: str = "auto",
        sensor_contract: str | None = None,
        scene_calibration: str | None = None,
        profiler: RuntimeProfiler | None = None,
        log_level: str = "normal",
    ):
        if execution_horizon < 1:
            raise ValueError("execution_horizon must be positive.")
        self.execution_horizon = execution_horizon
        self.profiler = profiler or GLOBAL_PROFILER
        self.log_level = log_level
        self.client = MinimalPolicyClient(host, port, timeout_ms=timeout_ms, profiler=self.profiler)
        try:
            if not self.client.ping():
                raise ConnectionError(f"GR00T server at {host}:{port} returned an invalid ping.")
            self.model_info = self.client.get_model_info()
            self.modality_configs = self.client.get_modality_config()
            self.policy_type = validate_contract(self.modality_configs, execution_horizon, policy)
            if checkpoint is not None or expected_checkpoint_identity is not None:
                expected = expected_checkpoint_identity or checkpoint_identity(checkpoint)
                if not expected.startswith("sha256:") or len(expected) != 71:
                    raise ValueError("Expected checkpoint identity must be sha256:<64 hex digits>.")
                actual = self.model_info["checkpoint_identity"]
                if expected != actual:
                    raise ValueError(
                        f"Checkpoint identity mismatch: requested {expected}, server loaded {actual} "
                        f"({self.model_info.get('checkpoint_path')})."
                    )
            contract = (
                add_scene_calibration(load_sensor_contract(sensor_contract), scene_calibration)
                if self.policy_type == "wah"
                else None
            )
            self.history = ObservationHistory(self.modality_configs, contract)
            self._latest_formatted = None
            if self.log_level == "debug":
                print(
                    f"Policy={self.policy_type} checkpoint={self.model_info['checkpoint_identity']} "
                    f"DiT={self.model_info.get('dit_type')} mode={self.model_info.get('dit_training_mode')} "
                    f"Sidecar={self.model_info.get('sidecar_type')} LoRA={self.model_info.get('dit_lora_enabled')} "
                    f"modalities={list(self.modality_configs)} "
                    f"windows={ {k: len(field(v, 'delta_indices')) for k, v in self.modality_configs.items()} } "
                    f"action_shape=[1,{self.model_info.get('action_horizon')},8]",
                    flush=False,
                )
        except Exception:
            self.client.close()
            raise

    def reset(self) -> None:
        self.history.reset()
        self._latest_formatted = None
        self.client.reset()

    def record_observation(
        self, observation: dict[str, Any], instruction: str, timestamp: int
    ) -> None:
        """Record each post-action sensor sample, even when a chunk executes several actions."""
        self._latest_formatted = self.history.append(observation, instruction, timestamp)

    def predict_chunk(
        self, observation: dict[str, Any], instruction: str, *, timestamp: int | None = None
    ) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
        if (
            timestamp is not None
            and timestamp == self.history.timestamp
            and self._latest_formatted is not None
        ):
            formatted = self._latest_formatted
        else:
            observation_started = time.perf_counter_ns()
            self.record_observation(
                observation,
                instruction,
                self.history.timestamp + 1 if timestamp is None else timestamp,
            )
            self.profiler.record_ns(
                "inference.observation_preparation", time.perf_counter_ns() - observation_started
            )
            formatted = self._latest_formatted
        if self.history.timestamp == 0:
            shapes = {
                name: {
                    key: (value.shape, str(value.dtype))
                    for key, value in values.items()
                    if isinstance(value, np.ndarray)
                }
                for name, values in formatted.items()
                if isinstance(values, dict)
            }
            if "temporal_valid" in formatted:
                valid = formatted["temporal_valid"]
                shapes["temporal_valid"] = (valid.shape, str(valid.dtype))
            if self.log_level == "debug":
                print("Input shapes: " + repr(shapes), flush=False)
        chunk = extract_action_chunk(self.client.get_action(formatted))
        expected_horizon = self.model_info.get("action_horizon")
        if expected_horizon is not None and len(chunk) != expected_horizon:
            raise ValueError(
                f"GR00T server declared action horizon {expected_horizon} but returned {len(chunk)}."
            )
        if self.execution_horizon > len(chunk):
            raise ValueError(
                f"execution_horizon={self.execution_horizon} exceeds returned horizon {len(chunk)}."
            )
        return chunk[: self.execution_horizon], formatted

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "Gr00tUniVTACPolicy":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

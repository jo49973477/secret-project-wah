# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Point-cloud encoders for multimodal action conditioning.

Concerto is an optional dependency.  It is imported only when a Concerto
backend is selected so RGB-only and lightweight point-cloud workflows do not
need the sparse-convolution stack.
"""

from contextlib import nullcontext
import importlib
import inspect
import logging
import os
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from gr00t.utils.model_load_diagnostics import log_model_load_memory


LOGGER = logging.getLogger(__name__)


def _index_points(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather ``[B, ...]`` point indices from a ``[B, N, C]`` tensor."""
    batch_indices = torch.arange(points.shape[0], device=points.device)
    batch_indices = batch_indices.view(-1, *([1] * (indices.ndim - 1)))
    return points[batch_indices, indices]


def _farthest_point_sample(
    xyz: torch.Tensor,
    num_samples: int,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select representative points with deterministic farthest-point sampling."""
    batch_size, num_points, _ = xyz.shape
    num_samples = min(num_samples, num_points)

    sampled_indices = torch.zeros(batch_size, num_samples, dtype=torch.long, device=xyz.device)
    sampled_mask = torch.zeros(batch_size, num_samples, dtype=torch.bool, device=xyz.device)
    min_distances = torch.full(
        (batch_size, num_points), torch.inf, dtype=xyz.dtype, device=xyz.device
    )
    selected_mask = torch.zeros_like(valid_mask)
    valid_counts = valid_mask.sum(dim=1)

    # Start with the first valid point. Rows without a valid point use index 0
    # as a harmless placeholder and remain False in sampled_mask.
    farthest_indices = valid_mask.to(dtype=torch.int64).argmax(dim=1)
    batch_indices = torch.arange(batch_size, device=xyz.device)

    for sample_index in range(num_samples):
        sample_is_valid = sample_index < valid_counts
        sampled_indices[:, sample_index] = farthest_indices
        sampled_mask[:, sample_index] = sample_is_valid
        selected_mask[batch_indices, farthest_indices] |= sample_is_valid

        centroids = xyz[batch_indices, farthest_indices]
        squared_distances = ((xyz - centroids[:, None, :]) ** 2).sum(dim=-1)
        min_distances = torch.minimum(min_distances, squared_distances)

        candidate_scores = min_distances.masked_fill(~valid_mask | selected_mask, -1.0)
        farthest_indices = candidate_scores.argmax(dim=1)

    return sampled_indices, sampled_mask


class SetAbstraction(nn.Module):
    """One PointNet++ set-abstraction stage using k-nearest neighborhoods."""

    def __init__(
        self,
        num_samples: int,
        num_neighbors: int,
        input_feature_dim: int,
        mlp_channels: tuple[int, ...],
    ) -> None:
        super().__init__()
        if num_samples <= 0 or num_neighbors <= 0:
            raise ValueError("num_samples and num_neighbors must be positive.")

        layers: list[nn.Module] = []
        input_channels = input_feature_dim + 3  # point features plus relative XYZ
        for output_channels in mlp_channels:
            layers.extend(
                [
                    nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            input_channels = output_channels

        self.num_samples = num_samples
        self.num_neighbors = num_neighbors
        self.shared_mlp = nn.Sequential(*layers)

    def forward(
        self,
        xyz: torch.Tensor,
        point_features: Optional[torch.Tensor],
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample centroids and aggregate their local neighborhoods."""
        centroid_indices, centroid_mask = _farthest_point_sample(xyz, self.num_samples, valid_mask)
        centroid_xyz = _index_points(xyz, centroid_indices)

        # squared_distances: [B, N_centroid, N_input]
        squared_distances = ((centroid_xyz[:, :, None] - xyz[:, None, :]) ** 2).sum(dim=-1)
        squared_distances = squared_distances.masked_fill(~valid_mask[:, None, :], torch.inf)

        num_neighbors = min(self.num_neighbors, xyz.shape[1])
        neighbor_distances, neighbor_indices = squared_distances.topk(
            k=num_neighbors, dim=-1, largest=False, sorted=False
        )
        neighbor_mask = torch.isfinite(neighbor_distances) & centroid_mask[:, :, None]

        # topk fills short masked neighborhoods with arbitrary invalid indices.
        # Repeat a valid neighbor instead so padded point values do not enter
        # the shared MLP or its batch-normalization statistics.
        fallback_indices = squared_distances.argmin(dim=-1, keepdim=True)
        neighbor_indices = torch.where(neighbor_mask, neighbor_indices, fallback_indices)

        # grouped_xyz: [B, N_centroid, K, 3]
        grouped_xyz = _index_points(xyz, neighbor_indices) - centroid_xyz[:, :, None]
        if point_features is None:
            grouped_features = grouped_xyz
        else:
            grouped_features = torch.cat(
                (grouped_xyz, _index_points(point_features, neighbor_indices)), dim=-1
            )
        grouped_features = grouped_features.masked_fill(~centroid_mask[:, :, None, None], 0.0)

        # shared_features: [B, C_out, N_centroid, K]
        shared_features = self.shared_mlp(grouped_features.permute(0, 3, 1, 2))
        minimum_value = torch.finfo(shared_features.dtype).min
        shared_features = shared_features.masked_fill(~neighbor_mask[:, None, :, :], minimum_value)
        pooled_features = shared_features.max(dim=-1).values.transpose(1, 2)
        pooled_features = pooled_features.masked_fill(~centroid_mask[:, :, None], 0.0)

        return centroid_xyz, pooled_features, centroid_mask


class PointNet2Encoder(nn.Module):
    """Encode XYZ and optional per-point features with two PointNet++ stages.

    The first three input values are always interpreted as XYZ. Remaining
    values, normally RGB, are local point features.

    Input:
        points: ``[B, N, input_dim]``.

    Output:
        point_tokens: ``[B, min(N, num_samples[-1]), point_dim]``.
    """

    def __init__(
        self,
        input_dim: int = 6,
        point_dim: int = 256,
        num_samples: tuple[int, int] = (256, 64),
        num_neighbors: tuple[int, int] = (32, 32),
        dropout: float = 0.0,
        *,
        token_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if input_dim < 3:
            raise ValueError(f"input_dim must include XYZ and be at least 3, got {input_dim}.")
        if len(num_samples) != 2 or len(num_neighbors) != 2:
            raise ValueError("num_samples and num_neighbors must each contain two values.")

        # token_dim and hidden_dim are retained for compatibility with the old
        # PointEncoder constructor used by early extension checkpoints.
        if token_dim is not None:
            point_dim = token_dim
        first_stage_dim = hidden_dim or min(128, point_dim)

        self.input_dim = input_dim
        self.point_dim = point_dim
        self.stage1 = SetAbstraction(
            num_samples=num_samples[0],
            num_neighbors=num_neighbors[0],
            input_feature_dim=input_dim - 3,
            mlp_channels=(64, 64, first_stage_dim),
        )
        self.stage2 = SetAbstraction(
            num_samples=num_samples[1],
            num_neighbors=num_neighbors[1],
            input_feature_dim=first_stage_dim,
            mlp_channels=(first_stage_dim, point_dim, point_dim),
        )
        self.output_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(point_dim)

    def forward(
        self,
        points: torch.Tensor,
        point_mask: Optional[torch.Tensor] = None,
        *,
        return_mask: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if points.ndim != 3:
            raise ValueError(
                f"points must have shape [B, N, input_dim], got {tuple(points.shape)}."
            )
        if points.shape[-1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} values per point, got {points.shape[-1]}.")
        if points.shape[1] == 0:
            raise ValueError("points must contain at least one point.")

        batch_size, num_points, _ = points.shape
        if point_mask is None:
            point_mask = torch.ones(batch_size, num_points, dtype=torch.bool, device=points.device)
        elif point_mask.shape != (batch_size, num_points):
            raise ValueError(
                "point_mask must have shape "
                f"{(batch_size, num_points)}, got {tuple(point_mask.shape)}."
            )
        else:
            point_mask = point_mask.to(device=points.device, dtype=torch.bool)

        xyz = points[..., :3]
        input_features = points[..., 3:] if self.input_dim > 3 else None
        stage1_xyz, stage1_features, stage1_mask = self.stage1(xyz, input_features, point_mask)
        _, stage2_features, stage2_mask = self.stage2(stage1_xyz, stage1_features, stage1_mask)

        # point_tokens: [B, N_out, point_dim]
        point_tokens = self.output_norm(self.output_dropout(stage2_features))
        point_tokens = point_tokens.masked_fill(~stage2_mask[:, :, None], 0.0)

        if return_mask:
            return point_tokens, stage2_mask
        return point_tokens


class PointTransformerEncoder(nn.Module):
    """Small point transformer that keeps one output token per input point.

    This intentionally simple backend is a replaceable baseline, not a port of
    a larger sparse-convolution or Point Transformer framework.
    """

    def __init__(
        self,
        input_dim: int = 6,
        point_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        feedforward_dim: Optional[int] = None,
        dropout: float = 0.0,
        *,
        token_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if input_dim < 3:
            raise ValueError(f"input_dim must include XYZ and be at least 3, got {input_dim}.")
        if token_dim is not None:
            point_dim = token_dim
        if point_dim % num_heads != 0:
            raise ValueError(
                f"point_dim ({point_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.input_dim = input_dim
        self.point_dim = point_dim
        self.input_projection = nn.Linear(input_dim, point_dim)
        self.position_projection = nn.Sequential(
            nn.Linear(3, point_dim),
            nn.GELU(),
            nn.Linear(point_dim, point_dim),
        )
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=point_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim or point_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(point_dim)

    def forward(
        self,
        points: torch.Tensor,
        point_mask: Optional[torch.Tensor] = None,
        *,
        return_mask: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if points.ndim != 3:
            raise ValueError(
                f"points must have shape [B, N, input_dim], got {tuple(points.shape)}."
            )
        if points.shape[-1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} values per point, got {points.shape[-1]}.")
        if points.shape[1] == 0:
            raise ValueError("points must contain at least one point.")

        batch_size, num_points, _ = points.shape
        if point_mask is None:
            point_mask = torch.ones(batch_size, num_points, dtype=torch.bool, device=points.device)
        elif point_mask.shape != (batch_size, num_points):
            raise ValueError(
                "point_mask must have shape "
                f"{(batch_size, num_points)}, got {tuple(point_mask.shape)}."
            )
        else:
            point_mask = point_mask.to(device=points.device, dtype=torch.bool)

        safe_mask = point_mask.clone()
        rows_without_points = ~safe_mask.any(dim=1)
        if rows_without_points.any():
            safe_mask[rows_without_points, 0] = True

        # point_tokens: [B, N, point_dim]
        point_tokens = self.input_projection(points)
        point_tokens = point_tokens + self.position_projection(points[..., :3])
        point_tokens = self.transformer(point_tokens, src_key_padding_mask=~safe_mask)
        point_tokens = self.output_norm(point_tokens)
        point_tokens = point_tokens.masked_fill(~point_mask[:, :, None], 0.0)

        if return_mask:
            return point_tokens, point_mask
        return point_tokens


class ConcertoPointEncoder(nn.Module):
    """Adapt an official pretrained Concerto PTv3 model to GR00T point tokens.

    Input scenes are dense ``[B, N, 6]`` XYZRGB arrays. XYZ is in one common
    metric frame and RGB is stored in the dataset as ``[0, 255]``. The wrapper
    applies Concerto's published preprocessing: per-scene ``CenterShift``,
    grid coordinates at ``grid_size``, RGB division by 255, and zero normals
    (the official demo's supported ``--wo_normal`` path). Concerto receives
    ``[centered XYZ, RGB/255, zero normals]`` features.
    """

    SUPPORTED_MODELS = {"concerto_small", "concerto_base"}

    def __init__(
        self,
        input_dim: int = 6,
        point_dim: int = 256,
        *,
        model_name: str = "concerto_small",
        repo_id: str = "Pointcept/Concerto",
        checkpoint_path: str | None = None,
        download_root: str | None = None,
        grid_size: float = 0.02,
        enable_flash: bool | None = None,
        load_pretrained: bool = True,
        architecture_config: dict | None = None,
        backbone: nn.Module | None = None,
        backbone_output_dim: int | None = None,
        pretrained_loaded: bool = False,
        memory_map_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if input_dim != 6:
            raise ValueError(
                "Concerto requires one unified XYZRGB scene field with 6 values per point; "
                f"received point_input_dim={input_dim}. Use pointnet2/point_transformer for "
                "legacy XYZ-only data."
            )
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported Concerto model {model_name!r}; expected one of "
                f"{sorted(self.SUPPORTED_MODELS)}."
            )
        if grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {grid_size}.")
        constructing_official_backbone = backbone is None

        self.input_dim = input_dim
        self.point_dim = point_dim
        self.model_name = model_name
        self.repo_id = repo_id
        self.grid_size = float(grid_size)
        self.checkpoint_path = checkpoint_path
        self.download_root = download_root
        self.enable_flash = enable_flash
        self.architecture_config: dict | None = None

        if backbone is None and load_pretrained:
            (
                backbone,
                backbone_output_dim,
                checkpoint_label,
                parameter_count,
                self.architecture_config,
            ) = self._load_official_backbone(
                model_name=model_name,
                repo_id=repo_id,
                checkpoint_path=checkpoint_path,
                download_root=download_root,
                enable_flash=enable_flash,
                memory_map_checkpoint=memory_map_checkpoint,
            )
            pretrained_loaded = True
            LOGGER.info("Point encoder: %s", model_name.replace("_", "-").title())
            LOGGER.info("Checkpoint: %s", checkpoint_label)
            LOGGER.info("Loaded pretrained parameters: %s", f"{parameter_count:,}")
            LOGGER.info("Missing keys: []")
            LOGGER.info("Unexpected keys: []")
        elif backbone is None:
            if architecture_config is None:
                raise RuntimeError(
                    "Deferred Concerto construction requires the exact architecture config "
                    "saved in the multimodal GR00T checkpoint."
                )
            concerto = self._import_concerto()
            self.architecture_config = dict(architecture_config)
            backbone_output_dim = self._output_dim_from_config(self.architecture_config)
            with torch.device("cpu"):
                backbone = concerto.model.PointTransformerV3(**self.architecture_config)
            LOGGER.info("Point encoder architecture: %s", model_name.replace("_", "-").title())
            LOGGER.info("Checkpoint: embedded GR00T checkpoint")
        elif backbone_output_dim is None:
            raise ValueError("backbone_output_dim is required when injecting a Concerto backbone")
        else:
            self.architecture_config = dict(architecture_config) if architecture_config else None

        if load_pretrained and not pretrained_loaded:
            raise RuntimeError(
                "Concerto was requested without verified pretrained weights. Randomly initialized "
                "Concerto backbones are intentionally unsupported."
            )
        assert backbone_output_dim is not None
        self.backbone = backbone
        self._configure_sparse_conv_precision()
        self._configure_zero3_sparse_conv_leaves()
        if constructing_official_backbone and torch.cuda.is_available():
            # Fail during setup, rather than at the first sparse GEMM, when
            # the installed spconv build cannot execute our FP16 island.
            cppcore = importlib.import_module("spconv.pytorch.cppcore")
            if torch.float16 not in cppcore._TORCH_DTYPE_TO_TV:
                raise RuntimeError(
                    "Installed spconv does not support FP16 sparse features; "
                    "Concerto CUDA precision cannot be initialized safely."
                )
        self.projection = nn.Sequential(
            nn.Linear(backbone_output_dim, point_dim),
            nn.LayerNorm(point_dim),
        )
        self.pretrained_loaded = pretrained_loaded
        self._precision_diagnostics_logged = False
        if pretrained_loaded:
            self._mark_hf_initialized()
        LOGGER.info(
            "Trainable: %s", any(parameter.requires_grad for parameter in self.parameters())
        )

    @staticmethod
    def _import_concerto():
        try:
            return importlib.import_module("concerto")
        except ImportError as exc:
            raise ImportError(
                "Concerto point encoding requires the official Pointcept/Concerto package and "
                "its spconv + torch-scatter dependencies. Install it as documented in "
                "examples/UniVTAC/README.md."
            ) from exc

    @staticmethod
    def _output_dim_from_config(config: dict) -> int:
        if config.get("in_channels") != 9:
            raise RuntimeError(
                "This wrapper follows official XYZ+RGB+normal feature construction and expects "
                f"a 9-channel Concerto checkpoint, got in_channels={config.get('in_channels')}."
            )
        if config.get("enc_mode", False):
            return int(config["enc_channels"][-1])
        return int(config["dec_channels"][0])

    def _mark_hf_initialized(self) -> None:
        for module in self.backbone.modules():
            module._is_hf_initialized = True

    def _configure_sparse_conv_precision(self) -> None:
        """Cast operation arguments, never Parameters, for each sparse convolution.

        spconv's eval path bypasses custom_fwd's AMP casts and forwards its
        registered bias straight to tensorview. DeepSpeed stores that bias in
        BF16, which tensorview does not support in our spconv build. The local
        boundary covers training and eval, including the fused inference bias.
        ZeRO's module/leaf hooks still run normally before this method receives
        the gathered parameters. Differentiable casts feed gradients back to
        those same parameters; no converted tensors are cached across calls.
        """
        for name, module in self.backbone.named_modules():
            if not module.__class__.__module__.startswith("spconv."):
                continue
            if not isinstance(getattr(module, "weight", None), nn.Parameter):
                continue
            if getattr(module, "_gr00t_sparse_precision", False):
                continue
            original = getattr(module, "_conv_forward", None)
            if not callable(original) or list(inspect.signature(original).parameters)[:4] != [
                "training",
                "input",
                "weight",
                "bias",
            ]:
                raise RuntimeError(
                    f"Unsupported spconv convolution API at {name}; expected "
                    "_conv_forward(training, input, weight, bias, ...). "
                    "Cannot enforce the Concerto precision boundary safely."
                )

            def conv_with_supported_precision(
                training,
                input,
                weight,
                bias,
                add_input=None,
                *args,
                _forward=original,
                _name=name,
                _module=module,
                **kwargs,
            ):
                features = input.features
                dtype = torch.float16 if features.is_cuda else torch.float32
                sparse = input.replace_feature(features.to(dtype=dtype))
                residual = (
                    add_input.replace_feature(add_input.features.to(dtype=dtype))
                    if add_input is not None
                    else None
                )
                compute_weight = weight.to(dtype=dtype)
                compute_bias = bias.to(dtype=dtype) if bias is not None else None
                debug = not getattr(_module, "_gr00t_sparse_precision_logged", False) and any(
                    os.environ.get(key, "").lower() in {"1", "true", "yes"}
                    for key in ("GR00T_DTYPE_DEBUG", "GROOT_DEBUG_CONCERTO_PRECISION")
                )
                if debug:
                    LOGGER.info(
                        "Concerto sparse boundary [%s] training=%s features=%s->%s "
                        "weight=%s->%s bias=%s->%s",
                        _name,
                        training,
                        features.dtype,
                        sparse.features.dtype,
                        weight.dtype,
                        compute_weight.dtype,
                        bias.dtype if bias is not None else None,
                        compute_bias.dtype if compute_bias is not None else None,
                    )
                # Explicitly cover eval, native/implicit GEMM, 1x1, and bias
                # addition outside the custom autograd function's AMP scope.
                with torch.autocast(features.device.type, enabled=False):
                    output = _forward(
                        training, sparse, compute_weight, compute_bias, residual, *args, **kwargs
                    )
                if debug:
                    LOGGER.info(
                        "Concerto sparse boundary [%s] output=%s", _name, output.features.dtype
                    )
                    _module._gr00t_sparse_precision_logged = True
                return output

            module._conv_forward = conv_with_supported_precision
            module._gr00t_sparse_precision = True

    def _configure_zero3_sparse_conv_leaves(self) -> None:
        """Keep spconv weights gathered through sparse backward under ZeRO-3.

        The official PointSequential calls SubMConv3d with a SparseConvTensor.
        DeepSpeed cannot find tensors inside that custom input/output type, so
        hooks on the conv itself cannot re-gather its partitioned weight for
        backward. Its PointSequential parent accepts/returns a dict-like Point
        containing ``feat``; making that parent a leaf lets ZeRO gather the
        small CPE block as one unit and hook its visible feature tensor.
        """
        for name, module in self.backbone.named_modules():
            if not name.endswith(".cpe") or module.__class__.__name__ != "PointSequential":
                continue
            if any(child.__class__.__module__.startswith("spconv.") for child in module.children()):
                # DeepSpeed's set_z3_leaf_module sets this same attribute.
                # Avoid importing DeepSpeed for non-DeepSpeed inference.
                module._z3_leaf = True
                original_forward = module.forward

                def forward_with_synced_sparse_input(point, *, _forward=original_forward):
                    # ZeRO's input hook replaces Point.feat with a tensor that
                    # triggers its post-backward hook. Concerto's parallel
                    # SparseConvTensor must use that same tensor, or the sparse
                    # path bypasses the hook and releases weights too early.
                    sparse = point.sparse_conv_feat
                    if sparse.features is not point.feat:
                        point.sparse_conv_feat = sparse.replace_feature(point.feat)
                    return _forward(point)

                # Keep the original module/parameter hierarchy and state-dict
                # names; only adapt the operation boundary.
                module.forward = forward_with_synced_sparse_input

    def validate_pretrained_backbone(self, source: str) -> None:
        """Validate weights loaded externally or embedded in a GR00T checkpoint."""
        parameters = list(self.backbone.named_parameters())
        if not parameters:
            raise RuntimeError(f"Concerto {source} loaded no pretrained parameters.")
        meta_parameters = [name for name, parameter in parameters if parameter.is_meta]
        meta_buffers = [name for name, buffer in self.backbone.named_buffers() if buffer.is_meta]
        if meta_parameters or meta_buffers:
            raise RuntimeError(
                f"Concerto {source} left meta tensors unresolved: "
                f"parameters={meta_parameters[:10]}, buffers={meta_buffers[:10]}"
            )
        if not any(
            torch.count_nonzero(parameter.detach()).item() > 0 for _, parameter in parameters
        ):
            raise RuntimeError(f"Concerto {source} loaded no non-zero pretrained parameters.")
        self.pretrained_loaded = True
        self._mark_hf_initialized()

    @staticmethod
    def _load_official_backbone(
        *,
        model_name: str,
        repo_id: str,
        checkpoint_path: str | None,
        download_root: str | None,
        enable_flash: bool | None,
        memory_map_checkpoint: bool = False,
    ) -> tuple[nn.Module, int, str, int, dict]:
        concerto = ConcertoPointEncoder._import_concerto()

        checkpoint_name = str(Path(checkpoint_path).expanduser()) if checkpoint_path else model_name
        custom_config = {}
        if enable_flash is not None:
            custom_config["enable_flash"] = enable_flash
        try:
            if memory_map_checkpoint:
                from huggingface_hub import hf_hub_download

                from gr00t.utils.checkpoint_io import load_torch_checkpoint
                from gr00t.utils.model_load_diagnostics import model_load_phase

                # Same official ckpt_only resolution/config policy, with PyTorch's
                # file-backed storages. Do not create an upstream model as well.
                path = (
                    Path(checkpoint_path).expanduser()
                    if checkpoint_path
                    else Path(
                        hf_hub_download(
                            repo_id=repo_id,
                            filename=f"{model_name}.pth",
                            repo_type="model",
                            revision="main",
                            local_dir=download_root or os.path.expanduser("~/.cache/concerto/ckpt"),
                        )
                    )
                )
                with model_load_phase("Concerto.read", checkpoint=path):
                    checkpoint = load_torch_checkpoint(path, weights_only=True)
                checkpoint["config"].update(custom_config)
            else:
                checkpoint = concerto.load(
                    checkpoint_name,
                    repo_id=repo_id,
                    download_root=download_root,
                    custom_config=custom_config or None,
                    ckpt_only=True,
                )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load requested pretrained Concerto checkpoint {checkpoint_name!r} "
                f"from {repo_id!r}. No random-init fallback is allowed."
            ) from exc

        if not isinstance(checkpoint, dict) or not {"config", "state_dict"}.issubset(checkpoint):
            raise RuntimeError(
                "Official Concerto checkpoint must contain 'config' and 'state_dict' entries."
            )
        config = dict(checkpoint["config"])
        output_dim = ConcertoPointEncoder._output_dim_from_config(config)

        state_dict = checkpoint["state_dict"]
        checkpoint_meta_tensors = [
            name
            for name, tensor in state_dict.items()
            if isinstance(tensor, torch.Tensor) and tensor.is_meta
        ]
        if checkpoint_meta_tensors:
            raise RuntimeError(
                f"Concerto checkpoint itself contains meta tensors: {checkpoint_meta_tensors[:10]}"
            )

        try:
            log_model_load_memory("before Concerto architecture construction")
            # Transformers' empty-weight context can force registered parameters back onto
            # meta even inside this explicit device context. Construct on CPU first, then
            # materialize below if that registration hook was active.
            with torch.device("cpu"):
                model = concerto.model.PointTransformerV3(**config)
            log_model_load_memory("after Concerto architecture construction")

            construction_meta_parameters = [
                name for name, parameter in model.named_parameters() if parameter.is_meta
            ]
            construction_meta_buffers = [
                name for name, buffer in model.named_buffers() if buffer.is_meta
            ]
            checkpointless_meta_buffers = [
                name for name in construction_meta_buffers if name not in state_dict
            ]
            if checkpointless_meta_buffers:
                raise RuntimeError(
                    "Concerto construction produced meta buffers that are not available in the "
                    f"checkpoint: {checkpointless_meta_buffers[:10]}"
                )

            materialized_from_meta = bool(construction_meta_parameters or construction_meta_buffers)
            if materialized_from_meta:
                # Preserve initialized non-persistent buffers, which are absent from state_dict.
                preserved_buffers = {
                    name: buffer.detach().clone()
                    for name, buffer in model.named_buffers()
                    if not buffer.is_meta and name not in state_dict
                }
                model.to_empty(device=torch.device("cpu"))
                materialized_buffers = dict(model.named_buffers())
                with torch.no_grad():
                    for name, value in preserved_buffers.items():
                        materialized_buffers[name].copy_(value)

            log_model_load_memory("before Concerto checkpoint load")
            incompatible = model.load_state_dict(
                state_dict,
                strict=False,
                # assign=True re-registers parameters and the outer Transformers hook sends
                # them back to meta. Once materialized, copy checkpoint values in place instead.
                assign=not materialized_from_meta,
            )
            log_model_load_memory("after Concerto checkpoint load")
        except Exception as exc:
            raise RuntimeError(
                f"Could not construct Concerto from pretrained checkpoint {checkpoint_name!r}."
            ) from exc
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Concerto checkpoint is incompatible with its declared architecture: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}."
            )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count == 0:
            raise RuntimeError("Concerto checkpoint loaded no pretrained parameters.")
        meta_parameters = [
            name for name, parameter in model.named_parameters() if parameter.is_meta
        ]
        if meta_parameters:
            raise RuntimeError(
                "Concerto checkpoint loading left meta parameters unresolved: "
                f"{meta_parameters[:10]}"
            )
        meta_buffers = [name for name, buffer in model.named_buffers() if buffer.is_meta]
        if meta_buffers:
            raise RuntimeError(
                f"Concerto checkpoint loading left meta buffers unresolved: {meta_buffers[:10]}"
            )
        if not any(
            torch.count_nonzero(parameter.detach()).item() > 0 for parameter in model.parameters()
        ):
            raise RuntimeError("Concerto checkpoint loaded no non-zero pretrained parameters.")
        del state_dict
        del checkpoint
        return model, output_dim, checkpoint_name, parameter_count, config

    @staticmethod
    def read_architecture_config(
        *,
        model_name,
        checkpoint_path=None,
        repo_id="Pointcept/Concerto",
        download_root=None,
        enable_flash=None,
    ):
        """Read actual checkpoint metadata without constructing a second backbone."""
        from pathlib import Path

        from gr00t.utils.checkpoint_io import load_torch_checkpoint

        if checkpoint_path:
            path = Path(checkpoint_path).expanduser()
        else:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{model_name}.pth",
                repo_type="model",
                revision="main",
                local_dir=download_root or os.path.expanduser("~/.cache/concerto/ckpt"),
            )
        checkpoint = load_torch_checkpoint(path, weights_only=True)
        config = dict(checkpoint["config"])
        if enable_flash is not None:
            config["enable_flash"] = enable_flash
        ConcertoPointEncoder._output_dim_from_config(config)
        return config

    def validate_stage_mapping(self, names):
        from gr00t.model.extension.multilevel_local_sidecar import resolve_point_stages

        names, dimensions = resolve_point_stages(self.architecture_config, names, len(names))
        encoder = getattr(self.backbone, "enc", None)
        modules = dict(encoder.named_children()) if encoder is not None else {}
        if any(name not in modules for name in names):
            raise ValueError(
                f"Concerto encoder modules {list(modules)} do not contain stages {names}"
            )
        return names, dimensions

    def forward_multilevel(self, points, point_mask=None, *, stage_names):
        """Extract stage boundaries in one backbone forward, preserving gradients.

        Decoder unpooling can mutate parent Point objects. Snapshot only the
        selected tensor fields at each boundary; never retain Point objects.
        Hooks are scoped to this call and are removed even on an exception.
        """
        from gr00t.model.extension.multilevel_local_sidecar import snapshot_level

        names, dimensions = self.validate_stage_mapping(stage_names)
        levels, handles = {}, []

        def capture(name, dim):
            def hook(module, args, output):
                levels[name] = snapshot_level(output, dim)

            return hook

        try:
            modules = dict(self.backbone.enc.named_children())
            for name, dim in zip(names, dimensions):
                handles.append(modules[name].register_forward_hook(capture(name, dim)))
            self.forward(points, point_mask)
        finally:
            for handle in handles:
                handle.remove()
        if set(levels) != set(names):
            raise RuntimeError(f"Concerto did not emit all requested encoder stages: {names}")
        return {name: levels[name] for name in names}

    def backbone_parameters(self):
        """Return only pretrained backbone parameters for optimizer LR grouping."""
        return self.backbone.parameters()

    def adapter_parameters(self):
        """Return newly initialized GR00T projection parameters."""
        return self.projection.parameters()

    def set_trainable(self, *, backbone: bool, projection: bool) -> None:
        """Freeze pretrained Concerto independently from its GR00T projection."""
        self.backbone.requires_grad_(backbone)
        self.projection.requires_grad_(projection)

    def _prepare_batch(
        self, points: torch.Tensor, point_mask: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor | float], list[int]]:
        coords = []
        features = []
        grid_coords = []
        counts = []
        for batch_index in range(points.shape[0]):
            scene = points[batch_index, point_mask[batch_index]]
            if scene.shape[0] == 0:
                raise ValueError(f"Concerto scene {batch_index} contains no valid points.")
            coord = scene[:, :3]
            coord_min = coord.amin(dim=0)
            coord_max = coord.amax(dim=0)
            shift = torch.stack(
                (
                    (coord_min[0] + coord_max[0]) / 2,
                    (coord_min[1] + coord_max[1]) / 2,
                    coord_min[2],
                )
            )
            coord = coord - shift
            grid_coord = torch.floor(coord / self.grid_size).to(torch.int32)
            grid_coord = grid_coord - grid_coord.amin(dim=0)
            color = scene[:, 3:6] / 255.0
            normal = torch.zeros_like(coord)
            coords.append(coord)
            grid_coords.append(grid_coord)
            features.append(torch.cat((coord, color, normal), dim=-1))
            counts.append(int(scene.shape[0]))

        count_tensor = torch.tensor(counts, dtype=torch.long, device=points.device)
        offset = count_tensor.cumsum(dim=0)
        batch = torch.arange(points.shape[0], device=points.device).repeat_interleave(count_tensor)
        return {
            "coord": torch.cat(coords, dim=0),
            "grid_coord": torch.cat(grid_coords, dim=0),
            "feat": torch.cat(features, dim=0),
            "offset": offset,
            "batch": batch,
            "grid_size": self.grid_size,
        }, counts

    @staticmethod
    def _pad_tokens(
        flat_tokens: torch.Tensor, counts: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_tokens = max(counts)
        batch_size = len(counts)
        tokens = flat_tokens.new_zeros((batch_size, max_tokens, flat_tokens.shape[-1]))
        mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool, device=flat_tokens.device)
        start = 0
        for batch_index, count in enumerate(counts):
            tokens[batch_index, :count] = flat_tokens[start : start + count]
            mask[batch_index, :count] = True
            start += count
        if start != flat_tokens.shape[0]:
            raise RuntimeError(
                f"Concerto output offsets account for {start} tokens, got {flat_tokens.shape[0]}."
            )
        return tokens, mask

    def forward(
        self,
        points: torch.Tensor,
        point_mask: Optional[torch.Tensor] = None,
        *,
        return_mask: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if points.ndim != 3 or points.shape[-1] != 6:
            raise ValueError(
                f"Concerto points must have shape [B, N, 6], got {tuple(points.shape)}."
            )
        # Preserve metric XYZ and voxel boundaries in FP32. In particular,
        # never center/floor BF16 coordinates supplied by a mixed-precision
        # caller, and keep these operations outside the outer AMP region.
        geometry_context = torch.autocast(points.device.type, enabled=False)
        with geometry_context:
            geometry_points = points.to(dtype=torch.float32)
            if not torch.isfinite(geometry_points).all():
                raise ValueError("Concerto points contain NaN or Inf.")
        if point_mask is None:
            point_mask = torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
        elif point_mask.shape != points.shape[:2]:
            raise ValueError(
                f"point_mask must have shape {tuple(points.shape[:2])}, "
                f"got {tuple(point_mask.shape)}."
            )
        else:
            point_mask = point_mask.to(device=points.device, dtype=torch.bool)

        with torch.autocast(points.device.type, enabled=False):
            concerto_input, _input_counts = self._prepare_batch(geometry_points, point_mask)
        log_precision = not self._precision_diagnostics_logged and any(
            os.environ.get(name, "").lower() in {"1", "true", "yes"}
            for name in ("GR00T_DTYPE_DEBUG", "GROOT_DEBUG_CONCERTO_PRECISION")
        )
        if log_precision:
            sparse_weight_dtype = next(
                (
                    module.weight.dtype
                    for module in self.backbone.modules()
                    if module.__class__.__name__.startswith("SubMConv")
                ),
                None,
            )
            LOGGER.info(
                "Concerto precision before backbone: raw_points=%s geometry=%s feat=%s coord=%s "
                "grid_coord=%s offset=%s batch=%s sparse_weight=%s outer_cuda_amp=%s "
                "outer_cuda_amp_dtype=%s",
                points.dtype,
                geometry_points.dtype,
                concerto_input["feat"].dtype,
                concerto_input["coord"].dtype,
                concerto_input["grid_coord"].dtype,
                concerto_input["offset"].dtype,
                concerto_input["batch"].dtype,
                sparse_weight_dtype,
                torch.is_autocast_enabled("cuda"),
                torch.get_autocast_dtype("cuda"),
            )

        # Dense Concerto operations use local CUDA FP16 autocast. Each sparse
        # convolution also casts its features/weight/bias explicitly, because
        # spconv's eval path bypasses the AMP-decorated autograd function.
        # Registered parameter storage and ZeRO ownership stay unchanged.
        autocast_context = (
            torch.autocast("cuda", dtype=torch.float16) if points.is_cuda else nullcontext()
        )
        with autocast_context:
            if log_precision:
                LOGGER.info(
                    "Concerto backbone CUDA autocast: enabled=%s dtype=%s",
                    torch.is_autocast_enabled("cuda"),
                    torch.get_autocast_dtype("cuda"),
                )
            output = self.backbone(concerto_input)
        if not hasattr(output, "feat") or not hasattr(output, "offset"):
            raise RuntimeError("Concerto backbone output must expose .feat and .offset tensors.")
        output_counts = torch.diff(
            torch.cat((output.offset.new_zeros(1), output.offset.to(torch.long)))
        ).tolist()
        # DeepSpeed BF16 mode casts projection weights to BF16, while the
        # sparse backbone returns FP16. Match the registered projection weight
        # before the dense layer; this differentiable cast preserves gradients.
        projection_input = output.feat.to(dtype=self.projection[0].weight.dtype)
        point_tokens = self.projection(projection_input)
        if log_precision:
            LOGGER.info(
                "Concerto precision after backbone: feat=%s projection_input=%s "
                "projection_weight=%s point_tokens=%s",
                output.feat.dtype,
                projection_input.dtype,
                self.projection[0].weight.dtype,
                point_tokens.dtype,
            )
            self._precision_diagnostics_logged = True
        point_tokens, output_mask = self._pad_tokens(point_tokens, output_counts)
        if return_mask:
            return point_tokens, output_mask
        return point_tokens


def build_point_encoder(
    encoder_type: str = "pointnet2",
    **encoder_kwargs,
) -> PointNet2Encoder | PointTransformerEncoder | ConcertoPointEncoder:
    """Build one of the lightweight point encoder backends."""
    normalized_type = encoder_type.lower().replace("-", "_")
    if normalized_type in {"pointnet", "pointnet2", "pointnet++"}:
        return PointNet2Encoder(**encoder_kwargs)
    if normalized_type in {"point_transformer", "transformer"}:
        return PointTransformerEncoder(**encoder_kwargs)
    if normalized_type in ConcertoPointEncoder.SUPPORTED_MODELS:
        return ConcertoPointEncoder(model_name=normalized_type, **encoder_kwargs)
    raise ValueError(
        f"Unknown point encoder type {encoder_type!r}. Expected 'pointnet2', "
        "'point_transformer', 'concerto_small', or 'concerto_base'."
    )


class PointEncoder(PointNet2Encoder):
    """Backward-compatible PointNet++ replacement for the old ``PointEncoder``.

    The argument order matches the former point-wise MLP encoder while the
    implementation now uses :class:`PointNet2Encoder`.
    """

    def __init__(
        self,
        input_dim: int = 6,
        token_dim: int = 256,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            point_dim=token_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )


__all__ = [
    "PointEncoder",
    "ConcertoPointEncoder",
    "PointNet2Encoder",
    "PointTransformerEncoder",
    "build_point_encoder",
]

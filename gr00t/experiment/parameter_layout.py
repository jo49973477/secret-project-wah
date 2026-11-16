"""Initialization-only parameter storage diagnostics (never logs weight values)."""

import logging

import torch


def audit_parameter_layout(model: torch.nn.Module) -> list[dict]:
    entries = list(model.named_parameters(remove_duplicate=False))
    storage_names = {}
    for name, parameter in entries:
        if not parameter.is_meta and parameter.numel():
            key = (parameter.device, parameter.untyped_storage().data_ptr())
            storage_names.setdefault(key, []).append(name)
    suspicious = []
    for name, parameter in entries:
        if parameter.is_meta:
            continue
        shared = (
            storage_names.get((parameter.device, parameter.untyped_storage().data_ptr()), [])
            if parameter.numel()
            else []
        )
        if (
            not parameter.is_contiguous()
            or (
                parameter.dtype == torch.bfloat16
                and parameter.ndim > 0
                and parameter.stride(-1) != 1
            )
            or parameter.storage_offset()
            or parameter._base is not None
            or len(shared) > 1
            or parameter.untyped_storage().nbytes() != parameter.numel() * parameter.element_size()
        ):
            row = dict(
                parameter_name=name,
                shape=tuple(parameter.shape),
                stride=parameter.stride(),
                dtype=str(parameter.dtype),
                is_contiguous=parameter.is_contiguous(),
                requires_grad=parameter.requires_grad,
                storage_offset=parameter.storage_offset(),
                shared_storage=shared,
                is_view=parameter._base is not None,
                storage_nbytes=parameter.untyped_storage().nbytes(),
            )
            logging.warning("parameter-layout %s", row)
            suspicious.append(row)
    logging.info(
        "parameter-layout audited %d parameter names; suspicious=%d", len(entries), len(suspicious)
    )
    return suspicious

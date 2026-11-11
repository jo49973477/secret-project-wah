"""Offline post-rollout analysis for UniVTAC evaluations."""

from .generator import generate_report
from .parser import load_run


__all__ = ["generate_report", "load_run"]

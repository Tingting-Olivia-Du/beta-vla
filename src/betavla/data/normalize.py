"""Normalization utilities for actions and states.

Adapted from openpi/src/openpi/shared/normalize.py.
Uses plain dataclasses + numpy to avoid pydantic/numpydantic dependency.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class NormStats:
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray | None = None
    q99: np.ndarray | None = None


def normalize_quantile(x: np.ndarray, stats: NormStats) -> np.ndarray:
    """Map x to [-1, 1] using quantile stats."""
    if stats.q01 is None or stats.q99 is None:
        raise ValueError("NormStats must have q01 and q99 for quantile normalization")
    q01 = np.asarray(stats.q01, dtype=np.float32)
    q99 = np.asarray(stats.q99, dtype=np.float32)
    span = (q99 - q01) + 1e-6
    return (np.asarray(x, dtype=np.float32) - q01) / span * 2.0 - 1.0


def unnormalize_quantile(x: np.ndarray, stats: NormStats) -> np.ndarray:
    """Inverse of normalize_quantile."""
    if stats.q01 is None or stats.q99 is None:
        raise ValueError("NormStats must have q01 and q99 for quantile unnormalization")
    q01 = np.asarray(stats.q01, dtype=np.float32)
    q99 = np.asarray(stats.q99, dtype=np.float32)
    span = (q99 - q01) + 1e-6
    return (np.asarray(x, dtype=np.float32) + 1.0) / 2.0 * span + q01


def load_norm_stats(path: str | Path | None) -> dict[str, NormStats] | None:
    """Load norm_stats.json. Returns None if path is None or file missing."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    raw: dict = json.loads(p.read_text(encoding="utf-8"))
    # Support two JSON layouts:
    #   {"norm_stats": {"action": {...}, "state": {...}}}   (openpi format)
    #   {"action": {...}, "state": {...}}                   (flat format)
    if "norm_stats" in raw:
        raw = raw["norm_stats"]
    result: dict[str, NormStats] = {}
    for key, val in raw.items():
        result[key] = NormStats(
            mean=np.array(val.get("mean", []), dtype=np.float32),
            std=np.array(val.get("std", []), dtype=np.float32),
            q01=np.array(val["q01"], dtype=np.float32) if "q01" in val else None,
            q99=np.array(val["q99"], dtype=np.float32) if "q99" in val else None,
        )
    return result

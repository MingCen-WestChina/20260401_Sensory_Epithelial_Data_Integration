"""Shared path and h5AD helpers for manuscript figure scripts."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


def repository_root(script_file: str | Path) -> Path:
    """Return the repository root for a script in figures/<figure>/ or analysis/<step>/."""
    return Path(script_file).resolve().parents[2]


def environment_path(variable: str, default: Path) -> Path:
    value = os.environ.get(variable)
    return Path(value).expanduser().resolve() if value else default.resolve()


def data_directory(script_file: str | Path) -> Path:
    root = repository_root(script_file)
    return environment_path("COCHLEA_DATA_DIR", root / "Data")


def results_directory(script_file: str | Path) -> Path:
    root = repository_root(script_file)
    return environment_path("COCHLEA_RESULTS_DIR", root / "results")


def resolve_input_path(*candidates: Path) -> Path:
    """Return the first existing candidate, preserving a primary error path."""
    if not candidates:
        raise ValueError("At least one input candidate is required.")
    for candidate in candidates:
        candidate = Path(candidate)
        if candidate.is_file():
            return candidate
    return Path(candidates[0])


def decode(values) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def read_categorical(handle, key: str) -> tuple[np.ndarray, list[str]]:
    """Read a categorical or string obs column from an h5AD file opened by h5py."""
    node = handle[f"obs/{key}"]
    if hasattr(node, "keys") and "categories" in node and "codes" in node:
        categories = np.asarray(decode(node["categories"][()]), dtype=object)
        codes = np.asarray(node["codes"][()], dtype=int)
        values = np.full(codes.shape, None, dtype=object)
        valid = codes >= 0
        values[valid] = categories[codes[valid]]
        return values, categories.tolist()
    values = np.asarray(decode(node[()]), dtype=object)
    return values, list(dict.fromkeys(values.tolist()))


def configure_matplotlib() -> str:
    """Configure an available sans-serif font while preserving editable PDF text."""
    requested = os.environ.get("COCHLEA_FONT_FAMILY", "Arial")
    font_file = font_manager.findfont(
        font_manager.FontProperties(family=requested),
        fallback_to_default=True,
    )
    family = font_manager.FontProperties(fname=font_file).get_name()
    plt.rcParams.update(
        {
            "font.family": family,
            "font.sans-serif": [family],
            "pdf.fonttype": 42,
            "pdf.use14corefonts": False,
            "pdf.compression": 9,
            "ps.fonttype": 42,
        }
    )
    return family


def integer_ticks(limits, step: int = 5) -> np.ndarray:
    start = int(np.ceil(limits[0] / step) * step)
    stop = int(np.floor(limits[1] / step) * step)
    return np.arange(start, stop + step, step)

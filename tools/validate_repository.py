#!/usr/bin/env python3
"""Run lightweight, non-training checks on the public repository.

This validator deliberately does not execute scVI/scANVI or Monocle analyses.
It checks Python and R syntax, repository structure, and common traces of local
development paths or embedded credentials.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tmp",
    ".venv",
    "__pycache__",
    "env",
    "temp",
    "tmp",
    "venv",
}
REQUIRED_PATHS = (
    "README.md",
    ".gitignore",
    "config/cell_type_markers.xlsx",
    "analysis/01_stage_clustering",
    "analysis/02_scanvi_integration",
    "analysis/03_embryonic_trajectory",
    "analysis/04_hair_cell_trajectory",
    "analysis/05_mouse_ush2a",
    "analysis/06_macaque_ush2a",
    "figures/fig1",
    "figures/fig2",
    "figures/fig4",
)

LOCAL_PATH_PATTERNS = {
    "Windows absolute path": re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]"),
    "Unix home path": re.compile(r"(?:^|[\"'])/(?:home|Users)/"),
    "private source-tree reference": re.compile(
        r"Scripts_org|Figure in article|Scirpts_confirm|\.codex", re.IGNORECASE
    ),
}
SECRET_ASSIGNMENT = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|secret|password)\s*=\s*[\"'][^\"']+[\"']"
)


def relative(path: Path) -> str:
    return path.relative_to(REPOSITORY_ROOT).as_posix()


def source_files(pattern: str) -> list[Path]:
    """Return repository source files while ignoring caches and test artifacts."""
    return sorted(
        path
        for path in REPOSITORY_ROOT.rglob(pattern)
        if not EXCLUDED_DIRECTORY_NAMES.intersection(path.relative_to(REPOSITORY_ROOT).parts)
    )


def check_structure(errors: list[str]) -> None:
    for item in REQUIRED_PATHS:
        if not (REPOSITORY_ROOT / item).exists():
            errors.append(f"missing required path: {item}")


def check_python_syntax(errors: list[str]) -> int:
    files = source_files("*.py")
    for path in files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            errors.append(f"Python syntax/read error in {relative(path)}: {exc}")
    return len(files)


def check_notebook_syntax(errors: list[str]) -> tuple[int, int]:
    notebooks = source_files("*.ipynb")
    code_cells = 0
    for path in notebooks:
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook.get("cells", [])):
                if cell.get("cell_type") != "code":
                    continue
                code_cells += 1
                source = "".join(cell.get("source", []))
                try:
                    ast.parse(source, filename=f"{path} cell {index}")
                except SyntaxError as exc:
                    errors.append(
                        f"Notebook syntax error in {relative(path)} cell {index}: {exc}"
                    )
        except (OSError, ValueError, UnicodeError) as exc:
            errors.append(f"Notebook read/JSON error in {relative(path)}: {exc}")
    return len(notebooks), code_cells


def find_rscript() -> str | None:
    configured = os.environ.get("COCHLEA_RSCRIPT")
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return str(candidate)
    return shutil.which("Rscript")


def check_r_syntax(errors: list[str], warnings: list[str]) -> int:
    files = source_files("*.R")
    if not files:
        return 0
    rscript = find_rscript()
    if rscript is None:
        warnings.append(
            "Rscript was not found; set COCHLEA_RSCRIPT to enable R syntax checks."
        )
        return len(files)

    expression = (
        "args <- commandArgs(trailingOnly=TRUE); status <- 0L; "
        "for (f in args) tryCatch(parse(file=f), error=function(e) { "
        "message(f, ': ', conditionMessage(e)); status <<- 1L }); quit(status=status)"
    )
    completed = subprocess.run(
        [rscript, "--vanilla", "-e", expression, *map(str, files)],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        errors.append(f"R syntax check failed: {detail}")
    return len(files)


def check_portability_and_secrets(errors: list[str]) -> int:
    files = sorted(
        path
        for suffix in ("*.py", "*.R", "*.ipynb")
        for path in source_files(suffix)
        if path.resolve() != Path(__file__).resolve()
    )
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            errors.append(f"Could not scan {relative(path)}: {exc}")
            continue
        for label, pattern in LOCAL_PATH_PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                errors.append(f"{label} in {relative(path)}:{line}")
        for match in SECRET_ASSIGNMENT.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            errors.append(f"possible embedded credential in {relative(path)}:{line}")
    return len(set(files))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-r",
        action="store_true",
        help="Skip R parsing even when Rscript is available.",
    )
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    check_structure(errors)
    python_files = check_python_syntax(errors)
    notebooks, code_cells = check_notebook_syntax(errors)
    scanned_files = check_portability_and_secrets(errors)
    r_files = 0 if args.skip_r else check_r_syntax(errors, warnings)

    print(
        "Checked "
        f"{python_files} Python files, {r_files} R files, "
        f"{notebooks} notebooks ({code_cells} code cells), and "
        f"{scanned_files} code files for portability/secrets."
    )
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        print(f"Validation failed with {len(errors)} error(s).")
        return 1
    print("Validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

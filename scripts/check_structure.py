"""Enforce the repository layout rules from RULES.md section 4.4.

Run with ``uv run scripts/check_structure.py``. Exits 1 and lists every
violation if any check fails.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "pocketslam"
MAX_TRACKED_BYTES = 5 * 1024 * 1024

REQUIRED = [
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    "RULES.md",
    "README.md",
    "config/pre-commit.yaml",
    "config/env.example",
    "docker/Dockerfile.dockerignore",
    ".gitignore",
    "compose.yaml",
    "docker/Dockerfile",
    "scripts/check_structure.py",
    "scripts/gen_dev_cert.sh",
    "web/index.html",
    "web/app.js",
    "src/pocketslam/__init__.py",
    "src/pocketslam/__main__.py",
    "src/pocketslam/config.py",
    "src/pocketslam/pipeline.py",
    "src/pocketslam/core/types.py",
    "src/pocketslam/core/device.py",
    "src/pocketslam/server",
    "src/pocketslam/transport",
    "src/pocketslam/depth",
    "src/pocketslam/slam",
    "src/pocketslam/viz",
    "tests",
]

FORBIDDEN_FILES = ["requirements.txt", "setup.py", "setup.cfg"]
# Hidden entries allowed in the repo root. Each one must live there:
# git, GitHub, uv (venv, Python pin) and the shared tool cache.
ROOT_HIDDEN_ALLOWED = {".git", ".github", ".venv", ".cache", ".gitignore", ".python-version"}
DEVICE_MODULE = PKG / "core" / "device.py"
DEVICE_PATTERN = re.compile(r"""\.cuda\(|["']cuda["']|["']mps["']|torch\.device\(""")
PRINT_PATTERN = re.compile(r"(?<![\w.])print\(")
NO_TEST_NEEDED = {"__init__.py", "__main__.py"}


def check_required() -> list[str]:
    """Rule 1: required files and directories exist."""
    return [f"missing required path: {p}" for p in REQUIRED if not (ROOT / p).exists()]


def check_init_files() -> list[str]:
    """Rule 2: every package directory has an __init__.py."""
    return [
        f"package without __init__.py: {d.relative_to(ROOT)}"
        for d in [PKG, *PKG.rglob("*")]
        if d.is_dir() and d.name != "__pycache__" and not (d / "__init__.py").exists()
    ]


def check_tests_mirror_src() -> list[str]:
    """Rule 3: each module has tests/<pkg>/test_<mod>.py."""
    errors = []
    for module in PKG.rglob("*.py"):
        if module.name in NO_TEST_NEEDED:
            continue
        rel = module.relative_to(PKG)
        expected = ROOT / "tests" / rel.parent / f"test_{rel.name}"
        if not expected.exists():
            errors.append(
                f"no tests for {module.relative_to(ROOT)}: expected {expected.relative_to(ROOT)}"
            )
    return errors


def check_source_patterns() -> list[str]:
    """Rules 4 and 8: backend names only in core/device.py, and no print()."""
    errors = []
    for module in PKG.rglob("*.py"):
        rel = module.relative_to(ROOT)
        for lineno, line in enumerate(module.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if module != DEVICE_MODULE and DEVICE_PATTERN.search(code):
                errors.append(f"{rel}:{lineno}: backend named outside core/device.py")
            if PRINT_PATTERN.search(code):
                errors.append(f"{rel}:{lineno}: print() in src, use logging")
    return errors


def check_tracked_file_sizes() -> list[str]:
    """Rule 5: no tracked file over 5 MB."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"],  # noqa: S607 - git from PATH is intended
            cwd=ROOT,
            capture_output=True,
            check=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []  # not a git checkout (e.g. inside a Docker build context)
    errors = []
    for name in filter(None, out.split("\0")):
        path = ROOT / name
        if path.is_file() and path.stat().st_size > MAX_TRACKED_BYTES:
            errors.append(f"tracked file over 5 MB: {name} (move it to data/ or download it)")
    return errors


def check_forbidden_files() -> list[str]:
    """Rule 6: pyproject.toml is the only packaging file."""
    errors = [f"forbidden file: {name}" for name in FORBIDDEN_FILES if (ROOT / name).exists()]
    errors += [
        f"forbidden file: {p.relative_to(ROOT)}"
        for p in ROOT.glob("requirements*.txt")
        if p.name != "requirements.txt"
    ]
    return errors


def check_root_hidden() -> list[str]:
    """Rule 9: no stray dotfiles or tool caches in the repo root."""
    return [
        f"hidden entry in repo root: {p.name} (move it under config/, docker/ or .cache/)"
        for p in sorted(ROOT.iterdir())
        if p.name.startswith(".") and p.name not in ROOT_HIDDEN_ALLOWED
    ]


def check_python_version() -> list[str]:
    """Rule 7: .python-version matches requires-python."""
    pinned = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
    with (ROOT / "pyproject.toml").open("rb") as f:
        requires = tomllib.load(f)["project"]["requires-python"]
    match = re.search(r">=\s*(\d+\.\d+)", requires)
    if match is None or not pinned.startswith(match.group(1)):
        return [f".python-version is {pinned} but requires-python is {requires!r}"]
    return []


CHECKS = [
    check_required,
    check_init_files,
    check_tests_mirror_src,
    check_source_patterns,
    check_tracked_file_sizes,
    check_forbidden_files,
    check_python_version,
    check_root_hidden,
]


def main() -> int:
    errors = [error for check in CHECKS for error in check()]
    for error in errors:
        print(f"FAIL {error}")
    if errors:
        print(f"\n{len(errors)} structure violation(s). See RULES.md section 4.")
        return 1
    print(f"structure OK ({len(CHECKS)} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

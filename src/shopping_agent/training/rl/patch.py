#!/usr/bin/env python3
"""Apply or restore the vendored veRL 0.8 dynamic-sampling patch.

The patch is applied with GNU ``patch`` and identified by a marker comment left in
the patched file, so applying it is idempotent: if ``ray_trainer.py`` already
carries the marker the command is a no-op and the RL entry point can call it on
every run.  The pristine file is kept in a ``.orig`` backup next to the target,
which is what ``--restore`` uses to go back.  ``--check`` verifies the marker and
that the patched file still compiles, without touching anything.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import py_compile
import shutil
import subprocess
from importlib.metadata import PackageNotFoundError
from pathlib import Path


EXPECTED_VERL_VERSION = "0.8.0"
PATCH_MARKER = "SHOPPING_AGENT_DYNAMIC_SAMPLING_PATCH_V4"
BACKUP_SUFFIX = ".shopping-agent-dynamic-sampling.orig"
PROJECT_ROOT = Path(__file__).resolve().parents[4]
PATCH_FILE = PROJECT_ROOT / "patches/verl-0.8.0-dynamic-sampling.patch"


def resolve_installed_ray_trainer() -> Path:
    try:
        installed_version = importlib.metadata.version("verl")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "veRL is not installed in this interpreter; run "
            "`python scripts/00_environment.py setup` first"
        ) from exc
    if installed_version != EXPECTED_VERL_VERSION:
        raise RuntimeError(
            f"expected verl=={EXPECTED_VERL_VERSION}, got verl=={installed_version}"
        )

    import verl

    verl_source = Path(verl.__file__).resolve()
    expected_environment = (PROJECT_ROOT / ".venv").resolve()
    if not verl_source.is_relative_to(expected_environment):
        raise RuntimeError(f"verl.__file__ is not from the project environment: {verl_source}")

    target = verl_source.parent / "trainer" / "ppo" / "ray_trainer.py"
    if not target.is_file():
        raise RuntimeError(f"installed ray_trainer.py does not exist: {target}")
    return target.resolve()


def validate_runtime_and_target(target_override: Path | None) -> Path:
    installed_target = resolve_installed_ray_trainer()
    if target_override is None:
        return installed_target
    target = target_override.resolve()
    if not target.is_file():
        raise RuntimeError(f"target ray_trainer.py does not exist: {target}")
    return target


def backup_path(target: Path) -> Path:
    return Path(str(target) + BACKUP_SUFFIX)


def has_current_marker(target: Path) -> bool:
    return PATCH_MARKER in target.read_text(encoding="utf-8")


def verify_patched(target: Path) -> None:
    """Fail unless the target carries the applied patch marker and still compiles."""

    if not has_current_marker(target):
        raise RuntimeError(f"patched ray_trainer.py is missing marker {PATCH_MARKER}")
    py_compile.compile(str(target), doraise=True)


def apply_patch(target: Path) -> None:
    if not PATCH_FILE.is_file():
        raise RuntimeError(f"patch file is missing: {PATCH_FILE}")
    if has_current_marker(target):
        verify_patched(target)
        print(f"veRL dynamic-sampling patch already applied: {target}")
        return

    backup = backup_path(target)
    if not backup.exists():
        shutil.copy2(target, backup)

    patch_program = shutil.which("patch")
    if patch_program is None:
        raise RuntimeError(
            "required system 'patch' executable is unavailable; install it "
            "(Debian/Ubuntu: apt-get install -y patch) and re-run"
        )

    try:
        subprocess.run(
            [patch_program, "--batch", "--forward", "--silent", str(target), str(PATCH_FILE)],
            check=True,
            cwd=PROJECT_ROOT,
        )
        verify_patched(target)
    except Exception:
        shutil.copy2(backup, target)
        raise

    print(f"applied veRL dynamic-sampling patch: {target}")
    print(f"backup: {backup}")


def restore_patch(target: Path) -> None:
    if not has_current_marker(target):
        print(f"veRL ray_trainer.py is already original: {target}")
        return
    backup = backup_path(target)
    if not backup.is_file():
        raise RuntimeError(f"cannot restore without backup: {backup}")

    shutil.copy2(backup, target)
    py_compile.compile(str(target), doraise=True)
    print(f"restored original veRL ray_trainer.py: {target}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--restore",
        action="store_true",
        help="restore the original file from the automatic backup",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that the target is already patched without modifying it",
    )
    parser.add_argument(
        "--target",
        type=Path,
        help="override ray_trainer.py target for isolated patch-script tests",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if sum((args.restore, args.check)) > 1:
        raise SystemExit("--restore and --check are mutually exclusive")
    try:
        target = validate_runtime_and_target(args.target)
        if args.restore:
            restore_patch(target)
        elif args.check:
            verify_patched(target)
            print(f"verified veRL dynamic-sampling patch: {target}")
        else:
            apply_patch(target)
    except (OSError, RuntimeError, subprocess.CalledProcessError, py_compile.PyCompileError) as exc:
        raise SystemExit(f"veRL dynamic-sampling patch error: {exc}") from exc


if __name__ == "__main__":
    main()

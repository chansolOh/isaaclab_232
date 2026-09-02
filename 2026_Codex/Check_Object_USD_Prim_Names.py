"""Check whether object USD root prim names match grasp collector assumptions.

The grasp collector derives a referenced object's child prim name from the USD
filename (``bike_bell_rigid.usd`` -> ``bike_bell``).  A USD authored with a
different default/top-level prim name therefore makes FrameTransformer startup
fail.  This script finds those mismatches before collection starts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


DEFAULT_MODEL_ROOT = Path(
    "/nas/ochansol/3d_model/peel3_scan_data_2026"
)


parser = argparse.ArgumentParser(
    description="Check object USD filename/defaultPrim/top-level-prim consistency"
)
parser.add_argument(
    "paths",
    nargs="*",
    help=(
        "USD files or directories to inspect. If omitted, scan the default "
        "2026 object-model root."
    ),
)
parser.add_argument(
    "--model-root",
    default=str(DEFAULT_MODEL_ROOT),
    help="Default directory used when no paths are supplied.",
)
parser.add_argument(
    "--all-usd",
    action="store_true",
    help="Scan every .usd file in directories instead of only *_rigid.usd.",
)
parser.add_argument(
    "--show-ok",
    action="store_true",
    help="Print matching USD files too. By default only problems are printed.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from pxr import Usd


def expected_prim_name(path: Path) -> str:
    """Return the name inferred by Direct_RL_main_new_filter.py."""
    return path.stem.removesuffix("_rigid")


def collect_usd_paths() -> list[Path]:
    requested = [Path(value).expanduser() for value in args_cli.paths]
    if not requested:
        requested = [Path(args_cli.model_root).expanduser()]

    pattern = "*.usd" if args_cli.all_usd else "*_rigid.usd"
    found: set[Path] = set()
    for path in requested:
        if path.is_file():
            if path.suffix.lower() == ".usd":
                found.add(path.resolve())
            else:
                print(f"[SKIP] not a USD file: {path}")
        elif path.is_dir():
            found.update(candidate.resolve() for candidate in path.rglob(pattern))
        else:
            print(f"[MISSING] {path}")
    return sorted(found)


def inspect_usd(path: Path) -> tuple[str, str]:
    expected = expected_prim_name(path)
    try:
        stage = Usd.Stage.Open(str(path))
    except Exception as exc:
        return "OPEN_ERROR", f"expected={expected!r} error={exc}"
    if stage is None:
        return "OPEN_ERROR", f"expected={expected!r} stage=None"

    top_level_names = [prim.GetName() for prim in stage.GetPseudoRoot().GetChildren()]
    default_prim = stage.GetDefaultPrim()
    default_name = default_prim.GetName() if default_prim and default_prim.IsValid() else None
    default_children = (
        [prim.GetName() for prim in default_prim.GetChildren()]
        if default_prim and default_prim.IsValid()
        else []
    )

    details = (
        f"expected={expected!r} defaultPrim={default_name!r} "
        f"defaultChildren={default_children!r} topLevel={top_level_names!r}"
    )
    if default_name is None:
        return "NO_DEFAULT_PRIM", details
    # Referencing the USD's default prim at /objXX composes its children below
    # /objXX.  Some simpler assets instead use the object itself as defaultPrim.
    if expected != default_name and expected not in default_children:
        return "NAME_MISMATCH", details
    return "OK", details


def main() -> int:
    paths = collect_usd_paths()
    if not paths:
        print("No USD files found.")
        return 2

    counts: dict[str, int] = {}
    problem_count = 0
    print(f"Checking {len(paths)} USD file(s)...")
    for path in paths:
        status, details = inspect_usd(path)
        counts[status] = counts.get(status, 0) + 1
        if status != "OK":
            problem_count += 1
        if status != "OK" or args_cli.show_ok:
            print(f"[{status}] {path}")
            print(f"  {details}")

    summary = " ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    print(f"Summary: total={len(paths)} problems={problem_count} {summary}")
    return 1 if problem_count else 0


if __name__ == "__main__":
    try:
        exit_code = main()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)

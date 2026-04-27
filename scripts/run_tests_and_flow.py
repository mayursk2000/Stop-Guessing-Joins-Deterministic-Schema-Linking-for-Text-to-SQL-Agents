"""Run poster tests, then print the architecture flow and metrics table."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(title: str, command: list[str]) -> None:
    print("=" * 88, flush=True)
    print(title, flush=True)
    print("=" * 88, flush=True)
    subprocess.run(command, cwd=ROOT, check=True, stderr=sys.stdout)
    print(flush=True)


def main() -> None:
    run_step(
        "POSTER TESTS",
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
    )
    run_step(
        "FLOW DEMO + METRICS",
        [sys.executable, "scripts/run_flow_demo.py"],
    )


if __name__ == "__main__":
    main()

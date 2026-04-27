"""Run poster tests and emit JSON summary."""

from __future__ import annotations

import json
import sys
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    started = time.perf_counter()
    suite = unittest.defaultTestLoader.discover("tests")
    captured = StringIO()
    with redirect_stdout(captured), redirect_stderr(captured):
        result = unittest.TextTestRunner(stream=captured, verbosity=0).run(suite)
    elapsed = time.perf_counter() - started
    payload = {
        "step": "poster_tests",
        "status": "passed" if result.wasSuccessful() else "failed",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "duration_seconds": round(elapsed, 3),
        "checks": [
            "graph join policy rewrites invalid SQL joins",
            "large schema fixture has 100+ tables and relationships",
            "benchmark table improves join and execution accuracy",
            "comparison table contains Spider and BIRD references",
        ],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

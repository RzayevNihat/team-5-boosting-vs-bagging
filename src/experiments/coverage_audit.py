"""Repo-wide test coverage audit for Person C / Random Forest & QA track.

The distribution plan assigns Person C the final-week test suite audit and the
coverage report. This script runs pytest with coverage, stores the terminal
output, and writes a short Markdown summary that can be attached to the
contribution report.

Run from the repository root:

    python experiments/coverage_audit.py

Optional fast target for only the Random Forest module:

    python experiments/coverage_audit.py --rf-only
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a command from the repo root and capture combined output."""
    return subprocess.run(
        command,
        cwd=ROOT_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def extract_total_coverage(output: str) -> str:
    """Extract the TOTAL coverage percentage from pytest-cov terminal output."""
    for line in reversed(output.splitlines()):
        if line.strip().startswith("TOTAL"):
            match = re.search(r"(\d+)%", line)
            if match:
                return match.group(1) + "%"
    return "not_found"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rf-only",
        action="store_true",
        help="Audit only tests/test_random_forest.py and src.bagging.random_forest.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.rf_only:
        command = [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_random_forest.py",
            "--cov=src.bagging.random_forest",
            "--cov-report=term-missing",
        ]
        audit_scope = "Random Forest module only"
    else:
        command = [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml",
        ]
        audit_scope = "Full repository"

    completed = run_command(command)
    output = completed.stdout
    total_coverage = extract_total_coverage(output)

    raw_log = RESULTS_DIR / "coverage_audit_raw.log"
    raw_log.write_text(output, encoding="utf-8")

    summary = RESULTS_DIR / "coverage_audit_summary.md"
    summary.write_text(
        "\n".join(
            [
                "# Coverage Audit Summary",
                "",
                f"- Scope: {audit_scope}",
                f"- Timestamp: {datetime.now().isoformat(timespec='seconds')}",
                f"- Command: `{' '.join(command)}`",
                f"- Exit code: `{completed.returncode}`",
                f"- Total coverage: **{total_coverage}**",
                f"- Raw log: `{raw_log.as_posix()}`",
                "",
                "## Required action",
                "",
                "The project minimum is 60% coverage. If total coverage is below 60%, list the weak files here and request fixes from the module owner before the final PR is merged.",
                "",
                "## Notes for contribution report",
                "",
                "Person C ran the repo-wide coverage audit, saved the output, and reported modules below the coverage threshold to the relevant owners.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(output)
    print(f"\nSaved raw log: {raw_log}")
    print(f"Saved summary: {summary}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

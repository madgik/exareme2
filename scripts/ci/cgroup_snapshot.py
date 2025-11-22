#!/usr/bin/env python3
"""Emit shell exports that capture the current cgroup memory usage/peak."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from scripts.ci.report_time_stats import detect_cgroup_current_bytes
    from scripts.ci.report_time_stats import detect_cgroup_peak_bytes
else:
    from .report_time_stats import detect_cgroup_current_bytes
    from .report_time_stats import detect_cgroup_peak_bytes


def main() -> None:
    exports = []
    current = detect_cgroup_current_bytes()
    if current is not None:
        exports.append(f"export CGROUP_BASELINE_CURRENT={current}")
    peak = detect_cgroup_peak_bytes()
    if peak is not None:
        exports.append(f"export CGROUP_BASELINE_PEAK={peak}")
    if exports:
        print("\n".join(exports))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Parse `/usr/bin/time -v` output and surface the most useful numbers in both
stdout and the GitHub step summary so the resource usage is easy to find in CI.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Tuple

TIME_FIELDS = {
    "User time (seconds)": "user_seconds",
    "System time (seconds)": "system_seconds",
    "Percent of CPU this job got": "cpu_percent",
    "Elapsed (wall clock) time (h:mm:ss or m:ss)": "wall_clock",
    "Maximum resident set size (kbytes)": "max_rss_kb",
}


def parse_time_report(path: Path) -> Dict[str, str]:
    metrics: Dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if ":" not in raw:
                continue
            key, value = raw.split(":", 1)
            normalized_key = key.strip()
            if normalized_key in TIME_FIELDS:
                metrics[TIME_FIELDS[normalized_key]] = value.strip()
    return metrics


def read_first_int(path: Path) -> Optional[int]:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def resolve_cgroup_memory_paths() -> Optional[Tuple[Path, Path]]:
    """Return (current_path, peak_path) for the current cgroup, if available."""
    cgroup_file = Path("/proc/self/cgroup")
    try:
        cgroup_lines = cgroup_file.read_text(encoding="utf-8").strip().splitlines()
    except OSError:
        return None

    unified_path: Optional[str] = None
    memory_path: Optional[str] = None
    for line in cgroup_lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, path = parts
        if controllers == "":
            unified_path = path
        elif "memory" in controllers.split(","):
            memory_path = path

    if unified_path is not None:
        base = Path("/sys/fs/cgroup") / unified_path.lstrip("/")
        return (base / "memory.current", base / "memory.peak")

    if memory_path is not None:
        base = Path("/sys/fs/cgroup/memory") / memory_path.lstrip("/")
        return (base / "memory.usage_in_bytes", base / "memory.max_usage_in_bytes")

    return None


def detect_cgroup_peak_bytes() -> Optional[int]:
    """Return peak memory usage (bytes) recorded for this cgroup, if available."""
    paths = resolve_cgroup_memory_paths()
    if not paths:
        return None
    _, peak_path = paths
    return read_first_int(peak_path)


def detect_cgroup_current_bytes() -> Optional[int]:
    """Return current memory usage (bytes) recorded for this cgroup, if available."""
    paths = resolve_cgroup_memory_paths()
    if not paths:
        return None
    current_path, _ = paths
    return read_first_int(current_path)


def _safe_int(raw: Optional[str]) -> Optional[int]:
    if raw is None or raw == "":
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def _format_bytes(value: int) -> str:
    mb = value / (1024 * 1024)
    return f"{value:,} bytes ({mb:0.1f} MB)"


def _format_bytes_str(raw: str) -> str:
    parsed = _safe_int(raw)
    return _format_bytes(parsed) if parsed is not None else raw


def format_metrics(metrics: Dict[str, str]) -> Iterable[Tuple[str, str]]:
    """Yield (metric, value) pairs ready to be rendered."""
    before_current_raw = metrics.get("runner_current_before")
    before_current_int = _safe_int(before_current_raw)
    before_peak_raw = metrics.get("runner_peak_before")
    before_peak_int = _safe_int(before_peak_raw)
    if before_current_raw:
        yield ("Runner memory before test", _format_bytes_str(before_current_raw))
    if before_peak_raw:
        yield ("Runner peak before test", _format_bytes_str(before_peak_raw))
    if wall_clock := metrics.get("wall_clock"):
        yield ("Wall clock", wall_clock)
    if user := metrics.get("user_seconds"):
        yield ("User time", f"{user} s")
    if system := metrics.get("system_seconds"):
        yield ("System time", f"{system} s")
    if cpu := metrics.get("cpu_percent"):
        yield ("% CPU", cpu)
    peak_bytes_raw = metrics.get("job_peak_bytes")
    peak_bytes_int = _safe_int(peak_bytes_raw)
    if rss := metrics.get("max_rss_kb"):
        try:
            rss_kb = int(float(rss))
            rss_mb = rss_kb / 1024
            rss_value = f"{rss_kb} kB ({rss_mb:0.1f} MB)"
        except ValueError:
            rss_value = rss
        yield ("Peak RSS", rss_value)
    if peak_bytes_raw:
        yield ("Runner cgroup peak", _format_bytes_str(peak_bytes_raw))
    if peak_bytes_int is not None and before_current_int is not None:
        delta = peak_bytes_int - before_current_int
        sign = "+" if delta >= 0 else "-"
        yield ("Runner memory change vs start", f"{sign}{_format_bytes(abs(delta))}")
    if peak_bytes_int is not None and before_peak_int is not None:
        delta = peak_bytes_int - before_peak_int
        if delta > 0:
            yield ("Runner peak increase", _format_bytes(delta))
        else:
            yield ("Runner peak increase", "no change")


def append_summary(title: str, metric_rows: Iterable[Tuple[str, str]]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    rows = list(metric_rows)
    if not rows:
        return
    lines = [
        f"### {title} resource usage",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for metric, value in rows:
        lines.append(f"| {metric} | {value} |")
    lines.append("")
    with open(summary_path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def emit_stdout(title: str, metric_rows: Iterable[Tuple[str, str]]) -> None:
    rows = list(metric_rows)
    if not rows:
        print(f"{title}: no resource metrics found in time report")
        return
    print(f"{title} resource usage:")
    for metric, value in rows:
        print(f"  - {metric}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path, help="path to `/usr/bin/time -v` output")
    parser.add_argument(
        "--title",
        default="Pytest",
        help="Label used for stdout and summary headings (default: %(default)s)",
    )
    parser.add_argument(
        "--baseline-runner-current",
        help="Runner memory.current reading (bytes) collected before the test",
    )
    parser.add_argument(
        "--baseline-runner-peak",
        help="Runner memory peak (bytes) collected before the test",
    )
    args = parser.parse_args()

    metrics = parse_time_report(args.report)
    peak = detect_cgroup_peak_bytes()
    if peak is not None:
        metrics["job_peak_bytes"] = str(peak)
    if args.baseline_runner_current:
        metrics["runner_current_before"] = args.baseline_runner_current
    if args.baseline_runner_peak:
        metrics["runner_peak_before"] = args.baseline_runner_peak
    metric_rows = list(format_metrics(metrics))
    emit_stdout(args.title, metric_rows)
    append_summary(args.title, metric_rows)


if __name__ == "__main__":
    main()

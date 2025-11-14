import statistics
import sys
import time
from pathlib import Path

YE_SQL_PATH = Path("/home/YeSQL/")
if str(YE_SQL_PATH) not in sys.path:
    sys.path.insert(0, str(YE_SQL_PATH))
    sys.path.insert(0, str(Path("/home/YeSQL/YeSQL/")))

import yesql

RUN_CONFIGS = [1, 5, 20]
LOG_PATH = Path(__file__).with_name("yesql_pca_benchmark.log")

BENCH_QUERY = """cast 'string' run_on_exaflow 'pca' SELECT "row_id", "blood_cholest_ldl", "blood_cholest_tot", "blood_glucose", "blood_creat"
FROM "Stroke:3.7"."primary_data"
WHERE (("blood_cholest_ldl" IS NOT NULL AND "blood_cholest_tot" IS NOT NULL AND "blood_glucose" IS NOT NULL AND "blood_creat" IS NOT NULL) AND ("dataset" IN ('SSR')));
;"""


def _connect():
    return yesql.connect_init(
        "executor",
        "executor",
        "172.17.0.1",
        "db",
        50002,
        "monetdb",
        "/home/YeSQL/udfs",
    )


def _run_query(cursor):
    start = time.perf_counter()
    cursor.execute(BENCH_QUERY)
    cursor.fetchall()
    return time.perf_counter() - start


def _new_connection_runs(iterations: int) -> list[float]:
    durations: list[float] = []
    for _ in range(iterations):
        conn = _connect()
        cursor = conn.cursor()
        try:
            durations.append(_run_query(cursor))
        finally:
            cursor.close()
            conn.close()
    return durations


def _reused_connection_runs(iterations: int) -> list[float]:
    conn = _connect()
    cursor = conn.cursor()
    durations: list[float] = []
    try:
        for _ in range(iterations):
            durations.append(_run_query(cursor))
    finally:
        cursor.close()
        conn.close()
    return durations


def _format_stats(values: list[float]) -> str:
    avg = statistics.mean(values)
    total = sum(values)
    runs = ", ".join(f"{val:.3f}" for val in values)
    return f"avg {avg:.3f}s, total {total:.3f}s (runs: {runs})"


def test_py_conversion():
    lines: list[str] = []

    def log(message: str) -> None:
        print(message)
        lines.append(message)

    log("YeSQL PCA benchmark")
    for iterations in RUN_CONFIGS:
        new_times = _new_connection_runs(iterations)
        warm_times = _reused_connection_runs(iterations)
        log(f"Runs: {iterations}")
        log("  New connection each time: " + _format_stats(new_times))
        log("  Same connection reused:  " + _format_stats(warm_times))

    LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"Results saved to {LOG_PATH}")

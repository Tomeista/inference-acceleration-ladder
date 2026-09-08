"""MLflow logging.

Structure: one parent run per server configuration, one nested child run per
(prompt class, concurrency) cell. That shape is what makes the comparison
queryable later. A single flat run per cell would force every plot to
reconstruct which cells belonged to the same server, and the whole study is
about differences between servers holding the cell fixed.

Raw per-request records are attached to each child run as an artifact. The
aggregates answer the questions we know to ask now; the records are what a
question we have not thought of yet will need.
"""

from __future__ import annotations

import json
import math
import os
import platform
import socket
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import mlflow

DEFAULT_EXPERIMENT = "inference-acceleration"


def _clean_metrics(values: dict[str, Any]) -> dict[str, float]:
    """Drop anything MLflow cannot store as a metric.

    NaN and inf arrive whenever a cell produced no successful request. Logging
    them raises, and logging them as 0.0 would be worse than not logging them,
    because a zero p95 reads as an extraordinarily fast run.
    """
    out: dict[str, float] = {}
    for key, value in values.items():
        if value is None or isinstance(value, bool):
            continue
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(f) or math.isinf(f):
            continue
        out[key] = f
    return out


def _clean_params(values: dict[str, Any]) -> dict[str, str]:
    return {k: ("" if v is None else str(v)) for k, v in values.items()}


def environment_tags() -> dict[str, str]:
    """Host and driver facts that explain a result months later."""
    tags: dict[str, str] = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,compute_cap,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if out.returncode == 0 and out.stdout.strip():
            lines = [ln.strip() for ln in out.stdout.strip().splitlines()]
            tags["gpu"] = lines[0]
            tags["gpu_count"] = str(len(lines))
    except Exception:  # noqa: BLE001 - absent on the workstation, present on the server
        tags["gpu"] = "unavailable"
    return tags


def setup(experiment: str = DEFAULT_EXPERIMENT, tracking_uri: str | None = None) -> None:
    """Point MLflow at its store.

    Defaults to a local ./mlruns directory. Set MLFLOW_TRACKING_URI to push to a
    shared server without touching this code.
    """
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)


@contextmanager
def config_run(
    run_name: str, params: dict[str, Any], tags: dict[str, str] | None = None
) -> Iterator[Any]:
    """Parent run for one server configuration."""
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({**environment_tags(), **(tags or {})})
        mlflow.log_params(_clean_params(params))
        yield run


@contextmanager
def cell_run(run_name: str, params: dict[str, Any]) -> Iterator[Any]:
    """Nested run for one (class, concurrency) cell."""
    with mlflow.start_run(run_name=run_name, nested=True) as run:
        mlflow.log_params(_clean_params(params))
        yield run


def log_cell_results(
    metrics: dict[str, Any],
    notes: dict[str, Any],
    records: list[dict[str, Any]],
    artifact_dir: Path,
    cell_id: str,
) -> None:
    mlflow.log_metrics(_clean_metrics(metrics))

    # Notes are strings and booleans (finish reasons, validity checks), so they
    # are tags rather than metrics.
    mlflow.set_tags({f"note.{k}": json.dumps(v) for k, v in notes.items()})

    artifact_dir.mkdir(parents=True, exist_ok=True)
    records_path = artifact_dir / f"{cell_id}_requests.jsonl"
    with records_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    mlflow.log_artifact(str(records_path), artifact_path="requests")

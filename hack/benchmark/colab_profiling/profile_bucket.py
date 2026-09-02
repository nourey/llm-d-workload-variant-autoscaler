#!/usr/bin/env python3
"""Bounded fixed-concurrency V_M profiling harness for #1546.

This module implements the fixed-concurrency profiling phase authorized by
``.claude/work/1546/DECISIONS.md`` and
``.claude/work/1546/IMPLEMENTATION_BRIEF.md``, generalized to run against any
one of the three approved dataset buckets:

```
balanced
input-heavy
output-heavy
```

Exactly one bucket is profiled per run. The measurement methodology itself
is unchanged from the balanced-only implementation that was validated on
real Colab Tesla T4 hardware: fixed closed-loop concurrency, explicit
settling/measurement/drain phases, half-open ``[T0, T1)`` terminal-window
accounting, and the same raw-token ``/v1/completions`` request contract.

As of this implementation:

* the ``balanced`` bucket has been VALIDATED on real Tesla T4 hardware
  (two independent 180s confirmation runs; see the #1546 evidence record):
  ``V_M^(balanced) ~= 1274 logical token/s``;
* the ``input-heavy`` bucket has also been VALIDATED on independent real
  Tesla T4 runs: ``V_M^(input-heavy) ~= 1820 logical token/s``;
* the ``output-heavy`` bucket is NOT YET VALIDATED. Real-runtime profiling
  exposed a suspected initial-admission phase-synchronization /
  completion-wave-aliasing artifact (near-exact multiples of the target
  concurrency completing together, producing an apparently non-monotonic
  throughput curve at several concurrency points despite zero failures,
  zero preemptions, and bounded drains). See the completion-clustering
  diagnostic (``summarize_completion_clustering``) and the deterministic
  startup-ramp mechanism (``TimingConfig.ramp_admission_interval_seconds``)
  added to investigate and mitigate this before drawing any output-heavy
  capacity conclusion.

It intentionally reuses the already-accepted dataset and request-contract
layers instead of duplicating them:

* ``generate_dataset.py`` defines the approved bucket geometry
  (``DEFAULT_BUCKETS``) and the stable ``prompt_hash`` used by the dataset.
* ``run_request_smoke.py`` defines the exact raw-token ``/v1/completions``
  request contract (``request_payload``), the response-contract validator
  (``validate_response``), and the profiling-JSONL loader/validator
  (``load_profiling_records``).

This module does NOT:

* run held-out mixtures;
* claim isolated decoder V_D or physical KV-release throughput;
* make an automatic final plateau decision;
* implement mixed-workload composition, P/D profiling, a predictor, or
  autoscaling.

The resulting artifacts are raw evidence for a HUMAN REVIEW decision.
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import generate_dataset as generator
import run_request_smoke as smoke


MANIFEST_SCHEMA_VERSION = "llm-d-colab-bucket-profiling-manifest-v1"
REQUEST_RESULT_SCHEMA_VERSION = "llm-d-colab-bucket-profiling-request-result-v1"
POINT_SUMMARY_SCHEMA_VERSION = "llm-d-colab-bucket-profiling-point-summary-v1"
EXPERIMENT_SUMMARY_SCHEMA_VERSION = "llm-d-colab-bucket-profiling-summary-v1"

MANIFEST_FILENAME = "experiment_manifest.json"
REQUEST_RESULTS_FILENAME = "request_results.jsonl"
POINT_SUMMARIES_FILENAME = "point_summaries.jsonl"
SUMMARY_FILENAME = "summary.json"
VLLM_METRICS_FILENAME = "telemetry/vllm_metrics.jsonl"
GPU_METRICS_FILENAME = "telemetry/gpu_metrics.jsonl"

DEFAULT_CONCURRENCY_LADDER: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
DEFAULT_SETTLING_SECONDS = 30.0
DEFAULT_MEASUREMENT_SECONDS = 60.0
DEFAULT_DRAIN_TIMEOUT_SECONDS = 120.0
DEFAULT_METRICS_INTERVAL_SECONDS = 1.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = smoke.DEFAULT_TIMEOUT_SECONDS
DEFAULT_GPU_MEMORY_UTILIZATION = 0.90

# Deterministic initial-admission ramp (see TimingConfig.
# ramp_admission_interval_seconds and run_load_point). 0.05s keeps the ramp
# for the default concurrency ladder (max C=32) under ~1.6s, while spreading
# a C=192 initial fill over ~9.6s -- large enough to decorrelate equal-
# length-output completion waves, small relative to settling/measurement,
# and never adaptively derived from observed latency.
DEFAULT_RAMP_ADMISSION_INTERVAL_SECONDS = 0.05

# Completion-burst / phase-synchronization diagnostic defaults (D16-style
# evidence for human review; never auto-invalidates a point). 0.5s is far
# smaller than any plausible single balanced/input-heavy/output-heavy
# request latency, so a burst window this wide captures genuinely
# near-simultaneous completions (e.g. from a shared vLLM decode step)
# without conflating them with normal request-to-request latency variance
# -- e.g. a healthy ~3-4 req/s continuous stream (inter-completion gaps
# ~0.25-0.33s) contains only ~2 completions in any real 0.5s window, not
# the whole stream (see max_completions_in_fixed_window: this is a
# non-chaining fixed-width sliding window, NOT single-linkage clustering).
DEFAULT_BURST_WINDOW_SECONDS = 0.5
DEFAULT_NEAR_CONCURRENCY_BURST_THRESHOLD_FRACTION = 0.8

MAX_RESPONSE_BYTES = smoke.MAX_RESPONSE_BYTES

# The approved bucket geometry is owned by generate_dataset.py; this module
# only selects among the buckets it already defines.
BUCKETS_BY_NAME: dict[str, generator.Bucket] = {
    bucket.name: bucket for bucket in generator.DEFAULT_BUCKETS
}

# Real-hardware validation status per bucket. Only "balanced" has been run
# on a real GPU as of this implementation (see #1546 evidence). This is
# purely a documentation/traceability aid; it never gates execution.
BUCKET_VALIDATION_STATUS: dict[str, str] = {
    "balanced": (
        "VALIDATED on real Tesla T4 hardware: two independent 180s "
        "confirmation runs reproduced C=48 -> ~1228.8 tok/s and "
        "C=64 -> ~1274.3 tok/s (adjacent gain ~3.7%). "
        "V_M^(balanced) ~= 1274 logical token/s."
    ),
    "input-heavy": (
        "VALIDATED on independent real Tesla T4 hardware runs under the "
        "same monolithic non-P/D serving configuration. "
        "V_M^(input-heavy) ~= 1820 logical token/s."
    ),
    "output-heavy": (
        "NOT YET VALIDATED. Real-runtime profiling exposed a suspected "
        "initial-admission phase-synchronization / completion-wave "
        "aliasing artifact: several concurrency points completed near-"
        "exact multiples of the target concurrency (e.g. completed = "
        "2*C), producing an apparently non-monotonic throughput curve "
        "despite zero request failures, zero preemptions, and bounded "
        "drains. See the completion_clustering diagnostic in each point "
        "summary and the startup_ramp mechanism before drawing any "
        "output-heavy capacity conclusion; re-profiling on real hardware "
        "with ramping enabled has not yet been performed."
    ),
}

TransportError = smoke.TransportError
HttpResponse = smoke.HttpResponse
Transport = smoke.Transport
Probe = Callable[[str, float], HttpResponse]
GpuSampler = Callable[[], dict[str, Any]]
Clock = Callable[[], float]


class ProfilingError(ValueError):
    """Raised when profiling setup or configuration is invalid."""


def resolve_bucket(bucket_name: str) -> generator.Bucket:
    """Return the approved bucket definition for ``bucket_name``, or fail closed."""

    try:
        return BUCKETS_BY_NAME[bucket_name]
    except KeyError as error:
        raise ProfilingError(
            f"unknown bucket {bucket_name!r}; expected one of "
            f"{sorted(BUCKETS_BY_NAME)}"
        ) from error


# ---------------------------------------------------------------------------
# Dataset selection (reuses generate_dataset/run_request_smoke contracts)
# ---------------------------------------------------------------------------


def select_bucket_records(
    records: Sequence[Mapping[str, Any]],
    bucket_name: str,
) -> list[Mapping[str, Any]]:
    """Return only ``bucket_name`` profiling records, in file order.

    Rejects unknown bucket names (fail closed) rather than silently
    returning an empty list.
    """

    resolve_bucket(bucket_name)  # fail closed on an unapproved bucket name
    selected = [record for record in records if record.get("bucket") == bucket_name]
    if not selected:
        raise ProfilingError(f"no {bucket_name!r}-bucket profiling records were found")
    return selected


class PromptCycle:
    """A thread-safe, deterministic cyclic iterator over one bucket's prompts.

    Records are sorted by ``request_id`` once, at construction, so the cycle
    order does not depend on JSONL line order or dict iteration order. Each
    call to :meth:`next` hands out the next record in round-robin order and
    is reproducible from the recorded record set alone (D8). Callers are
    responsible for passing only records from a single selected bucket;
    this class does not itself filter by bucket.
    """

    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        if not records:
            raise ProfilingError("prompt cycle requires at least one record")
        self._records: tuple[Mapping[str, Any], ...] = tuple(
            sorted(records, key=lambda record: record["request_id"])
        )
        self._lock = threading.Lock()
        self._next_index = 0

    def __len__(self) -> int:
        return len(self._records)

    def next(self) -> tuple[Mapping[str, Any], int]:
        """Return ``(record, global_sequence_number)`` for the next prompt."""

        with self._lock:
            sequence = self._next_index
            self._next_index += 1
        return self._records[sequence % len(self._records)], sequence


# ---------------------------------------------------------------------------
# Server endpoints (reuses run_request_smoke's base-URL validation)
# ---------------------------------------------------------------------------


def _server_root(base_url: str) -> str:
    completions = smoke.completions_endpoint(base_url)
    return completions[: -len("/v1/completions")]


def models_endpoint(base_url: str) -> str:
    return _server_root(base_url) + "/v1/models"


def metrics_endpoint(base_url: str) -> str:
    return _server_root(base_url) + "/metrics"


def version_endpoint(base_url: str) -> str:
    return _server_root(base_url) + "/version"


def http_get(endpoint: str, timeout_seconds: float) -> HttpResponse:
    """GET an endpoint using only the standard library, bounded and fail-closed."""

    request = urllib.request.Request(endpoint, headers={"Accept": "*/*"}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
            if len(body) > MAX_RESPONSE_BYTES:
                raise TransportError(
                    f"response exceeded the {MAX_RESPONSE_BYTES}-byte safety limit"
                )
            return HttpResponse(status_code=response.status, body=body)
    except urllib.error.HTTPError as error:
        return HttpResponse(status_code=error.code, body=b"")
    except (
        urllib.error.URLError,
        TimeoutError,
        OSError,
        http.client.HTTPException,
    ) as error:
        raise TransportError(f"HTTP GET failed: {error}") from error


# ---------------------------------------------------------------------------
# Per-request execution (reuses request_payload/validate_response contracts)
# ---------------------------------------------------------------------------


def execute_profiling_request(
    record: Mapping[str, Any],
    endpoint: str,
    model: str,
    timeout_seconds: float,
    transport: Transport,
    clock: Clock,
) -> dict[str, Any]:
    """Execute one raw-token completion and return a fail-closed audit record.

    This reuses the accepted D18 request contract (``request_payload``) and
    response contract (``validate_response``) rather than reimplementing
    them. It never assumes any particular bucket's token lengths: the
    expected/target token counts are derived entirely from ``record``.
    """

    payload = smoke.request_payload(record, model)
    result: dict[str, Any] = {
        "schema_version": REQUEST_RESULT_SCHEMA_VERSION,
        "request_id": record["request_id"],
        "prompt_hash": record["prompt_hash"],
        "bucket": record["bucket"],
        "expected_prompt_tokens": record["prompt_token_count"],
        "target_completion_tokens": record["target_output_tokens"],
        "observed_prompt_tokens": None,
        "observed_completion_tokens": None,
        "observed_total_tokens": None,
        "finish_reason": None,
        "http_status": None,
        "passed": False,
        "failure_reasons": [],
        "submit_monotonic_s": None,
        "terminal_monotonic_s": None,
        "latency_s": None,
    }
    submit_ts = clock()
    result["submit_monotonic_s"] = submit_ts
    try:
        response = transport(endpoint, payload, timeout_seconds)
        result["http_status"] = response.status_code
        if response.status_code != 200:
            result["failure_reasons"].append(
                {
                    "reason": "http_status",
                    "detail": f"expected HTTP 200, got {response.status_code}",
                }
            )
        else:
            try:
                decoded = smoke._decode_response_json(response.body)
            except smoke.SmokeError as error:
                result["failure_reasons"].append(
                    {"reason": "response_json", "detail": str(error)}
                )
            else:
                evidence, failures = smoke.validate_response(decoded, record, model)
                result["observed_prompt_tokens"] = evidence[
                    "server_reported_prompt_tokens"
                ]
                result["observed_completion_tokens"] = evidence[
                    "server_reported_completion_tokens"
                ]
                result["observed_total_tokens"] = evidence[
                    "server_reported_total_tokens"
                ]
                result["finish_reason"] = evidence["finish_reason"]
                result["failure_reasons"].extend(failures)
    except (TransportError, OSError) as error:
        result["failure_reasons"].append(
            {"reason": "http_transport", "detail": str(error)}
        )
    finally:
        terminal_ts = clock()
        result["terminal_monotonic_s"] = terminal_ts
        result["latency_s"] = terminal_ts - submit_ts

    result["passed"] = not result["failure_reasons"]
    return result


# ---------------------------------------------------------------------------
# vLLM /metrics parsing (generic Prometheus text; no metric-name assumptions)
# ---------------------------------------------------------------------------

_PROM_SAMPLE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)"
)
_PROM_LABEL_PAIR_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')

KNOWN_VLLM_METRIC_NAMES: tuple[str, ...] = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc",
    "vllm:kv_cache_usage_perc",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:num_preemptions_total",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:time_to_first_token_seconds_count",
    "vllm:e2e_request_latency_seconds_sum",
    "vllm:e2e_request_latency_seconds_count",
)

# Metric names vary slightly across vLLM releases; the first candidate
# present in a sample is used, in priority order.
VLLM_RUNNING_METRIC_CANDIDATES: tuple[str, ...] = ("vllm:num_requests_running",)
VLLM_WAITING_METRIC_CANDIDATES: tuple[str, ...] = ("vllm:num_requests_waiting",)
VLLM_KV_CACHE_METRIC_CANDIDATES: tuple[str, ...] = (
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
)
VLLM_PREEMPTIONS_METRIC_CANDIDATES: tuple[str, ...] = ("vllm:num_preemptions_total",)


def parse_prometheus_text(text: str) -> list[dict[str, Any]]:
    """Parse Prometheus exposition text into raw ``{name, labels, value}`` samples.

    This is intentionally generic: it does not assume any particular vLLM
    0.28.0 metric exists. Unparsable lines (comments, HELP/TYPE lines, blank
    lines) are skipped rather than raising.
    """

    samples: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _PROM_SAMPLE_RE.match(stripped)
        if not match:
            continue
        name, label_blob, raw_value = match.groups()
        try:
            value = float(raw_value)
        except ValueError:
            continue
        labels = dict(_PROM_LABEL_PAIR_RE.findall(label_blob or ""))
        samples.append({"name": name, "labels": labels, "value": value})
    return samples


def select_known_metrics(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Extract known-useful metric samples, explicitly marking absence."""

    return {
        name: {
            "present": any(sample["name"] == name for sample in samples),
            "samples": [sample for sample in samples if sample["name"] == name],
        }
        for name in KNOWN_VLLM_METRIC_NAMES
    }


def _avg_max(values: Sequence[float]) -> dict[str, Any]:
    """Explicit-absence avg/max helper: never silently substitutes zero."""

    if not values:
        return {"available": False}
    return {
        "available": True,
        "avg": sum(values) / len(values),
        "max": max(values),
        "sample_count": len(values),
    }


def _metric_series_from_ok_samples(
    samples: Sequence[Mapping[str, Any]], metric_names: Sequence[str]
) -> list[float]:
    """One value per sample, from the first present candidate metric name."""

    values: list[float] = []
    for sample in samples:
        known = sample.get("known_metrics") or {}
        for name in metric_names:
            entry = known.get(name)
            if entry and entry.get("present") and entry.get("samples"):
                values.append(sum(metric["value"] for metric in entry["samples"]))
                break
    return values


def summarize_vllm_telemetry_window(
    samples: Sequence[Mapping[str, Any]], t0: float, t1: float
) -> dict[str, Any]:
    """Derived measurement-phase vLLM telemetry summary for one load point.

    Only ``status == "ok"`` samples whose ``timestamp`` falls inside the
    half-open measurement window ``[t0, t1)`` are used, so telemetry from
    another point's window (a different concurrency or point index) cannot
    leak into this summary. A metric that never appeared is reported as
    ``{"available": False}``, never silently coerced to zero.
    """

    window_samples = [
        sample
        for sample in samples
        if sample.get("status") == "ok"
        and sample.get("timestamp") is not None
        and t0 <= sample["timestamp"] < t1
    ]
    running = _metric_series_from_ok_samples(window_samples, VLLM_RUNNING_METRIC_CANDIDATES)
    waiting = _metric_series_from_ok_samples(window_samples, VLLM_WAITING_METRIC_CANDIDATES)
    kv_cache = _metric_series_from_ok_samples(window_samples, VLLM_KV_CACHE_METRIC_CANDIDATES)
    preemptions = _metric_series_from_ok_samples(
        window_samples, VLLM_PREEMPTIONS_METRIC_CANDIDATES
    )

    if preemptions:
        preemption_summary: dict[str, Any] = {
            "available": True,
            "start": preemptions[0],
            "end": preemptions[-1],
            "delta": preemptions[-1] - preemptions[0],
        }
    else:
        preemption_summary = {"available": False}

    return {
        "available": bool(window_samples),
        "measurement_sample_count": len(window_samples),
        "num_requests_running": _avg_max(running),
        "num_requests_waiting": _avg_max(waiting),
        "kv_cache_usage_perc": _avg_max(kv_cache),
        "num_preemptions_total": preemption_summary,
    }


# ---------------------------------------------------------------------------
# Lightweight GPU telemetry (diagnostic; failure is surfaced, not fatal)
# ---------------------------------------------------------------------------

GPU_QUERY_FIELDS: tuple[str, ...] = (
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
    "clocks.sm",
    "clocks_throttle_reasons.active",
)

_GPU_THROTTLE_INACTIVE_TOKENS = {"", "n/a", "not active", "none"}

SubprocessRunner = Callable[..., "subprocess.CompletedProcess[str]"]


def sample_gpu_telemetry(
    run: SubprocessRunner = subprocess.run,
) -> dict[str, Any]:
    """Sample one ``nvidia-smi`` snapshot; never raises."""

    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(GPU_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = run(command, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "error": str(error)}
    if completed.returncode != 0:
        return {
            "available": False,
            "error": (completed.stderr or "").strip() or "nvidia-smi exited non-zero",
        }
    gpus: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(GPU_QUERY_FIELDS):
            return {
                "available": False,
                "error": f"unexpected nvidia-smi output shape: {line!r}",
            }
        gpus.append(dict(zip(GPU_QUERY_FIELDS, values)))
    return {"available": bool(gpus), "gpus": gpus}


def gpu_fingerprint(run: SubprocessRunner = subprocess.run) -> dict[str, Any]:
    """One-shot GPU identity snapshot for the experiment manifest."""

    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,uuid",
        "--format=csv,noheader",
    ]
    try:
        completed = run(command, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "error": str(error)}
    if completed.returncode != 0:
        return {
            "available": False,
            "error": (completed.stderr or "").strip() or "nvidia-smi exited non-zero",
        }
    gpus = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        gpus.append(
            {
                "name": parts[0],
                "driver_version": parts[1],
                "memory_total": parts[2],
                "uuid": parts[3],
            }
        )
    return {"available": bool(gpus), "gpus": gpus}


def _throttle_is_nonzero(raw_value: Any) -> bool:
    """Classify one ``clocks_throttle_reasons.active`` sample as active/inactive."""

    if raw_value is None:
        return False
    text = str(raw_value).strip()
    if text.lower() in _GPU_THROTTLE_INACTIVE_TOKENS:
        return False
    try:
        if text.lower().startswith("0x"):
            return int(text, 16) != 0
        return float(text) != 0
    except ValueError:
        # An unrecognized non-numeric, non-blank value is treated as an
        # active/nonzero signal rather than silently discarded.
        return True


def _gpu_field_series(
    samples: Sequence[Mapping[str, Any]], field_name: str, gpu_index: int = 0
) -> list[float]:
    values: list[float] = []
    for sample in samples:
        gpus = sample.get("gpus") or []
        if len(gpus) <= gpu_index:
            continue
        raw = gpus[gpu_index].get(field_name)
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            continue
    return values


def summarize_gpu_telemetry_window(
    samples: Sequence[Mapping[str, Any]], t0: float, t1: float
) -> dict[str, Any]:
    """Derived measurement-phase GPU telemetry summary for one load point.

    Only samples whose ``timestamp`` falls inside ``[t0, t1)`` are used, so
    telemetry from another point cannot leak into this summary.
    """

    window_samples = [
        sample
        for sample in samples
        if sample.get("timestamp") is not None and t0 <= sample["timestamp"] < t1
    ]
    available_samples = [sample for sample in window_samples if sample.get("available")]
    if not available_samples:
        return {"available": False, "measurement_sample_count": len(window_samples)}

    throttle_raw = [
        sample["gpus"][0].get("clocks_throttle_reasons.active")
        for sample in available_samples
        if sample.get("gpus")
    ]
    nonzero_throttle = [value for value in throttle_raw if _throttle_is_nonzero(value)]

    return {
        "available": True,
        "measurement_sample_count": len(window_samples),
        "utilization.gpu": _avg_max(_gpu_field_series(available_samples, "utilization.gpu")),
        "memory.used": _avg_max(_gpu_field_series(available_samples, "memory.used")),
        "temperature.gpu": _avg_max(_gpu_field_series(available_samples, "temperature.gpu")),
        "power.draw": _avg_max(_gpu_field_series(available_samples, "power.draw")),
        "clocks_throttle_reasons.active": {
            "total_sample_count": len(throttle_raw),
            "nonzero_sample_count": len(nonzero_throttle),
            "distinct_nonzero_values": sorted({str(value) for value in nonzero_throttle}),
        },
    }


class TelemetrySampler:
    """Periodically samples vLLM ``/metrics`` and GPU telemetry (D13/D14).

    Telemetry failures are recorded as diagnostic entries and never raise out
    of the sampling loop; a dead telemetry source must not crash profiling.
    """

    def __init__(
        self,
        *,
        metrics_endpoint: str,
        metrics_transport: Probe,
        gpu_sampler: GpuSampler,
        clock: Clock,
        interval_seconds: float,
        request_timeout_seconds: float,
        on_vllm_sample: Callable[[dict[str, Any]], None],
        on_gpu_sample: Callable[[dict[str, Any]], None],
    ) -> None:
        if interval_seconds <= 0 or not math.isfinite(interval_seconds):
            raise ProfilingError("metrics interval seconds must be finite and positive")
        self._metrics_endpoint = metrics_endpoint
        self._metrics_transport = metrics_transport
        self._gpu_sampler = gpu_sampler
        self._clock = clock
        self._interval_seconds = interval_seconds
        self._request_timeout_seconds = request_timeout_seconds
        self._on_vllm_sample = on_vllm_sample
        self._on_gpu_sample = on_gpu_sample
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def sample_once(self) -> None:
        timestamp = self._clock()
        vllm_entry: dict[str, Any] = {"timestamp": timestamp}
        try:
            response = self._metrics_transport(
                self._metrics_endpoint, self._request_timeout_seconds
            )
        except TransportError as error:
            vllm_entry.update({"status": "transport_error", "detail": str(error)})
        else:
            if response.status_code == 200:
                text = response.body.decode("utf-8", errors="replace")
                samples = parse_prometheus_text(text)
                vllm_entry.update(
                    {
                        "status": "ok",
                        "http_status": response.status_code,
                        "sample_count": len(samples),
                        "known_metrics": select_known_metrics(samples),
                        "raw_samples": samples,
                    }
                )
            else:
                vllm_entry.update(
                    {"status": "http_error", "http_status": response.status_code}
                )
        self._on_vllm_sample(vllm_entry)

        gpu_entry: dict[str, Any] = {"timestamp": timestamp}
        try:
            gpu_entry.update(self._gpu_sampler())
        except Exception as error:  # telemetry must never crash the profiling run
            gpu_entry.update({"available": False, "error": f"sampler raised: {error}"})
        self._on_gpu_sample(gpu_entry)

    def start(self) -> None:
        if self._thread is not None:
            raise ProfilingError("telemetry sampler is already running")
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="telemetry-sampler", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.sample_once()
            except Exception:
                pass
            self._stop_event.wait(self._interval_seconds)

    def stop(self, join_timeout_seconds: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(join_timeout_seconds)
            self._thread = None


@dataclass(frozen=True)
class TelemetryConfig:
    """Wiring needed by :func:`run_load_point` to sample and archive telemetry.

    ``on_vllm_sample``/``on_gpu_sample`` receive every raw sample (for full,
    unsummarized archival, e.g. into an experiment-wide JSONL list); the
    measurement-window derived summaries are computed separately by
    :func:`run_load_point` itself, from its own point-local copies, so they
    can never be contaminated by another point's samples.
    """

    metrics_endpoint: str
    metrics_transport: Probe
    gpu_sampler: GpuSampler
    interval_seconds: float
    request_timeout_seconds: float
    on_vllm_sample: Callable[[dict[str, Any]], None] = lambda entry: None
    on_gpu_sample: Callable[[dict[str, Any]], None] = lambda entry: None


# ---------------------------------------------------------------------------
# Fixed-concurrency closed-loop load point (D7/D9/D10)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimingConfig:
    """Per-point phase timing (D9). Defaults are pilot values, not final.

    ``ramp_admission_interval_seconds`` controls the deterministic initial-
    admission ramp (see :func:`run_load_point`): consecutive initial
    admissions (the first ``concurrency`` requests of a point, before
    settling begins) are spaced this many seconds apart instead of being
    submitted in one immediate burst. ``0.0`` (the pre-ramp behavior) fully
    disables ramping for exact A/B comparison against earlier artifacts.

    ``burst_window_seconds`` and ``near_concurrency_burst_threshold_fraction``
    configure the completion-burst diagnostic (see
    :func:`summarize_completion_clustering`); they are purely diagnostic and
    never affect measurement validity or capacity arithmetic.
    """

    settling_seconds: float = DEFAULT_SETTLING_SECONDS
    measurement_seconds: float = DEFAULT_MEASUREMENT_SECONDS
    drain_timeout_seconds: float = DEFAULT_DRAIN_TIMEOUT_SECONDS
    metrics_interval_seconds: float = DEFAULT_METRICS_INTERVAL_SECONDS
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    ramp_admission_interval_seconds: float = DEFAULT_RAMP_ADMISSION_INTERVAL_SECONDS
    burst_window_seconds: float = DEFAULT_BURST_WINDOW_SECONDS
    near_concurrency_burst_threshold_fraction: float = (
        DEFAULT_NEAR_CONCURRENCY_BURST_THRESHOLD_FRACTION
    )

    def validate(self) -> None:
        for name in (
            "settling_seconds",
            "measurement_seconds",
            "drain_timeout_seconds",
            "metrics_interval_seconds",
            "request_timeout_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ProfilingError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.ramp_admission_interval_seconds)
            or self.ramp_admission_interval_seconds < 0
        ):
            raise ProfilingError(
                "ramp_admission_interval_seconds must be finite and non-negative "
                "(0 disables ramping)"
            )
        if (
            not math.isfinite(self.burst_window_seconds)
            or self.burst_window_seconds < 0
        ):
            raise ProfilingError(
                "burst_window_seconds must be finite and non-negative"
            )
        if not (0 < self.near_concurrency_burst_threshold_fraction <= 1):
            raise ProfilingError(
                "near_concurrency_burst_threshold_fraction must be in (0, 1]"
            )


IdleCheck = Callable[[], tuple[bool, str | None]]


# ---------------------------------------------------------------------------
# Completion-burst / phase-synchronization diagnostic
#
# Real output-heavy profiling exposed a suspicious pattern: several
# concurrency points completed near-exact multiples of the target
# concurrency C within one measurement window (e.g. completed = 2*C), with
# an apparently non-monotonic throughput curve, despite zero request
# failures, zero preemptions, and bounded drains. Since every bucket's
# records share one fixed target_output_tokens, and the current admission
# loop submits its initial C requests as fast as the calling thread can
# call executor.submit() in a tight for-loop (no delay at all -- see
# run_load_point), the initial C requests are dispatched within a narrow
# real-time burst. For a bucket whose records all decode to the same
# length, near-simultaneously started requests can progress through decode
# together and terminate close together, and their closed-loop replacements
# can then re-enter together, preserving a "completion wave" structure
# indefinitely. This is architecture-level evidence supporting the
# phase-synchronization hypothesis independent of the diagnostic below.
#
# CORRECTED ALGORITHM (this diagnostic previously used single-linkage/
# nearest-neighbor-chain clustering: a timestamp joined a cluster iff it
# was within a tolerance of the immediately PREVIOUS timestamp already in
# that cluster). That chains transitively: a perfectly healthy, continuous,
# high-throughput completion stream with small adjacent gaps (e.g. a
# steady ~3-4 req/s output-heavy stream, gaps ~0.25-0.33s, well under a
# 0.5s tolerance) would be merged into ONE arbitrarily large "cluster"
# spanning the entire measurement window, producing a false-positive
# phase_synchronization_suspected verdict on exactly the real regime this
# diagnostic exists to check. This has been replaced with a non-chaining,
# FIXED-WIDTH burst-window diagnostic: membership is always bounded by a
# distance to a fixed window anchor, never by transitive adjacency to a
# chain of neighbors, so a continuous stream can never accumulate into one
# giant burst merely because each individual gap is small.
#
# This diagnostic turns that qualitative pattern into a deterministic,
# reproducible measurement over a point's own request_results.jsonl
# terminal timestamps. It is evidence for HUMAN REVIEW only: it never
# changes run_valid, never changes capacity arithmetic, and never selects
# or rejects a plateau by itself.
# ---------------------------------------------------------------------------


def max_completions_in_fixed_window(
    timestamps: Sequence[float], window_seconds: float
) -> tuple[int, float | None]:
    """Maximum number of timestamps contained in any fixed-width window.

    Returns ``(max_count, window_start)`` where ``window_start`` is the
    start of one window (there may be ties) achieving ``max_count``, or
    ``(0, None)`` for an empty input.

    Boundary semantics are exact and non-chaining: for a window anchored at
    ``window_start``, a timestamp ``t`` is inside the window iff::

        window_start <= t <= window_start + window_seconds

    This is a real two-pointer sliding-window computation over ALL possible
    real-valued window positions, not merely windows anchored at an
    existing timestamp: sliding any window left only ever adds or keeps the
    same timestamps until its left edge reaches the next timestamp, so the
    true maximum is always achieved by some window anchored exactly at one
    of the input timestamps. Each timestamp can therefore only ever be
    counted relative to the window's own fixed anchor -- unlike single-
    linkage clustering, one timestamp can NEVER transitively pull in
    another timestamp more than ``window_seconds`` away from that anchor,
    so a continuous stream of small adjacent gaps cannot accumulate into one
    arbitrarily large burst. This is O(n log n) (for the initial sort; the
    sliding-window scan itself is O(n)).
    """

    if window_seconds < 0 or not math.isfinite(window_seconds):
        raise ProfilingError("burst window seconds must be finite and non-negative")
    ordered = sorted(timestamps)
    if not ordered:
        return 0, None

    best_count = 0
    best_start = ordered[0]
    left = 0
    for right, right_timestamp in enumerate(ordered):
        while right_timestamp - ordered[left] > window_seconds:
            left += 1
        count = right - left + 1
        if count > best_count:
            best_count = count
            best_start = ordered[left]
    return best_count, best_start


def partition_into_non_overlapping_windows(
    timestamps: Sequence[float], window_seconds: float
) -> list[list[float]]:
    """Greedily partition sorted timestamps into fixed-width, non-overlapping
    "burst episodes", for secondary repeated-wave human-review evidence.

    Each episode is anchored at the earliest not-yet-assigned timestamp and
    contains every subsequent timestamp satisfying
    ``timestamp <= anchor + window_seconds`` (same inclusive boundary as
    :func:`max_completions_in_fixed_window`); the next episode then starts
    fresh at the next unassigned timestamp. Episodes never overlap and an
    episode's width is always bounded by its own anchor, so -- unlike
    single-linkage clustering -- this cannot chain a continuous stream into
    one giant episode either. This is intentionally simpler than
    :func:`max_completions_in_fixed_window` (it does not search all window
    positions) and is meant only to give a human a quick read on whether
    *multiple, separated* near-concurrency bursts occurred, matching the
    real observation of repeated ``completed = k*C`` waves.
    """

    if window_seconds < 0 or not math.isfinite(window_seconds):
        raise ProfilingError("burst window seconds must be finite and non-negative")
    ordered = sorted(timestamps)
    windows: list[list[float]] = []
    index = 0
    while index < len(ordered):
        anchor = ordered[index]
        window: list[float] = []
        while index < len(ordered) and ordered[index] <= anchor + window_seconds:
            window.append(ordered[index])
            index += 1
        windows.append(window)
    return windows


def summarize_completion_clustering(
    *,
    results: Sequence[Mapping[str, Any]],
    t0: float,
    t1: float,
    concurrency: int,
    burst_window_seconds: float = DEFAULT_BURST_WINDOW_SECONDS,
    near_concurrency_burst_threshold_fraction: float = (
        DEFAULT_NEAR_CONCURRENCY_BURST_THRESHOLD_FRACTION
    ),
) -> dict[str, Any]:
    """Diagnostic-only completion-burst evidence for one load point.

    Operates on exactly the same population used for the capacity numerator
    (``in_measurement_window and passed``), so ``completion_count`` here is
    always equal to the point summary's ``completed_requests_in_window``.
    Only timestamps satisfying ``t0 <= terminal < t1`` are used, so
    telemetry/results from another point's window can never leak in.

    This is reusable directly against a real ``request_results.jsonl``
    artifact: each line already has the ``terminal_monotonic_s``,
    ``in_measurement_window``, and ``passed`` fields this function reads.
    """

    if burst_window_seconds < 0 or not math.isfinite(burst_window_seconds):
        raise ProfilingError("burst window seconds must be finite and non-negative")
    if not (0 < near_concurrency_burst_threshold_fraction <= 1):
        raise ProfilingError(
            "near-concurrency burst threshold fraction must be in (0, 1]"
        )

    terminals = sorted(
        result["terminal_monotonic_s"]
        for result in results
        if result.get("in_measurement_window")
        and result.get("passed")
        and result.get("terminal_monotonic_s") is not None
        and t0 <= result["terminal_monotonic_s"] < t1
    )
    completion_count = len(terminals)

    if completion_count >= 2:
        gaps = [second - first for first, second in zip(terminals, terminals[1:])]
        inter_completion_gap_seconds: dict[str, Any] = {
            "available": True,
            "count": len(gaps),
            "min": min(gaps),
            "max": max(gaps),
            "avg": sum(gaps) / len(gaps),
        }
    else:
        inter_completion_gap_seconds = {"available": False}

    max_completions, max_burst_window_start = max_completions_in_fixed_window(
        terminals, burst_window_seconds
    )
    max_burst_fraction_of_concurrency = (
        max_completions / concurrency if concurrency > 0 else None
    )
    phase_synchronization_suspected = (
        max_burst_fraction_of_concurrency is not None
        and max_burst_fraction_of_concurrency
        >= near_concurrency_burst_threshold_fraction
    )

    episodes = partition_into_non_overlapping_windows(terminals, burst_window_seconds)
    episode_sizes = [len(episode) for episode in episodes]
    near_concurrency_episode_count = sum(
        1
        for size in episode_sizes
        if concurrency > 0
        and size >= near_concurrency_burst_threshold_fraction * concurrency
    )

    return {
        "diagnostic_purpose": (
            "Evidence for HUMAN REVIEW of completion-burst / phase-"
            "synchronization risk (estimator quality). Does not affect "
            "run_valid and must never be used to automatically accept or "
            "reject a capacity point or plateau."
        ),
        "algorithm": (
            "fixed_width_sliding_window (non-chaining); see "
            "max_completions_in_fixed_window"
        ),
        "population": "valid_measurement_window_completions",
        "burst_window_seconds": burst_window_seconds,
        "near_concurrency_burst_threshold_fraction": (
            near_concurrency_burst_threshold_fraction
        ),
        "completion_count": completion_count,
        "inter_completion_gap_seconds": inter_completion_gap_seconds,
        "max_completions_in_burst_window": max_completions,
        "max_burst_window_start_s": max_burst_window_start,
        "max_burst_fraction_of_concurrency": max_burst_fraction_of_concurrency,
        "repeated_burst_episodes": {
            "note": (
                "Secondary, simpler evidence: greedy non-overlapping "
                "fixed-width episodes (not an exhaustive search like "
                "max_completions_in_burst_window above), for a quick read "
                "on whether MULTIPLE separated near-concurrency bursts "
                "occurred."
            ),
            "episode_count": len(episodes),
            "episode_sizes": episode_sizes,
            "near_concurrency_episode_count": near_concurrency_episode_count,
        },
        "phase_synchronization_suspected": phase_synchronization_suspected,
    }


def summarize_point(
    *,
    bucket: generator.Bucket,
    concurrency: int,
    results: Sequence[Mapping[str, Any]],
    max_observed_concurrency: int,
    t_start: float,
    t0: float,
    t1: float,
    measurement_seconds: float,
    submitted_count: int,
    outstanding_at_t1: int,
    outstanding_after_drain: int,
    drain_duration: float,
    drained: bool,
    extra_invalidation_reasons: Sequence[str] = (),
    ramp_admission_interval_seconds: float = 0.0,
    ramp_start: float | None = None,
    target_concurrency_reached: float | None = None,
) -> dict[str, Any]:
    """Pure summary computation for one load point (D12/D16), no threading.

    Kept separate from :func:`run_load_point` so it is directly unit
    testable with synthetic per-request results. The token-rate invariant is
    derived from ``bucket.total_target_tokens`` rather than a hard-coded
    constant, so it remains correct for any approved bucket.

    ``ramp_admission_interval_seconds``/``ramp_start``/
    ``target_concurrency_reached`` are optional startup-ramp auditability
    fields (see :func:`run_load_point`); omitting them (as every pre-ramp
    caller/test does) yields a zero-duration, disabled-ramp record anchored
    at ``t_start``, which is exactly the historical burst-admission
    behavior.
    """

    if ramp_start is None:
        ramp_start = t_start
    if target_concurrency_reached is None:
        target_concurrency_reached = t_start

    invalidation_reasons = list(extra_invalidation_reasons)

    valid_in_window = [
        result
        for result in results
        if result.get("in_measurement_window") and result.get("passed")
    ]
    completed_requests = len(valid_in_window)
    completed_request_rate = completed_requests / measurement_seconds
    completed_total_tokens = sum(
        (result["observed_prompt_tokens"] or 0)
        + (result["observed_completion_tokens"] or 0)
        for result in valid_in_window
    )
    completed_total_token_rate = completed_total_tokens / measurement_seconds
    expected_total_token_rate = bucket.total_target_tokens * completed_request_rate
    invariant_holds = math.isclose(
        completed_total_token_rate,
        expected_total_token_rate,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )
    if not invariant_holds:
        invalidation_reasons.append("token_rate_invariant_violated")

    if any(not result.get("passed") for result in results):
        invalidation_reasons.append("request_failures_present")
    if not drained:
        invalidation_reasons.append("drain_timeout")
    if max_observed_concurrency < concurrency:
        invalidation_reasons.append("concurrency_not_achieved")

    failure_counts = Counter(
        failure["reason"]
        for result in results
        for failure in result.get("failure_reasons", [])
    )

    return {
        "schema_version": POINT_SUMMARY_SCHEMA_VERSION,
        "bucket": bucket.name,
        "concurrency_target": concurrency,
        "max_observed_concurrency": max_observed_concurrency,
        "settling_start_s": t_start,
        "settling_end_s": t0,
        "measurement_t0_s": t0,
        "measurement_t1_s": t1,
        "measurement_duration_s": measurement_seconds,
        "requests_submitted": submitted_count,
        "completed_requests_in_window": completed_requests,
        "completed_requests_per_second": completed_request_rate,
        "completed_total_tokens_per_second": completed_total_token_rate,
        "outstanding_at_t1": outstanding_at_t1,
        "outstanding_after_drain": outstanding_after_drain,
        "drain_duration_s": drain_duration,
        "drain_outcome": "drained" if drained else "timed_out",
        "failure_counts": dict(sorted(failure_counts.items())),
        "total_failed_requests": sum(
            1 for result in results if not result.get("passed")
        ),
        "run_valid": len(invalidation_reasons) == 0,
        "invalidation_reasons": invalidation_reasons,
        "token_rate_invariant": {
            "expected_total_tokens_per_request": bucket.total_target_tokens,
            "expected_total_token_rate": expected_total_token_rate,
            "observed_total_token_rate": completed_total_token_rate,
            "holds": invariant_holds,
        },
        "adjacent_throughput_gain": None,
        "startup_ramp": {
            "ramp_admission_interval_seconds": ramp_admission_interval_seconds,
            "ramp_enabled": ramp_admission_interval_seconds > 0,
            "ramp_start_s": ramp_start,
            "target_concurrency_reached_s": target_concurrency_reached,
            "ramp_duration_s": target_concurrency_reached - ramp_start,
        },
        "capacity_quantity_warning": (
            "This is monolithic non-P/D V_M evidence using L_in+L_out total "
            "tokens; it is not physical KV-release throughput, not isolated "
            "decoder V_D, and not a final plateau determination."
        ),
    }


def compute_adjacent_gain(
    previous_valid_summary: Mapping[str, Any] | None,
    current_summary: Mapping[str, Any],
) -> float | None:
    """Relative throughput gain vs. the prior VALID point, when applicable (D16)."""

    if previous_valid_summary is None:
        return None
    previous_rate = previous_valid_summary["completed_total_tokens_per_second"]
    if not previous_rate:
        return None
    current_rate = current_summary["completed_total_tokens_per_second"]
    return (current_rate - previous_rate) / previous_rate


def _classify_result_window(
    result: Mapping[str, Any], *, ramp_end: float, t0: float, t1: float
) -> dict[str, Any]:
    """Annotate one raw execution result with its window/phase membership.

    This is deliberately deferred until ``ramp_end``/``t0``/``t1`` are all
    finalized (i.e. until the initial ramp has finished admitting the full
    target concurrency), rather than being computed live inside the
    completion callback. A request admitted early in the ramp can finish
    before the ramp itself finishes admitting later requests, so computing
    ``t0``/``t1`` only once the ramp completes -- as required for the ramp
    semantics -- would otherwise race with an in-flight callback that still
    needed them.
    """

    result = dict(result)
    terminal = result.get("terminal_monotonic_s")
    in_window = terminal is not None and t0 <= terminal < t1
    result["in_measurement_window"] = in_window
    if terminal is None:
        result["phase"] = "unknown"
    elif terminal < ramp_end:
        result["phase"] = "ramp"
    elif terminal < t0:
        result["phase"] = "settling"
    elif terminal < t1:
        result["phase"] = "measurement"
    else:
        result["phase"] = "drain"
    return result


def run_load_point(
    *,
    concurrency: int,
    cycle: PromptCycle,
    endpoint: str,
    model: str,
    bucket: generator.Bucket,
    timing: TimingConfig,
    transport: Transport,
    clock: Clock = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    idle_check: IdleCheck | None = None,
    telemetry_config: TelemetryConfig | None = None,
    run_id: str = "unassigned",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one fixed-concurrency closed-loop load point (D7/D9/D10).

    Returns ``(point_summary, per_request_results)``. If ``telemetry_config``
    is provided, a :class:`TelemetrySampler` is started for the duration of
    this point; its raw samples are forwarded to the caller-supplied sinks
    (for full archival) AND kept in a point-local copy used only to compute
    this point's own measurement-window telemetry summaries, so telemetry
    can never leak across points.

    Fail-closed precondition behavior: if ``idle_check`` reports the
    mandatory precondition as failed (e.g. the server is unreachable), this
    function returns an explicit no-execution invalid point immediately. It
    does not start telemetry, does not create request workers, does not
    admit any request, and therefore cannot generate request-execution HTTP
    traffic against an unavailable/misconfigured server. This was previously
    a gap: a failed precondition was recorded, but the closed-loop executor
    still ran and could flood a down server with tens of thousands of
    immediately-failing HTTP requests before the bounded window elapsed.

    Deterministic initial-admission ramp: the initial ``concurrency``
    requests are admitted one at a time, ``timing.
    ramp_admission_interval_seconds`` apart (``0`` reproduces the historical
    immediate-burst admission exactly). Settling begins only once the full
    target concurrency has actually been admitted (``ramp_end``); ``t0``/
    ``t1`` are derived from that instant exactly as before, so ramp time is
    never counted as settling or measurement time, and the measurement
    denominator (``timing.measurement_seconds``) is unaffected. Once the
    ramp completes, every replacement admission during settling/measurement/
    drain is issued immediately by the completion callback, exactly as
    before -- the ramp applies ONLY to a point's very first admissions.
    """

    if concurrency <= 0:
        raise ProfilingError("concurrency must be positive")
    timing.validate()

    extra_invalidation_reasons: list[str] = []
    precondition_ok = True
    precondition_reason: str | None = None

    if idle_check is not None:
        precondition_ok, precondition_reason = idle_check()
        if not precondition_ok:
            extra_invalidation_reasons.append(
                f"precondition_failed:{precondition_reason}"
            )

    if not precondition_ok:
        # Fail closed: no telemetry, no executor, no admitted requests, no
        # settling/measurement/drain execution of any kind for this point.
        # These timestamps are nominal (no real settling/measurement time
        # ever elapses); the normal execution path below computes its own
        # t_start/t0/t1 at its original point in the control flow, so this
        # early return cannot shift timing for points that actually run.
        skip_t_start = clock()
        skip_t0 = skip_t_start + timing.settling_seconds
        skip_t1 = skip_t0 + timing.measurement_seconds
        summary = summarize_point(
            bucket=bucket,
            concurrency=concurrency,
            results=[],
            max_observed_concurrency=0,
            t_start=skip_t_start,
            t0=skip_t0,
            t1=skip_t1,
            measurement_seconds=timing.measurement_seconds,
            submitted_count=0,
            outstanding_at_t1=0,
            outstanding_after_drain=0,
            drain_duration=0.0,
            drained=True,
            extra_invalidation_reasons=extra_invalidation_reasons,
            ramp_admission_interval_seconds=timing.ramp_admission_interval_seconds,
            ramp_start=skip_t_start,
            target_concurrency_reached=skip_t_start,
        )
        summary["execution_skipped"] = True
        summary["execution_skipped_reason"] = (
            f"precondition_failed:{precondition_reason}"
        )
        summary["vllm_telemetry"] = {
            "available": False,
            "reason": "execution_skipped_precondition_failed",
        }
        summary["gpu_telemetry"] = {
            "available": False,
            "reason": "execution_skipped_precondition_failed",
        }
        summary["completion_clustering"] = summarize_completion_clustering(
            results=[],
            t0=skip_t0,
            t1=skip_t1,
            concurrency=concurrency,
            burst_window_seconds=timing.burst_window_seconds,
            near_concurrency_burst_threshold_fraction=(
                timing.near_concurrency_burst_threshold_fraction
            ),
        )
        return summary, []

    local_vllm_samples: list[dict[str, Any]] = []
    local_gpu_samples: list[dict[str, Any]] = []
    local_telemetry_lock = threading.Lock()
    telemetry: TelemetrySampler | None = None
    if telemetry_config is not None:

        def _local_vllm_sink(entry: dict[str, Any]) -> None:
            with local_telemetry_lock:
                local_vllm_samples.append(entry)
            telemetry_config.on_vllm_sample(entry)

        def _local_gpu_sink(entry: dict[str, Any]) -> None:
            with local_telemetry_lock:
                local_gpu_samples.append(entry)
            telemetry_config.on_gpu_sample(entry)

        telemetry = TelemetrySampler(
            metrics_endpoint=telemetry_config.metrics_endpoint,
            metrics_transport=telemetry_config.metrics_transport,
            gpu_sampler=telemetry_config.gpu_sampler,
            clock=clock,
            interval_seconds=telemetry_config.interval_seconds,
            request_timeout_seconds=telemetry_config.request_timeout_seconds,
            on_vllm_sample=_local_vllm_sink,
            on_gpu_sample=_local_gpu_sink,
        )
        telemetry.start()

    results: list[dict[str, Any]] = []
    results_lock = threading.Lock()
    state_lock = threading.Lock()
    outstanding = 0
    max_observed_concurrency = 0
    submitted_count = 0
    admission_closed = threading.Event()
    all_drained = threading.Event()

    executor = ThreadPoolExecutor(
        max_workers=concurrency, thread_name_prefix=f"profile-c{concurrency}"
    )

    def submit_one() -> None:
        nonlocal outstanding, max_observed_concurrency, submitted_count
        with state_lock:
            if admission_closed.is_set():
                return
            outstanding += 1
            max_observed_concurrency = max(max_observed_concurrency, outstanding)
            submitted_count += 1
        record, sequence = cycle.next()
        try:
            future = executor.submit(
                execute_profiling_request,
                record,
                endpoint,
                model,
                timing.request_timeout_seconds,
                transport,
                clock,
            )
        except RuntimeError:
            # Executor is shutting down; treat as an unadmitted request.
            with state_lock:
                outstanding -= 1
                submitted_count -= 1
            return
        future.add_done_callback(lambda done: _on_done(done, sequence))

    def _on_done(future: Any, sequence: int) -> None:
        nonlocal outstanding
        try:
            result = future.result()
        except Exception as error:  # defensive: execute_profiling_request never raises
            result = {
                "schema_version": REQUEST_RESULT_SCHEMA_VERSION,
                "request_id": None,
                "prompt_hash": None,
                "bucket": bucket.name,
                "expected_prompt_tokens": None,
                "target_completion_tokens": None,
                "observed_prompt_tokens": None,
                "observed_completion_tokens": None,
                "observed_total_tokens": None,
                "finish_reason": None,
                "http_status": None,
                "passed": False,
                "failure_reasons": [
                    {"reason": "unexpected_exception", "detail": str(error)}
                ],
                "submit_monotonic_s": None,
                "terminal_monotonic_s": clock(),
                "latency_s": None,
            }
        # Window/phase membership (in_measurement_window, phase) is
        # deliberately NOT computed here. It is computed once, for every
        # result, only after the initial ramp has finished admitting the
        # full target concurrency and t0/t1 are therefore finalized (see
        # _classify_result_window below) -- an early ramp admission can
        # complete before later ramp admissions are even issued.
        result = dict(result)
        result["run_id"] = run_id
        result["concurrency"] = concurrency
        result["sequence"] = sequence

        with results_lock:
            results.append(result)

        should_signal_drained = False
        with state_lock:
            outstanding -= 1
            if admission_closed.is_set() and outstanding == 0:
                should_signal_drained = True
        if should_signal_drained:
            all_drained.set()
        if not admission_closed.is_set():
            submit_one()

    # Deterministic initial-admission ramp (D7 requirement 7/8/9/13/14): the
    # first `concurrency` admissions are spaced ramp_admission_interval
    # seconds apart instead of being submitted in one immediate burst.
    # `submit_one()` itself is completely unchanged; only the pacing of
    # these specific `concurrency` calls differs. Every later replacement
    # admission (issued from _on_done during settling/measurement/drain)
    # remains immediate, exactly as before.
    ramp_admission_interval = timing.ramp_admission_interval_seconds
    ramp_start = clock()
    for admission_index in range(concurrency):
        if admission_index > 0 and ramp_admission_interval > 0:
            sleep(ramp_admission_interval)
        submit_one()
    ramp_end = clock()  # full target concurrency has now been admitted

    # Settling begins only once the full target concurrency has actually
    # been reached, so t0/t1 must be derived from ramp_end, not ramp_start.
    # With ramp_admission_interval_seconds == 0 (the pre-ramp default),
    # ramp_start == ramp_end and this is exactly the historical behavior.
    t_start = ramp_end
    t0 = t_start + timing.settling_seconds
    t1 = t0 + timing.measurement_seconds

    remaining_to_t1 = t1 - clock()
    if remaining_to_t1 > 0:
        sleep(remaining_to_t1)

    with state_lock:
        admission_closed.set()
        outstanding_at_t1 = outstanding
        if outstanding_at_t1 == 0:
            all_drained.set()

    drain_start = clock()
    drained = all_drained.wait(timing.drain_timeout_seconds)
    drain_duration = clock() - drain_start

    with state_lock:
        outstanding_after_drain = outstanding

    executor.shutdown(wait=True)

    if telemetry is not None:
        telemetry.stop()

    if idle_check is not None:
        ok, reason = idle_check()
        if not ok:
            extra_invalidation_reasons.append(f"server_unreachable_after_point:{reason}")

    with results_lock:
        results_snapshot = [
            _classify_result_window(result, ramp_end=ramp_end, t0=t0, t1=t1)
            for result in results
        ]

    summary = summarize_point(
        bucket=bucket,
        concurrency=concurrency,
        results=results_snapshot,
        max_observed_concurrency=max_observed_concurrency,
        t_start=t_start,
        t0=t0,
        t1=t1,
        measurement_seconds=timing.measurement_seconds,
        submitted_count=submitted_count,
        outstanding_at_t1=outstanding_at_t1,
        outstanding_after_drain=outstanding_after_drain,
        drain_duration=drain_duration,
        drained=drained,
        extra_invalidation_reasons=extra_invalidation_reasons,
        ramp_admission_interval_seconds=ramp_admission_interval,
        ramp_start=ramp_start,
        target_concurrency_reached=ramp_end,
    )
    summary["execution_skipped"] = False
    summary["completion_clustering"] = summarize_completion_clustering(
        results=results_snapshot,
        t0=t0,
        t1=t1,
        concurrency=concurrency,
        burst_window_seconds=timing.burst_window_seconds,
        near_concurrency_burst_threshold_fraction=(
            timing.near_concurrency_burst_threshold_fraction
        ),
    )

    if telemetry_config is not None:
        with local_telemetry_lock:
            vllm_samples_snapshot = list(local_vllm_samples)
            gpu_samples_snapshot = list(local_gpu_samples)
        summary["vllm_telemetry"] = summarize_vllm_telemetry_window(
            vllm_samples_snapshot, t0, t1
        )
        summary["gpu_telemetry"] = summarize_gpu_telemetry_window(
            gpu_samples_snapshot, t0, t1
        )
    else:
        summary["vllm_telemetry"] = {
            "available": False,
            "reason": "telemetry_not_collected",
        }
        summary["gpu_telemetry"] = {
            "available": False,
            "reason": "telemetry_not_collected",
        }

    return summary, results_snapshot


# ---------------------------------------------------------------------------
# Experiment orchestration
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """All inputs that affect one single-bucket profiling experiment."""

    profiling_jsonl: Path
    output_dir: Path
    base_url: str
    model: str
    tokenizer_revision: str
    bucket: str
    vllm_version: str = "0.28.0"
    dtype: str = "float16"
    tensor_parallel_size: int = 1
    max_model_len: int = 1024
    generation_config: str = "vllm"
    prefix_caching: bool = False
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    concurrency_ladder: tuple[int, ...] = DEFAULT_CONCURRENCY_LADDER
    timing: TimingConfig = field(default_factory=TimingConfig)
    run_id: str = ""
    transport: Transport = smoke.post_completion
    probe_transport: Probe = http_get
    metrics_transport: Probe = http_get
    gpu_sampler: GpuSampler = sample_gpu_telemetry
    fingerprint_runner: SubprocessRunner = subprocess.run
    clock: Clock = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    collect_telemetry: bool = True

    def validate(self) -> None:
        if not self.model.strip():
            raise ProfilingError("model must not be empty")
        if not self.tokenizer_revision.strip():
            raise ProfilingError("tokenizer revision must not be empty")
        resolve_bucket(self.bucket)
        if not (0.0 < self.gpu_memory_utilization <= 1.0):
            raise ProfilingError("gpu_memory_utilization must be in (0.0, 1.0]")
        if not self.concurrency_ladder:
            raise ProfilingError("concurrency ladder must not be empty")
        previous = 0
        for value in self.concurrency_ladder:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ProfilingError("concurrency ladder values must be positive integers")
            if value <= previous:
                raise ProfilingError(
                    "concurrency ladder must be strictly increasing"
                )
            previous = value
        self.timing.validate()


def probe_server_identity(config: ExperimentConfig) -> dict[str, Any]:
    """Best-effort server identity evidence for the manifest (D4)."""

    identity: dict[str, Any] = {
        "served_model_ids": None,
        "models_probe_error": None,
        "vllm_version": None,
        "version_probe_error": None,
        "gpu_fingerprint": gpu_fingerprint(config.fingerprint_runner),
    }
    try:
        response = config.probe_transport(
            models_endpoint(config.base_url), config.timing.request_timeout_seconds
        )
        if response.status_code == 200:
            decoded = smoke._decode_response_json(response.body)
            if isinstance(decoded, dict):
                identity["served_model_ids"] = sorted(
                    entry.get("id")
                    for entry in decoded.get("data", [])
                    if isinstance(entry, dict) and isinstance(entry.get("id"), str)
                )
        else:
            identity["models_probe_error"] = f"http_{response.status_code}"
    except (TransportError, smoke.SmokeError) as error:
        identity["models_probe_error"] = str(error)

    try:
        response = config.probe_transport(
            version_endpoint(config.base_url), config.timing.request_timeout_seconds
        )
        if response.status_code == 200:
            decoded = smoke._decode_response_json(response.body)
            if isinstance(decoded, dict):
                identity["vllm_version"] = decoded.get("version")
        else:
            identity["version_probe_error"] = f"http_{response.status_code}"
    except (TransportError, smoke.SmokeError) as error:
        identity["version_probe_error"] = str(error)

    return identity


def build_idle_check(config: ExperimentConfig) -> IdleCheck:
    def idle_check() -> tuple[bool, str | None]:
        try:
            response = config.probe_transport(
                models_endpoint(config.base_url),
                config.timing.request_timeout_seconds,
            )
        except TransportError as error:
            return False, f"models_probe_transport_error:{error}"
        if response.status_code != 200:
            return False, f"models_probe_http_{response.status_code}"
        try:
            decoded = smoke._decode_response_json(response.body)
        except smoke.SmokeError as error:
            return False, f"models_probe_json_error:{error}"
        model_ids = set()
        if isinstance(decoded, dict):
            model_ids = {
                entry.get("id")
                for entry in decoded.get("data", [])
                if isinstance(entry, dict)
            }
        if config.model not in model_ids:
            return False, f"model_identity_mismatch:{sorted(filter(None, model_ids))}"
        return True, None

    return idle_check


def default_run_id(bucket_name: str) -> str:
    return f"{bucket_name}-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_manifest(
    config: ExperimentConfig,
    bucket: generator.Bucket,
    dataset_sha256: str,
    bucket_record_count: int,
    server_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "phase": f"{bucket.name}_bucket_fixed_concurrency_profiling",
        "bucket_validation_status": BUCKET_VALIDATION_STATUS.get(
            bucket.name, "unknown"
        ),
        "non_goals": [
            f"this artifact covers ONLY the {bucket.name!r} bucket; other "
            "approved buckets require separate runs and separate artifacts",
            "not a held-out mixture validation",
            "not a mixed-workload composition claim",
            "not P/D disaggregated profiling",
            "not isolated decoder V_D",
            "not physical KV-release throughput",
            "not a final plateau determination (human review required)",
            "not autoscaling or production-capacity guidance",
        ],
        "model": {
            "model_id": config.model,
            "immutable_revision": config.tokenizer_revision,
            "tokenizer_revision": config.tokenizer_revision,
        },
        "vllm": {
            "expected_version": config.vllm_version,
            "server_reported_version": server_identity.get("vllm_version"),
            "operator_declared_runtime_configuration": {
                "note": (
                    "Declared by the operator invoking this profiler; not "
                    "independently verified against the running server "
                    "process beyond /v1/models identity and a best-effort "
                    "/version probe."
                ),
                "dtype": config.dtype,
                "tensor_parallel_size": config.tensor_parallel_size,
                "max_model_len": config.max_model_len,
                "generation_config": config.generation_config,
                "prefix_caching": config.prefix_caching,
                "gpu_memory_utilization": config.gpu_memory_utilization,
            },
        },
        "server_identity": server_identity,
        "dataset": {
            "profiling_jsonl": str(config.profiling_jsonl),
            "dataset_sha256": dataset_sha256,
            "bucket_record_count": bucket_record_count,
            "bucket_definition": {
                "name": bucket.name,
                "input_tokens": bucket.input_tokens,
                "target_output_tokens": bucket.target_output_tokens,
                "total_target_tokens": bucket.total_target_tokens,
            },
        },
        "concurrency_ladder": list(config.concurrency_ladder),
        "timing": {
            "settling_seconds": config.timing.settling_seconds,
            "measurement_seconds": config.timing.measurement_seconds,
            "drain_timeout_seconds": config.timing.drain_timeout_seconds,
            "metrics_interval_seconds": config.timing.metrics_interval_seconds,
            "request_timeout_seconds": config.timing.request_timeout_seconds,
        },
        "startup_ramp": {
            "policy": "fixed_interval_between_initial_admissions",
            "ramp_admission_interval_seconds": (
                config.timing.ramp_admission_interval_seconds
            ),
            "enabled": config.timing.ramp_admission_interval_seconds > 0,
            "note": (
                "Deterministic, non-adaptive stagger applied ONLY to the "
                "initial `concurrency` admissions of each load point, "
                "before settling begins; every later replacement admission "
                "during settling/measurement/drain remains immediate. Ramp "
                "duration for one point with target concurrency C is "
                "(C - 1) * ramp_admission_interval_seconds. This never "
                "changes target concurrency, [T0,T1) semantics, the "
                "measurement denominator, or capacity arithmetic. Actual "
                "per-point ramp_start_s/target_concurrency_reached_s are "
                "recorded in point_summaries.jsonl's startup_ramp block. "
                "0 reproduces the original immediate-burst admission "
                "exactly, for A/B comparison against pre-ramp artifacts."
            ),
        },
        "completion_clustering_diagnostic": {
            "algorithm": "fixed_width_sliding_window (non-chaining)",
            "burst_window_seconds": config.timing.burst_window_seconds,
            "near_concurrency_burst_threshold_fraction": (
                config.timing.near_concurrency_burst_threshold_fraction
            ),
            "note": (
                "Diagnostic-only phase-synchronization/completion-burst "
                "evidence, computed per point from request_results.jsonl "
                "terminal timestamps and recorded in each point summary's "
                "completion_clustering block. Never affects run_valid and "
                "never selects or rejects a plateau; see README."
            ),
        },
        "base_url": config.base_url,
        "runtime_launch_assumptions": {
            "note": (
                "Operator-declared runtime launch configuration; verified "
                "only via /v1/models model identity and a best-effort "
                "/version probe, not independently re-derived. A different "
                "serving configuration defines a different capacity "
                "artifact."
            ),
            "recommended_launch_flags": [
                f"--max-model-len {config.max_model_len}",
                f"--generation-config {config.generation_config}",
                (
                    "--enable-prefix-caching"
                    if config.prefix_caching
                    else "--no-enable-prefix-caching"
                ),
                f"--dtype {config.dtype}",
                f"--tensor-parallel-size {config.tensor_parallel_size}",
                f"--gpu-memory-utilization {config.gpu_memory_utilization}",
            ],
        },
        "host": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "node": platform.node(),
        },
        "generated_at_wall_clock_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Run the full concurrency ladder for one selected bucket."""

    config.validate()
    bucket = resolve_bucket(config.bucket)
    records, dataset_sha256 = smoke.load_profiling_records(
        config.profiling_jsonl, config.model, config.tokenizer_revision
    )
    bucket_records = select_bucket_records(records, config.bucket)
    cycle = PromptCycle(bucket_records)
    endpoint = smoke.completions_endpoint(config.base_url)
    idle_check = build_idle_check(config)
    server_identity = probe_server_identity(config)

    vllm_metrics: list[dict[str, Any]] = []
    gpu_metrics: list[dict[str, Any]] = []
    telemetry_lock = threading.Lock()

    point_summaries: list[dict[str, Any]] = []
    request_results: list[dict[str, Any]] = []
    previous_valid_summary: dict[str, Any] | None = None

    for point_index, concurrency in enumerate(config.concurrency_ladder):
        telemetry_config: TelemetryConfig | None = None
        if config.collect_telemetry:

            def make_sink(sink_list: list[dict[str, Any]]):
                def _sink(entry: dict[str, Any]) -> None:
                    tagged = dict(entry)
                    tagged["run_id"] = config.run_id
                    tagged["bucket"] = bucket.name
                    tagged["concurrency"] = concurrency
                    tagged["point_index"] = point_index
                    with telemetry_lock:
                        sink_list.append(tagged)

                return _sink

            telemetry_config = TelemetryConfig(
                metrics_endpoint=metrics_endpoint(config.base_url),
                metrics_transport=config.metrics_transport,
                gpu_sampler=config.gpu_sampler,
                interval_seconds=config.timing.metrics_interval_seconds,
                request_timeout_seconds=config.timing.request_timeout_seconds,
                on_vllm_sample=make_sink(vllm_metrics),
                on_gpu_sample=make_sink(gpu_metrics),
            )

        summary, results = run_load_point(
            concurrency=concurrency,
            cycle=cycle,
            endpoint=endpoint,
            model=config.model,
            bucket=bucket,
            timing=config.timing,
            transport=config.transport,
            clock=config.clock,
            sleep=config.sleep,
            idle_check=idle_check,
            telemetry_config=telemetry_config,
            run_id=config.run_id,
        )
        summary["adjacent_throughput_gain"] = compute_adjacent_gain(
            previous_valid_summary, summary
        )
        if summary["run_valid"]:
            previous_valid_summary = summary
        point_summaries.append(summary)
        request_results.extend(results)

    manifest = build_manifest(
        config, bucket, dataset_sha256, len(bucket_records), server_identity
    )

    review_table = [
        {
            "bucket": summary["bucket"],
            "concurrency": summary["concurrency_target"],
            "completed_requests_per_second": summary["completed_requests_per_second"],
            "completed_total_tokens_per_second": summary[
                "completed_total_tokens_per_second"
            ],
            "adjacent_throughput_gain": summary["adjacent_throughput_gain"],
            "run_valid": summary["run_valid"],
            "invalidation_reasons": summary["invalidation_reasons"],
            "num_preemptions_delta": (
                summary["vllm_telemetry"]["num_preemptions_total"]["delta"]
                if summary["vllm_telemetry"].get("num_preemptions_total", {}).get(
                    "available"
                )
                else None
            ),
        }
        for summary in point_summaries
    ]

    experiment_summary = {
        "schema_version": EXPERIMENT_SUMMARY_SCHEMA_VERSION,
        "run_id": config.run_id,
        "bucket": bucket.name,
        "bucket_validation_status": BUCKET_VALIDATION_STATUS.get(
            bucket.name, "unknown"
        ),
        "manifest_artifact": MANIFEST_FILENAME,
        "point_summaries_artifact": POINT_SUMMARIES_FILENAME,
        "request_results_artifact": REQUEST_RESULTS_FILENAME,
        "review_table": review_table,
        "any_point_invalid": any(not summary["run_valid"] for summary in point_summaries),
        "capacity_quantity_warning": (
            "v_hat_M is monolithic non-P/D total-token (L_in+L_out) "
            "throughput. It is NOT physical KV-release throughput, NOT "
            "isolated decoder V_D, and NOT SLO-safe operating capacity."
        ),
        "plateau_acceptance_warning": (
            "This harness does not and must not select a final plateau "
            "automatically. Plateau acceptance is a HUMAN REVIEW decision."
        ),
    }

    return {
        "manifest": manifest,
        "point_summaries": point_summaries,
        "request_results": request_results,
        "vllm_metrics": vllm_metrics,
        "gpu_metrics": gpu_metrics,
        "summary": experiment_summary,
    }


# ---------------------------------------------------------------------------
# Artifact writing (atomic; mirrors run_request_smoke's publish-once pattern)
# ---------------------------------------------------------------------------


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for record in records
    )


def write_artifacts(output_dir: Path, bundle: Mapping[str, Any]) -> None:
    """Publish the complete artifact tree only after every file is durable."""

    if output_dir.exists():
        raise ProfilingError(f"output directory already exists: {output_dir}")
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=parent)
    )
    try:
        (temporary_dir / "telemetry").mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, bytes] = {
            MANIFEST_FILENAME: (
                json.dumps(
                    bundle["manifest"], ensure_ascii=False, indent=2, sort_keys=True
                )
                + "\n"
            ).encode("utf-8"),
            REQUEST_RESULTS_FILENAME: _jsonl_bytes(bundle["request_results"]),
            POINT_SUMMARIES_FILENAME: _jsonl_bytes(bundle["point_summaries"]),
            SUMMARY_FILENAME: (
                json.dumps(
                    bundle["summary"], ensure_ascii=False, indent=2, sort_keys=True
                )
                + "\n"
            ).encode("utf-8"),
            VLLM_METRICS_FILENAME: _jsonl_bytes(bundle["vllm_metrics"]),
            GPU_METRICS_FILENAME: _jsonl_bytes(bundle["gpu_metrics"]),
        }
        for filename, payload in artifacts.items():
            path = temporary_dir / filename
            with path.open("xb") as output_file:
                output_file.write(payload)
                output_file.flush()
                os.fsync(output_file.fileno())
        os.replace(temporary_dir, output_dir)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_concurrency_ladder(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "concurrency ladder must be a comma-separated list of integers"
        ) from error


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiling-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--bucket",
        required=True,
        choices=sorted(BUCKETS_BY_NAME),
        help="exactly one approved dataset bucket to profile",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="OpenAI-compatible server root",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--tokenizer-revision",
        required=True,
        help="expected immutable revision recorded in the source dataset",
    )
    parser.add_argument("--vllm-version", default="0.28.0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--generation-config", default="vllm")
    parser.add_argument(
        "--prefix-caching",
        action="store_true",
        help=(
            "record that prefix caching was enabled on the server; the "
            "validated D5 configuration keeps this disabled (the default)"
        ),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help="operator-declared vLLM --gpu-memory-utilization used at launch",
    )
    parser.add_argument(
        "--concurrency",
        type=_parse_concurrency_ladder,
        default=DEFAULT_CONCURRENCY_LADDER,
        help="comma-separated, strictly increasing concurrency ladder",
    )
    parser.add_argument(
        "--settling-seconds", type=float, default=DEFAULT_SETTLING_SECONDS
    )
    parser.add_argument(
        "--measurement-seconds", type=float, default=DEFAULT_MEASUREMENT_SECONDS
    )
    parser.add_argument(
        "--drain-timeout-seconds", type=float, default=DEFAULT_DRAIN_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--metrics-interval-seconds",
        type=float,
        default=DEFAULT_METRICS_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--ramp-admission-interval-seconds",
        type=float,
        default=DEFAULT_RAMP_ADMISSION_INTERVAL_SECONDS,
        help=(
            "deterministic spacing between each of the initial `concurrency` "
            "admissions of a load point (before settling begins); 0 "
            "reproduces the original immediate-burst admission for exact "
            "A/B comparison"
        ),
    )
    parser.add_argument(
        "--burst-window-seconds",
        type=float,
        default=DEFAULT_BURST_WINDOW_SECONDS,
        help=(
            "fixed-width window used by the non-chaining completion-burst "
            "diagnostic: a timestamp t is inside a window anchored at "
            "window_start iff window_start <= t <= window_start + "
            "burst_window_seconds (diagnostic only)"
        ),
    )
    parser.add_argument(
        "--near-concurrency-burst-threshold-fraction",
        type=float,
        default=DEFAULT_NEAR_CONCURRENCY_BURST_THRESHOLD_FRACTION,
        help=(
            "fraction of target concurrency a burst window must reach to "
            "count as a near-concurrency (suspected synchronized) burst "
            "(diagnostic only)"
        ),
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--no-telemetry",
        action="store_true",
        help="disable vLLM /metrics and GPU telemetry collection (diagnostic only)",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        profiling_jsonl=args.profiling_jsonl,
        output_dir=args.output_dir,
        base_url=args.base_url,
        model=args.model,
        tokenizer_revision=args.tokenizer_revision,
        bucket=args.bucket,
        vllm_version=args.vllm_version,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        generation_config=args.generation_config,
        prefix_caching=args.prefix_caching,
        gpu_memory_utilization=args.gpu_memory_utilization,
        concurrency_ladder=args.concurrency,
        timing=TimingConfig(
            settling_seconds=args.settling_seconds,
            measurement_seconds=args.measurement_seconds,
            drain_timeout_seconds=args.drain_timeout_seconds,
            metrics_interval_seconds=args.metrics_interval_seconds,
            request_timeout_seconds=args.request_timeout_seconds,
            ramp_admission_interval_seconds=args.ramp_admission_interval_seconds,
            burst_window_seconds=args.burst_window_seconds,
            near_concurrency_burst_threshold_fraction=(
                args.near_concurrency_burst_threshold_fraction
            ),
        ),
        run_id=args.run_id or default_run_id(args.bucket),
        collect_telemetry=not args.no_telemetry,
    )


def run_cli(config: ExperimentConfig) -> int:
    try:
        bundle = run_experiment(config)
        write_artifacts(config.output_dir, bundle)
    except (ProfilingError, smoke.SmokeError, OSError) as error:
        print(f"bucket profiling failed: {error}", file=sys.stderr)
        return 1

    print(f"bucket: {config.bucket}")
    print(f"run_id: {config.run_id}")
    print(f"artifacts written to: {config.output_dir}")
    print(f"validation status: {bundle['summary']['bucket_validation_status']}")
    for row in bundle["summary"]["review_table"]:
        print(
            f"  C={row['concurrency']:>4} "
            f"req/s={row['completed_requests_per_second']:.4f} "
            f"tok/s={row['completed_total_tokens_per_second']:.2f} "
            f"gain={row['adjacent_throughput_gain']} "
            f"preemptions_delta={row['num_preemptions_delta']} "
            f"valid={row['run_valid']} "
            f"reasons={row['invalidation_reasons']}"
        )
    if bundle["summary"]["any_point_invalid"]:
        print(
            "WARNING: at least one load point is invalid; see "
            "invalidation_reasons before drawing any conclusion.",
            file=sys.stderr,
        )
    print(
        "NOTE: plateau acceptance is a HUMAN REVIEW decision; this tool "
        "does not select a final V_M."
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    config = config_from_args(args)
    return run_cli(config)


if __name__ == "__main__":
    raise SystemExit(main())

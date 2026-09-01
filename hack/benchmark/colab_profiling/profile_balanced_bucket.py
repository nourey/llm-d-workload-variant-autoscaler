#!/usr/bin/env python3
"""Bounded balanced-bucket fixed-concurrency V_M profiling harness for #1546.

This module implements ONLY the balanced (256-input / 256-output) bucket
fixed-concurrency profiling phase authorized by
``.claude/work/1546/DECISIONS.md`` and
``.claude/work/1546/IMPLEMENTATION_BRIEF.md``.

It intentionally reuses the already-accepted dataset and request-contract
layers instead of duplicating them:

* ``generate_dataset.py`` defines the approved bucket geometry
  (``DEFAULT_BUCKETS``) and the stable ``prompt_hash`` used by the dataset.
* ``run_request_smoke.py`` defines the exact raw-token ``/v1/completions``
  request contract (``request_payload``), the response-contract validator
  (``validate_response``), and the profiling-JSONL loader/validator
  (``load_profiling_records``).

This module does NOT:

* profile the input-heavy or output-heavy buckets;
* run held-out mixtures;
* claim isolated decoder V_D or physical KV-release throughput;
* make an automatic final plateau decision;
* implement autoscaling.

The resulting artifacts are raw evidence for a HUMAN REVIEW decision after a
real Colab run.
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


MANIFEST_SCHEMA_VERSION = "llm-d-colab-balanced-profiling-manifest-v1"
REQUEST_RESULT_SCHEMA_VERSION = "llm-d-colab-balanced-profiling-request-result-v1"
POINT_SUMMARY_SCHEMA_VERSION = "llm-d-colab-balanced-profiling-point-summary-v1"
EXPERIMENT_SUMMARY_SCHEMA_VERSION = "llm-d-colab-balanced-profiling-summary-v1"

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
MAX_RESPONSE_BYTES = smoke.MAX_RESPONSE_BYTES

BALANCED_BUCKET = next(
    bucket for bucket in generator.DEFAULT_BUCKETS if bucket.name == "balanced"
)

TransportError = smoke.TransportError
HttpResponse = smoke.HttpResponse
Transport = smoke.Transport
Probe = Callable[[str, float], HttpResponse]
GpuSampler = Callable[[], dict[str, Any]]
Clock = Callable[[], float]


class ProfilingError(ValueError):
    """Raised when profiling setup or configuration is invalid."""


# ---------------------------------------------------------------------------
# Dataset selection (reuses generate_dataset/run_request_smoke contracts)
# ---------------------------------------------------------------------------


def select_balanced_records(
    records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Return only ``balanced``-bucket profiling records, in file order."""

    balanced = [record for record in records if record.get("bucket") == "balanced"]
    if not balanced:
        raise ProfilingError("no balanced-bucket profiling records were found")
    return balanced


class PromptCycle:
    """A thread-safe, deterministic cyclic iterator over balanced prompts.

    Records are sorted by ``request_id`` once, at construction, so the cycle
    order does not depend on JSONL line order or dict iteration order. Each
    call to :meth:`next` hands out the next record in round-robin order and
    is reproducible from the recorded record set alone (D8).
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
    them.
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


# ---------------------------------------------------------------------------
# Fixed-concurrency closed-loop load point (D7/D9/D10)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimingConfig:
    """Per-point phase timing (D9). Defaults are pilot values, not final."""

    settling_seconds: float = DEFAULT_SETTLING_SECONDS
    measurement_seconds: float = DEFAULT_MEASUREMENT_SECONDS
    drain_timeout_seconds: float = DEFAULT_DRAIN_TIMEOUT_SECONDS
    metrics_interval_seconds: float = DEFAULT_METRICS_INTERVAL_SECONDS
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS

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


IdleCheck = Callable[[], tuple[bool, str | None]]


def summarize_point(
    *,
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
) -> dict[str, Any]:
    """Pure summary computation for one load point (D12/D16), no threading.

    Kept separate from :func:`run_load_point` so it is directly unit
    testable with synthetic per-request results.
    """

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
    expected_total_token_rate = (
        BALANCED_BUCKET.total_target_tokens * completed_request_rate
    )
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
            "expected_total_tokens_per_request": BALANCED_BUCKET.total_target_tokens,
            "expected_total_token_rate": expected_total_token_rate,
            "observed_total_token_rate": completed_total_token_rate,
            "holds": invariant_holds,
        },
        "adjacent_throughput_gain": None,
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


def run_load_point(
    *,
    concurrency: int,
    cycle: PromptCycle,
    endpoint: str,
    model: str,
    timing: TimingConfig,
    transport: Transport,
    clock: Clock = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    idle_check: IdleCheck | None = None,
    telemetry: TelemetrySampler | None = None,
    run_id: str = "unassigned",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one fixed-concurrency closed-loop load point (D7/D9/D10).

    Returns ``(point_summary, per_request_results)``.
    """

    if concurrency <= 0:
        raise ProfilingError("concurrency must be positive")
    timing.validate()

    extra_invalidation_reasons: list[str] = []

    if idle_check is not None:
        ok, reason = idle_check()
        if not ok:
            extra_invalidation_reasons.append(f"precondition_failed:{reason}")

    if telemetry is not None:
        telemetry.start()

    results: list[dict[str, Any]] = []
    results_lock = threading.Lock()
    state_lock = threading.Lock()
    outstanding = 0
    max_observed_concurrency = 0
    submitted_count = 0
    admission_closed = threading.Event()
    all_drained = threading.Event()

    t_start = clock()
    t0 = t_start + timing.settling_seconds
    t1 = t0 + timing.measurement_seconds

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
                "bucket": BALANCED_BUCKET.name,
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
        terminal = result.get("terminal_monotonic_s")
        in_window = terminal is not None and t0 <= terminal < t1
        result = dict(result)
        result["run_id"] = run_id
        result["concurrency"] = concurrency
        result["sequence"] = sequence
        result["in_measurement_window"] = in_window
        if terminal is None:
            result["phase"] = "unknown"
        elif terminal < t0:
            result["phase"] = "settling"
        elif terminal < t1:
            result["phase"] = "measurement"
        else:
            result["phase"] = "drain"

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

    for _ in range(concurrency):
        submit_one()

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
        results_snapshot = list(results)

    summary = summarize_point(
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
    )
    return summary, results_snapshot


# ---------------------------------------------------------------------------
# Experiment orchestration
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """All inputs that affect one balanced-bucket profiling experiment."""

    profiling_jsonl: Path
    output_dir: Path
    base_url: str
    model: str
    tokenizer_revision: str
    vllm_version: str = "0.28.0"
    dtype: str = "float16"
    tensor_parallel_size: int = 1
    max_model_len: int = 1024
    generation_config: str = "vllm"
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


def default_run_id(clock: Clock = time.monotonic) -> str:
    return "balanced-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_manifest(
    config: ExperimentConfig,
    dataset_sha256: str,
    balanced_record_count: int,
    server_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "phase": "balanced_bucket_fixed_concurrency_profiling",
        "non_goals": [
            "not input-heavy or output-heavy bucket profiling",
            "not a held-out mixture validation",
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
            "dtype": config.dtype,
            "tensor_parallel_size": config.tensor_parallel_size,
            "max_model_len": config.max_model_len,
            "generation_config": config.generation_config,
        },
        "server_identity": server_identity,
        "dataset": {
            "profiling_jsonl": str(config.profiling_jsonl),
            "dataset_sha256": dataset_sha256,
            "balanced_record_count": balanced_record_count,
            "bucket_definition": {
                "name": BALANCED_BUCKET.name,
                "input_tokens": BALANCED_BUCKET.input_tokens,
                "target_output_tokens": BALANCED_BUCKET.target_output_tokens,
                "total_target_tokens": BALANCED_BUCKET.total_target_tokens,
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
        "base_url": config.base_url,
        "runtime_launch_assumptions": {
            "note": (
                "Declared by the operator invoking this profiler; verified "
                "only via /v1/models model identity and a best-effort "
                "/version probe, not independently re-derived."
            ),
            "recommended_launch_flags": [
                f"--max-model-len {config.max_model_len}",
                f"--generation-config {config.generation_config}",
                "--no-enable-prefix-caching",
                f"--dtype {config.dtype}",
                f"--tensor-parallel-size {config.tensor_parallel_size}",
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
    """Run the full balanced-bucket concurrency ladder and return raw evidence."""

    config.validate()
    records, dataset_sha256 = smoke.load_profiling_records(
        config.profiling_jsonl, config.model, config.tokenizer_revision
    )
    balanced_records = select_balanced_records(records)
    cycle = PromptCycle(balanced_records)
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
        telemetry: TelemetrySampler | None = None
        if config.collect_telemetry:

            def make_sink(sink_list: list[dict[str, Any]]):
                def _sink(entry: dict[str, Any]) -> None:
                    tagged = dict(entry)
                    tagged["run_id"] = config.run_id
                    tagged["concurrency"] = concurrency
                    tagged["point_index"] = point_index
                    with telemetry_lock:
                        sink_list.append(tagged)

                return _sink

            telemetry = TelemetrySampler(
                metrics_endpoint=metrics_endpoint(config.base_url),
                metrics_transport=config.metrics_transport,
                gpu_sampler=config.gpu_sampler,
                clock=config.clock,
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
            timing=config.timing,
            transport=config.transport,
            clock=config.clock,
            sleep=config.sleep,
            idle_check=idle_check,
            telemetry=telemetry,
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
        config, dataset_sha256, len(balanced_records), server_identity
    )

    review_table = [
        {
            "concurrency": summary["concurrency_target"],
            "completed_requests_per_second": summary["completed_requests_per_second"],
            "completed_total_tokens_per_second": summary[
                "completed_total_tokens_per_second"
            ],
            "adjacent_throughput_gain": summary["adjacent_throughput_gain"],
            "run_valid": summary["run_valid"],
            "invalidation_reasons": summary["invalidation_reasons"],
        }
        for summary in point_summaries
    ]

    experiment_summary = {
        "schema_version": EXPERIMENT_SUMMARY_SCHEMA_VERSION,
        "run_id": config.run_id,
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
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--no-telemetry",
        action="store_true",
        help="disable vLLM /metrics and GPU telemetry collection (diagnostic only)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    config = ExperimentConfig(
        profiling_jsonl=args.profiling_jsonl,
        output_dir=args.output_dir,
        base_url=args.base_url,
        model=args.model,
        tokenizer_revision=args.tokenizer_revision,
        vllm_version=args.vllm_version,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        generation_config=args.generation_config,
        concurrency_ladder=args.concurrency,
        timing=TimingConfig(
            settling_seconds=args.settling_seconds,
            measurement_seconds=args.measurement_seconds,
            drain_timeout_seconds=args.drain_timeout_seconds,
            metrics_interval_seconds=args.metrics_interval_seconds,
            request_timeout_seconds=args.request_timeout_seconds,
        ),
        run_id=args.run_id or default_run_id(),
        collect_telemetry=not args.no_telemetry,
    )

    try:
        bundle = run_experiment(config)
        write_artifacts(args.output_dir, bundle)
    except (ProfilingError, smoke.SmokeError, OSError) as error:
        print(f"balanced-bucket profiling failed: {error}", file=sys.stderr)
        return 1

    print(f"run_id: {config.run_id}")
    print(f"artifacts written to: {args.output_dir}")
    for row in bundle["summary"]["review_table"]:
        print(
            f"  C={row['concurrency']:>4} "
            f"req/s={row['completed_requests_per_second']:.4f} "
            f"tok/s={row['completed_total_tokens_per_second']:.2f} "
            f"gain={row['adjacent_throughput_gain']} "
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


if __name__ == "__main__":
    raise SystemExit(main())

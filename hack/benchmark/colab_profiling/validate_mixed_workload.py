#!/usr/bin/env python3
"""Open-loop mixed-workload composition-validation harness for #1546.

Pure single-bucket profiling (``profile_bucket.py``) has established that
independently measured, monolithic, non-P/D bucket capacities
``V_M^(b)`` are repeatable under the fixed serving configuration. That does
NOT by itself prove those numbers are USEFUL: it does not show that they
predict anything about a workload that mixes buckets together.

This module implements the next research gate: an OPEN-LOOP mixed-workload
load generator that offers all participating buckets simultaneously at
deterministic arrival rates derived from a target composition ratio

::

    rho_pred = sum_b lambda'_b / V_M^(b)
    lambda'_b = request_rate_b * W_b   (W_b = L_in + L_out for bucket b)

and records raw evidence (achieved rates, engine-side throughput, waiting/
running/outstanding trends, scheduling-lag/client-capacity diagnostics) for
a HUMAN to review. It never decides whether the composition model held.

This is a fixed-rate OPEN-LOOP generator, not a closed-loop one:

* the independent variable is the planned request ARRIVAL RATE, not a
  concurrency cap;
* each bucket's arrivals are scheduled at deterministic, drift-free
  absolute times (``scheduled_time(k) = origin + phase + k / rate``);
* a bucket's next arrival is scheduled purely from the plan -- never from
  when the previous request for that bucket happened to complete -- so the
  generator keeps offering load even while requests are still outstanding;
* the HTTP client concurrency budget (thread-pool size) is an explicit,
  configurable, recorded value, deliberately set far above any expected
  server-side backlog, so client-side thread-pool queueing can never
  masquerade as vLLM's own queueing. If the client budget is ever actually
  reached, the point is invalidated rather than silently reinterpreted.

It reuses, unmodified, the already-accepted machinery from
``profile_bucket.py``: the raw-token ``/v1/completions`` request contract
(``execute_profiling_request``), the mandatory engine-counter throughput
estimator (``summarize_engine_token_throughput``), the vLLM/GPU telemetry
sampler (``TelemetrySampler``/``summarize_vllm_telemetry_window``/
``summarize_gpu_telemetry_window``), the per-bucket prompt cycle
(``PromptCycle``), and the artifact writer (``write_artifacts``). It does
NOT modify ``profile_bucket.py`` or move any of this logic into
``profile_balanced_bucket.py``.

Strictly out of scope (see #1546 decisions): predictor logic, autoscaling,
HPA/KEDA integration, changing the serving configuration, bucket
definitions, dataset generation, the request contract, or ``W = L_in +
L_out``; automatically selecting/accepting a ``V_M`` value; automatically
deciding whether the composition model is valid.
"""

from __future__ import annotations

import argparse
import math
import platform
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import generate_dataset as generator
import profile_bucket as pb
import run_request_smoke as smoke


MANIFEST_SCHEMA_VERSION = "llm-d-colab-mixed-workload-manifest-v1"
REQUEST_RESULT_SCHEMA_VERSION = "llm-d-colab-mixed-workload-request-result-v1"
POINT_SUMMARY_SCHEMA_VERSION = "llm-d-colab-mixed-workload-point-summary-v1"
EXPERIMENT_SUMMARY_SCHEMA_VERSION = "llm-d-colab-mixed-workload-summary-v1"

MANIFEST_FILENAME = pb.MANIFEST_FILENAME
REQUEST_RESULTS_FILENAME = pb.REQUEST_RESULTS_FILENAME
POINT_SUMMARIES_FILENAME = pb.POINT_SUMMARIES_FILENAME
SUMMARY_FILENAME = pb.SUMMARY_FILENAME
VLLM_METRICS_FILENAME = pb.VLLM_METRICS_FILENAME
GPU_METRICS_FILENAME = pb.GPU_METRICS_FILENAME

DEFAULT_SETTLING_SECONDS = 60.0
DEFAULT_MEASUREMENT_SECONDS = 180.0
DEFAULT_DRAIN_TIMEOUT_SECONDS = 300.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = pb.DEFAULT_REQUEST_TIMEOUT_SECONDS
DEFAULT_METRICS_INTERVAL_SECONDS = 1.0

# Deliberately generous: this budget exists so the CLIENT never becomes the
# bottleneck. It must stay far above any outstanding population plausibly
# produced by real server-side backlog for the planned rates/durations; see
# ``client_concurrency_budget_exceeded`` below. It is NOT a concurrency
# target -- this harness is open-loop.
DEFAULT_CLIENT_CONCURRENCY_BUDGET = 4096

# Conservative, explicit, operator-configurable validity thresholds that
# distinguish a client-side load-generator saturation failure from genuine
# server-side backlog (which is expected, valid evidence, not a failure).
DEFAULT_MAX_SCHEDULING_LAG_P95_SECONDS = 0.5
DEFAULT_MAX_ACHIEVED_RATE_RELATIVE_ERROR = 0.10

# The fraction of the measurement window, at each edge, used to compute the
# "first window" / "last window" waiting-population medians.
DEFAULT_WAITING_TREND_WINDOW_FRACTION = 0.2

# Small deterministic per-bucket-index stagger so independent bucket
# streams do not all emit their first arrival at exactly the same instant.
DEFAULT_PHASE_OFFSET_STEP_SECONDS = 0.05

# The first intended experiment (#1546 Part L); fully overridable via
# --target-rho, never the only supported values.
DEFAULT_TARGET_RHO_VALUES: tuple[float, ...] = (0.70, 1.00, 1.15)

ProfilingError = pb.ProfilingError
TransportError = pb.TransportError
HttpResponse = pb.HttpResponse
Transport = pb.Transport
Probe = pb.Probe
GpuSampler = pb.GpuSampler
Clock = pb.Clock
SubprocessRunner = pb.SubprocessRunner
IdleCheck = pb.IdleCheck


# ---------------------------------------------------------------------------
# Composition math (#1546 Part B / L): pure, deterministic, unit-testable.
# ---------------------------------------------------------------------------


def normalize_weights(raw_weights: Mapping[str, float]) -> dict[str, float]:
    """Normalize arbitrary positive raw weights to sum to exactly 1.0.

    ``raw_weights`` describes each bucket's INTENDED relative contribution
    to ``rho_pred`` -- not request counts, not raw token rates. Fails
    closed on an empty mapping or a non-finite/non-positive weight/total.
    """

    if not raw_weights:
        raise ProfilingError("composition weights must not be empty")
    for bucket, weight in raw_weights.items():
        if not math.isfinite(weight) or weight < 0:
            raise ProfilingError(
                f"composition weight for {bucket!r} must be finite and non-negative"
            )
    total = sum(raw_weights.values())
    if not math.isfinite(total) or total <= 0:
        raise ProfilingError(
            "composition weights must sum to a finite, positive value"
        )
    return {bucket: weight / total for bucket, weight in raw_weights.items()}


def derive_bucket_targets(
    *,
    target_rho: float,
    alphas: Mapping[str, float],
    capacities: Mapping[str, float],
    bucket_definitions: Mapping[str, generator.Bucket],
) -> dict[str, dict[str, float]]:
    """Derive each bucket's target rho/lambda/request-rate for one ``target_rho``.

    For bucket ``b`` with normalized weight ``alpha_b``, capacity
    ``V_M(b)``, and work coordinate ``W_b = L_in + L_out``::

        rho_b            = target_rho * alpha_b
        lambda_b         = rho_b * V_M(b)
        request_rate_b   = lambda_b / W_b

    ``W_b`` is read from ``bucket_definitions`` (i.e. from the dataset's
    own bucket geometry) for every bucket independently; this function does
    NOT assume all buckets share the same ``W``. Fails closed on a missing/
    non-positive capacity, an unknown bucket, or a non-positive ``W``.
    """

    if not math.isfinite(target_rho) or target_rho <= 0:
        raise ProfilingError("target_rho must be finite and positive")

    targets: dict[str, dict[str, float]] = {}
    for bucket_name, alpha in alphas.items():
        if bucket_name not in capacities:
            raise ProfilingError(
                f"missing V_M capacity input for bucket {bucket_name!r}"
            )
        capacity = capacities[bucket_name]
        if not math.isfinite(capacity) or capacity <= 0:
            raise ProfilingError(
                f"V_M capacity for bucket {bucket_name!r} must be finite and "
                "positive"
            )
        if bucket_name not in bucket_definitions:
            raise ProfilingError(f"unknown bucket {bucket_name!r}")
        w_b = bucket_definitions[bucket_name].total_target_tokens
        if w_b <= 0:
            raise ProfilingError(f"bucket {bucket_name!r} has non-positive W")

        rho_b = target_rho * alpha
        lambda_b = rho_b * capacity
        request_rate_b = lambda_b / w_b
        targets[bucket_name] = {
            "alpha": alpha,
            "v_m": capacity,
            "w": float(w_b),
            "rho_b": rho_b,
            "lambda_b": lambda_b,
            "target_request_rate": request_rate_b,
        }
    return targets


def compute_rho_from_lambdas(
    lambda_by_bucket: Mapping[str, float], capacities: Mapping[str, float]
) -> float:
    """``rho_pred = sum_b lambda'_b / V_M(b)`` -- the one shared formula used
    for both the target and achieved predictions, so they are directly
    comparable."""

    total = 0.0
    for bucket_name, lambda_value in lambda_by_bucket.items():
        capacity = capacities[bucket_name]
        total += lambda_value / capacity
    return total


# ---------------------------------------------------------------------------
# Deterministic open-loop arrival scheduling (#1546 Part E)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeterministicArrivalSchedule:
    """Deterministic, absolute-time, drift-free open-loop arrival schedule
    for one bucket::

        scheduled_time(k) = origin + phase_seconds + k / request_rate_per_second

    This deliberately does NOT compute the next arrival as
    ``actual_submission_time + inter_arrival_seconds``: that would let
    runtime delay silently shrink the achieved arrival rate (schedule
    drift). Every arrival's scheduled time is derived directly from its own
    index ``k`` and the fixed ``origin``, so a late arrival never postpones
    any later one -- the schedule is fully reproducible from
    ``(origin, phase_seconds, request_rate_per_second)`` alone.
    """

    origin: float
    phase_seconds: float
    request_rate_per_second: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.request_rate_per_second)
            or self.request_rate_per_second <= 0
        ):
            raise ProfilingError("request rate must be finite and positive")
        if not math.isfinite(self.phase_seconds) or self.phase_seconds < 0:
            raise ProfilingError("phase_seconds must be finite and non-negative")

    @property
    def inter_arrival_seconds(self) -> float:
        return 1.0 / self.request_rate_per_second

    def scheduled_time(self, k: int) -> float:
        return self.origin + self.phase_seconds + k * self.inter_arrival_seconds


class ArrivalCounter:
    """Thread-safe, monotonically increasing arrival index for one bucket."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_index = 0

    def next(self) -> int:
        with self._lock:
            k = self._next_index
            self._next_index += 1
        return k


# ---------------------------------------------------------------------------
# Percentile / trend helpers (pure, deterministic)
# ---------------------------------------------------------------------------


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Linear-interpolation percentile (matches common statistics packages'
    default), over an already-sorted, non-empty sequence."""

    if not sorted_values:
        raise ProfilingError("cannot compute a percentile of an empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = fraction * (len(sorted_values) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[int(index)]
    weight = index - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight


def summarize_scheduling_lag(lag_values: Sequence[float | None]) -> dict[str, Any]:
    """count/mean/p50/p95/max of ``actual_submit - scheduled`` across all
    admitted requests in a point (all buckets combined)."""

    values = sorted(value for value in lag_values if value is not None)
    if not values:
        return {"available": False, "count": 0}
    return {
        "available": True,
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": values[-1],
    }


def _linear_trend_slope(series: Sequence[tuple[float, float]]) -> float | None:
    """Ordinary least-squares slope of ``value`` vs. ``timestamp`` (units:
    value-per-second). ``None`` if fewer than two points or zero time
    variance."""

    if len(series) < 2:
        return None
    n = len(series)
    mean_t = sum(t for t, _ in series) / n
    mean_v = sum(v for _, v in series) / n
    numerator = sum((t - mean_t) * (v - mean_v) for t, v in series)
    denominator = sum((t - mean_t) ** 2 for t, _ in series)
    if denominator == 0:
        return None
    return numerator / denominator


def summarize_time_series_with_trend(
    series: Sequence[tuple[float, float]],
    t0: float,
    t1: float,
    first_last_window_fraction: float,
) -> dict[str, Any]:
    """mean/min/max, first/last-window medians, and a linear trend slope for
    one (timestamp, value) telemetry series confined to ``[t0, t1)``.

    The "first window" / "last window" used for the medians are the first
    and last ``first_last_window_fraction`` of the ``[t0, t1)`` duration
    (e.g. the default ``0.2`` uses the first/last 20% of the window); this
    fraction is always recorded alongside the medians for auditability.
    """

    if not series:
        return {"available": False}
    values = [value for _, value in series]
    window_duration = t1 - t0
    edge_width = window_duration * first_last_window_fraction
    first_window_values = [v for t, v in series if t < t0 + edge_width]
    last_window_values = [v for t, v in series if t >= t1 - edge_width]
    first_median = statistics.median(first_window_values) if first_window_values else None
    last_median = statistics.median(last_window_values) if last_window_values else None
    return {
        "available": True,
        "sample_count": len(series),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "first_last_window_fraction": first_last_window_fraction,
        "first_window_median": first_median,
        "last_window_median": last_median,
        "last_minus_first_median": (
            last_median - first_median
            if first_median is not None and last_median is not None
            else None
        ),
        "trend_slope_per_second": _linear_trend_slope(series),
    }


def summarize_running_series(series: Sequence[tuple[float, float]]) -> dict[str, Any]:
    """mean/max, and the fraction of samples AT the observed max ("ceiling"),
    for one running-population series. The ceiling is whatever was actually
    observed in THIS run; no universal value (e.g. 256) is ever assumed."""

    if not series:
        return {"available": False}
    values = [value for _, value in series]
    ceiling = max(values)
    at_ceiling = sum(1 for value in values if value == ceiling)
    return {
        "available": True,
        "sample_count": len(values),
        "mean": sum(values) / len(values),
        "max": ceiling,
        "fraction_of_samples_at_observed_ceiling": at_ceiling / len(values),
    }


def _ok_window_samples(
    samples: Sequence[Mapping[str, Any]], t0: float, t1: float
) -> list[Mapping[str, Any]]:
    return [
        sample
        for sample in samples
        if sample.get("status") == "ok"
        and sample.get("timestamp") is not None
        and t0 <= sample["timestamp"] < t1
    ]


def _timestamped_metric_series(
    samples: Sequence[Mapping[str, Any]], metric_names: Sequence[str]
) -> list[tuple[float, float]]:
    """Like ``profile_bucket._metric_series_from_ok_samples`` but also keeps
    each sample's timestamp, needed for trend/median computation."""

    series: list[tuple[float, float]] = []
    for sample in samples:
        known = sample.get("known_metrics") or {}
        for name in metric_names:
            entry = known.get(name)
            if entry and entry.get("present") and entry.get("samples"):
                value = sum(metric["value"] for metric in entry["samples"])
                series.append((sample["timestamp"], value))
                break
    return series


def summarize_saturation_evidence(
    vllm_samples: Sequence[Mapping[str, Any]],
    t0: float,
    t1: float,
    *,
    first_last_window_fraction: float = DEFAULT_WAITING_TREND_WINDOW_FRACTION,
) -> dict[str, Any]:
    """Waiting/running saturation evidence for one point's measurement
    window (#1546 Part J). Diagnostic only; never decides ``run_valid``."""

    window_samples = _ok_window_samples(vllm_samples, t0, t1)
    waiting_series = _timestamped_metric_series(
        window_samples, pb.VLLM_WAITING_METRIC_CANDIDATES
    )
    running_series = _timestamped_metric_series(
        window_samples, pb.VLLM_RUNNING_METRIC_CANDIDATES
    )
    return {
        "measurement_sample_count": len(window_samples),
        "waiting": summarize_time_series_with_trend(
            waiting_series, t0, t1, first_last_window_fraction
        ),
        "running": summarize_running_series(running_series),
    }


# ---------------------------------------------------------------------------
# Open-loop admission state shared by every bucket's arrival thread
# ---------------------------------------------------------------------------


class _MixedLoadState:
    """Thread-safe outstanding/submission bookkeeping shared across all of
    one point's per-bucket arrival threads (analogous to
    ``profile_bucket.run_load_point``'s single-bucket counters, generalized
    to N independently paced streams)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.outstanding = 0
        self.max_observed_outstanding = 0
        self.submitted_count = 0
        self.admission_closed = threading.Event()
        self.all_drained = threading.Event()

    def record_admission(self) -> None:
        with self.lock:
            self.outstanding += 1
            self.max_observed_outstanding = max(
                self.max_observed_outstanding, self.outstanding
            )
            self.submitted_count += 1

    def undo_admission(self) -> None:
        with self.lock:
            self.outstanding -= 1
            self.submitted_count -= 1

    def snapshot_outstanding(self) -> int:
        with self.lock:
            return self.outstanding

    def record_completion(self) -> bool:
        """Decrement outstanding; return True iff this completion just
        drained the point (admission already closed and outstanding hit 0)."""

        with self.lock:
            self.outstanding -= 1
            return self.admission_closed.is_set() and self.outstanding == 0


def _on_bucket_arrival_done(
    future: Any,
    *,
    sequence: int,
    scheduled_time: float,
    bucket_name: str,
    record: Mapping[str, Any],
    state: _MixedLoadState,
    results: list[dict[str, Any]],
    results_lock: threading.Lock,
    run_id: str,
    point_index: int,
    target_rho: float,
) -> None:
    try:
        result = future.result()
    except Exception as error:  # defensive: execute_profiling_request never raises
        result = {
            "request_id": None,
            "prompt_hash": None,
            "expected_prompt_tokens": None,
            "target_completion_tokens": None,
            "observed_prompt_tokens": None,
            "observed_completion_tokens": None,
            "observed_total_tokens": None,
            "finish_reason": None,
            "http_status": None,
            "passed": False,
            "failure_reasons": [{"reason": "unexpected_exception", "detail": str(error)}],
            "submit_monotonic_s": None,
            "terminal_monotonic_s": time.monotonic(),
            "latency_s": None,
        }

    submit_ts = result.get("submit_monotonic_s")
    scheduling_lag = submit_ts - scheduled_time if submit_ts is not None else None

    tagged = dict(result)
    tagged.update(
        {
            "schema_version": REQUEST_RESULT_SCHEMA_VERSION,
            "run_id": run_id,
            "point_index": point_index,
            "target_rho": target_rho,
            "bucket": bucket_name,
            "sequence": sequence,
            "scheduled_monotonic_s": scheduled_time,
            "scheduling_lag_s": scheduling_lag,
            "total_logical_work": record.get("total_target_tokens"),
        }
    )
    with results_lock:
        results.append(tagged)

    if state.record_completion():
        state.all_drained.set()


def _bucket_arrival_loop(
    *,
    bucket_name: str,
    cycle: pb.PromptCycle,
    schedule: DeterministicArrivalSchedule,
    counter: ArrivalCounter,
    endpoint: str,
    model: str,
    request_timeout_seconds: float,
    transport: Transport,
    clock: Clock,
    executor: ThreadPoolExecutor,
    state: _MixedLoadState,
    results: list[dict[str, Any]],
    results_lock: threading.Lock,
    run_id: str,
    point_index: int,
    target_rho: float,
) -> None:
    """One bucket's independent, deterministically paced OPEN-LOOP arrival
    stream. Critically, the next arrival's scheduled time never depends on
    whether the previous arrival has completed -- only on the fixed
    schedule -- so the generator keeps offering load while requests are
    still outstanding, exactly as an open-loop generator must."""

    while not state.admission_closed.is_set():
        sequence = counter.next()
        scheduled_time = schedule.scheduled_time(sequence)
        delay = scheduled_time - clock()
        if delay > 0:
            # Event.wait returns True (and we stop immediately) if admission
            # was closed mid-wait; otherwise it waits the full delay and we
            # proceed to submit exactly on schedule.
            if state.admission_closed.wait(delay):
                break
        if state.admission_closed.is_set():
            break

        record, _ = cycle.next()
        state.record_admission()
        try:
            future = executor.submit(
                pb.execute_profiling_request,
                record,
                endpoint,
                model,
                request_timeout_seconds,
                transport,
                clock,
            )
        except RuntimeError:
            # Executor is shutting down; treat as an unadmitted arrival.
            state.undo_admission()
            break
        future.add_done_callback(
            lambda done, seq=sequence, sched=scheduled_time, rec=record: (
                _on_bucket_arrival_done(
                    done,
                    sequence=seq,
                    scheduled_time=sched,
                    bucket_name=bucket_name,
                    record=rec,
                    state=state,
                    results=results,
                    results_lock=results_lock,
                    run_id=run_id,
                    point_index=point_index,
                    target_rho=target_rho,
                )
            )
        )


# ---------------------------------------------------------------------------
# Point lifecycle (#1546 Part G)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MixedTimingConfig:
    settling_seconds: float = DEFAULT_SETTLING_SECONDS
    measurement_seconds: float = DEFAULT_MEASUREMENT_SECONDS
    drain_timeout_seconds: float = DEFAULT_DRAIN_TIMEOUT_SECONDS
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    metrics_interval_seconds: float = DEFAULT_METRICS_INTERVAL_SECONDS

    def validate(self) -> None:
        for name in (
            "settling_seconds",
            "measurement_seconds",
            "drain_timeout_seconds",
            "request_timeout_seconds",
            "metrics_interval_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ProfilingError(f"{name} must be finite and positive")


def _empty_engine_token_throughput(reason: str) -> dict[str, Any]:
    return pb._empty_engine_token_throughput(available=False, unavailable_reason=reason)


def _empty_mixed_point_summary(
    *,
    run_id: str,
    point_index: int,
    target_rho: float,
    point_start: float,
    t0: float,
    t1: float,
    timing: MixedTimingConfig,
    client_concurrency_budget: int,
    bucket_targets: Mapping[str, Mapping[str, float]],
    execution_skipped_reason: str,
    extra_invalidation_reasons: Sequence[str],
) -> dict[str, Any]:
    per_bucket_summary = {
        bucket_name: {
            "v_m_input": target["v_m"],
            "w_b": target["w"],
            "alpha_b": target["alpha"],
            "target_rho_contribution": target["rho_b"],
            "target_logical_token_rate": target["lambda_b"],
            "target_request_rate": target["target_request_rate"],
            "requests_submitted": 0,
            "achieved_request_count_in_window": 0,
            "achieved_request_rate": 0.0,
            "achieved_logical_token_rate": 0.0,
            "achieved_rho_contribution": 0.0,
            "achieved_rate_relative_error": None,
            "terminal_completions_in_measurement": 0,
            "valid_terminal_completions_in_measurement": 0,
            "failure_counts": {},
        }
        for bucket_name, target in bucket_targets.items()
    }
    rho_pred_target = compute_rho_from_lambdas(
        {b: t["lambda_b"] for b, t in bucket_targets.items()},
        {b: t["v_m"] for b, t in bucket_targets.items()},
    )
    return {
        "schema_version": POINT_SUMMARY_SCHEMA_VERSION,
        "run_id": run_id,
        "point_index": point_index,
        "target_rho": target_rho,
        "execution_skipped": True,
        "execution_skipped_reason": execution_skipped_reason,
        "point_start_s": point_start,
        "settling_start_s": point_start,
        "measurement_t0_s": t0,
        "measurement_t1_s": t1,
        "measurement_duration_s": timing.measurement_seconds,
        "client_concurrency_budget": client_concurrency_budget,
        "max_observed_outstanding": 0,
        "outstanding_at_t0": 0,
        "outstanding_at_t1": 0,
        "outstanding_delta": 0,
        "outstanding_after_drain": 0,
        "drain_duration_s": 0.0,
        "drain_outcome": "drained",
        "requests_submitted_total": 0,
        "scheduling_lag_seconds": {"available": False, "count": 0},
        "buckets": per_bucket_summary,
        "rho_pred_target": rho_pred_target,
        "rho_pred_achieved": 0.0,
        "engine_token_throughput": _empty_engine_token_throughput(execution_skipped_reason),
        "vllm_telemetry": {"available": False, "reason": execution_skipped_reason},
        "gpu_telemetry": {"available": False, "reason": execution_skipped_reason},
        "saturation_evidence": {
            "waiting": {"available": False},
            "running": {"available": False},
            "outstanding": {"at_t0": 0, "at_t1": 0, "delta": 0},
            "engine": {
                "prompt_tokens_per_second": None,
                "generation_tokens_per_second": None,
                "total_tokens_per_second": None,
            },
        },
        "run_valid": False,
        "invalidation_reasons": list(extra_invalidation_reasons),
        "capacity_quantity_warning": (
            "This is monolithic non-P/D V_M evidence; not physical "
            "KV-release throughput, not isolated decoder V_D, and not "
            "SLO-safe operating capacity."
        ),
    }


def run_mixed_load_point(
    *,
    target_rho: float,
    point_index: int,
    bucket_cycles: Mapping[str, pb.PromptCycle],
    bucket_targets: Mapping[str, Mapping[str, float]],
    endpoint: str,
    model: str,
    timing: MixedTimingConfig,
    transport: Transport,
    client_concurrency_budget: int,
    max_scheduling_lag_p95_seconds: float,
    max_achieved_rate_relative_error: float,
    waiting_trend_window_fraction: float,
    phase_offset_step_seconds: float,
    clock: Clock = time.monotonic,
    idle_check: IdleCheck | None = None,
    telemetry_config: pb.TelemetryConfig | None = None,
    run_id: str = "unassigned",
    previous_point_drained: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one open-loop mixed-workload target-rho point end to end.

    Fails closed (zero admission, ``execution_skipped=True``) if the
    PREVIOUS point did not drain successfully, or if this point's own
    mandatory precondition check fails -- mirroring
    ``profile_bucket.run_load_point``'s precondition fail-closed contract,
    generalized across the point sequence (#1546 Part G: "every point must
    start only after the previous point has drained successfully").
    """

    timing.validate()
    if client_concurrency_budget <= 0:
        raise ProfilingError("client_concurrency_budget must be positive")

    extra_invalidation_reasons: list[str] = []

    if not previous_point_drained:
        extra_invalidation_reasons.append("previous_point_drain_failed")
        point_start = clock()
        t0 = point_start + timing.settling_seconds
        t1 = t0 + timing.measurement_seconds
        return (
            _empty_mixed_point_summary(
                run_id=run_id,
                point_index=point_index,
                target_rho=target_rho,
                point_start=point_start,
                t0=t0,
                t1=t1,
                timing=timing,
                client_concurrency_budget=client_concurrency_budget,
                bucket_targets=bucket_targets,
                execution_skipped_reason="previous_point_drain_failed",
                extra_invalidation_reasons=extra_invalidation_reasons,
            ),
            [],
        )

    precondition_ok = True
    precondition_reason: str | None = None
    if idle_check is not None:
        precondition_ok, precondition_reason = idle_check()
        if not precondition_ok:
            extra_invalidation_reasons.append(f"precondition_failed:{precondition_reason}")

    if not precondition_ok:
        point_start = clock()
        t0 = point_start + timing.settling_seconds
        t1 = t0 + timing.measurement_seconds
        return (
            _empty_mixed_point_summary(
                run_id=run_id,
                point_index=point_index,
                target_rho=target_rho,
                point_start=point_start,
                t0=t0,
                t1=t1,
                timing=timing,
                client_concurrency_budget=client_concurrency_budget,
                bucket_targets=bucket_targets,
                execution_skipped_reason=f"precondition_failed:{precondition_reason}",
                extra_invalidation_reasons=extra_invalidation_reasons,
            ),
            [],
        )

    # --- Real open-loop execution ---
    local_vllm_samples: list[dict[str, Any]] = []
    local_gpu_samples: list[dict[str, Any]] = []
    local_telemetry_lock = threading.Lock()
    telemetry: pb.TelemetrySampler | None = None
    if telemetry_config is not None:

        def _local_vllm_sink(entry: dict[str, Any]) -> None:
            with local_telemetry_lock:
                local_vllm_samples.append(entry)
            telemetry_config.on_vllm_sample(entry)

        def _local_gpu_sink(entry: dict[str, Any]) -> None:
            with local_telemetry_lock:
                local_gpu_samples.append(entry)
            telemetry_config.on_gpu_sample(entry)

        telemetry = pb.TelemetrySampler(
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

    state = _MixedLoadState()
    results: list[dict[str, Any]] = []
    results_lock = threading.Lock()
    executor = ThreadPoolExecutor(
        max_workers=client_concurrency_budget,
        thread_name_prefix=f"mixed-point{point_index}",
    )

    point_start = clock()
    t0 = point_start + timing.settling_seconds
    t1 = t0 + timing.measurement_seconds

    threads: list[threading.Thread] = []
    for bucket_index, bucket_name in enumerate(sorted(bucket_targets)):
        target = bucket_targets[bucket_name]
        schedule = DeterministicArrivalSchedule(
            origin=point_start,
            phase_seconds=bucket_index * phase_offset_step_seconds,
            request_rate_per_second=target["target_request_rate"],
        )
        thread = threading.Thread(
            target=_bucket_arrival_loop,
            kwargs=dict(
                bucket_name=bucket_name,
                cycle=bucket_cycles[bucket_name],
                schedule=schedule,
                counter=ArrivalCounter(),
                endpoint=endpoint,
                model=model,
                request_timeout_seconds=timing.request_timeout_seconds,
                transport=transport,
                clock=clock,
                executor=executor,
                state=state,
                results=results,
                results_lock=results_lock,
                run_id=run_id,
                point_index=point_index,
                target_rho=target_rho,
            ),
            name=f"mixed-arrivals-{bucket_name}",
            daemon=True,
        )
        threads.append(thread)

    for thread in threads:
        thread.start()

    remaining_to_t0 = t0 - clock()
    if remaining_to_t0 > 0:
        time.sleep(remaining_to_t0)
    outstanding_at_t0 = state.snapshot_outstanding()

    remaining_to_t1 = t1 - clock()
    if remaining_to_t1 > 0:
        time.sleep(remaining_to_t1)
    with state.lock:
        state.admission_closed.set()
        outstanding_at_t1 = state.outstanding
        if outstanding_at_t1 == 0:
            state.all_drained.set()

    for thread in threads:
        thread.join(timeout=5.0)

    drain_start = clock()
    drained = state.all_drained.wait(timing.drain_timeout_seconds)
    drain_duration = clock() - drain_start

    with state.lock:
        outstanding_after_drain = state.outstanding

    executor.shutdown(wait=True)

    if telemetry is not None:
        telemetry.stop()

    if idle_check is not None:
        ok, reason = idle_check()
        if not ok:
            extra_invalidation_reasons.append(f"server_unreachable_after_point:{reason}")

    with results_lock:
        results_snapshot = list(results)

    if telemetry_config is not None:
        with local_telemetry_lock:
            vllm_samples_snapshot = list(local_vllm_samples)
            gpu_samples_snapshot = list(local_gpu_samples)
        vllm_telemetry_summary = pb.summarize_vllm_telemetry_window(
            vllm_samples_snapshot, t0, t1
        )
        gpu_telemetry_summary = pb.summarize_gpu_telemetry_window(
            gpu_samples_snapshot, t0, t1
        )
        engine_token_throughput = pb.summarize_engine_token_throughput(
            vllm_samples_snapshot, t0, t1
        )
        saturation_waiting_running = summarize_saturation_evidence(
            vllm_samples_snapshot,
            t0,
            t1,
            first_last_window_fraction=waiting_trend_window_fraction,
        )
    else:
        vllm_telemetry_summary = {"available": False, "reason": "telemetry_not_collected"}
        gpu_telemetry_summary = {"available": False, "reason": "telemetry_not_collected"}
        engine_token_throughput = _empty_engine_token_throughput("telemetry_not_collected")
        saturation_waiting_running = {
            "measurement_sample_count": 0,
            "waiting": {"available": False},
            "running": {"available": False},
        }

    if not engine_token_throughput["available"]:
        extra_invalidation_reasons.append(
            "engine_token_throughput_unavailable:"
            f"{engine_token_throughput['unavailable_reason']}"
        )

    if state.max_observed_outstanding >= client_concurrency_budget:
        extra_invalidation_reasons.append(
            "client_concurrency_budget_exceeded:"
            f"{state.max_observed_outstanding}/{client_concurrency_budget}"
        )

    if not drained:
        extra_invalidation_reasons.append("drain_timeout")

    per_bucket_summary: dict[str, Any] = {}
    scheduling_lag_values: list[float | None] = []
    for bucket_name, target in bucket_targets.items():
        bucket_results = [r for r in results_snapshot if r["bucket"] == bucket_name]
        for result in bucket_results:
            terminal = result.get("terminal_monotonic_s")
            result["in_measurement_window"] = terminal is not None and t0 <= terminal < t1
            submit_ts = result.get("submit_monotonic_s")
            if submit_ts is None:
                result["phase"] = "unknown"
            elif submit_ts < t0:
                result["phase"] = "settling"
            elif submit_ts < t1:
                result["phase"] = "measurement"
            else:
                result["phase"] = "drain"
            scheduling_lag_values.append(result.get("scheduling_lag_s"))

        submitted_in_window = [
            r
            for r in bucket_results
            if r.get("submit_monotonic_s") is not None and t0 <= r["submit_monotonic_s"] < t1
        ]
        terminal_in_measurement = [r for r in bucket_results if r.get("in_measurement_window")]
        valid_terminal_in_measurement = [
            r for r in terminal_in_measurement if r.get("passed")
        ]
        failure_counts = Counter(
            failure["reason"]
            for r in bucket_results
            for failure in r.get("failure_reasons", [])
        )

        w_b = target["w"]
        v_m = target["v_m"]
        achieved_request_count = len(submitted_in_window)
        achieved_request_rate = achieved_request_count / timing.measurement_seconds
        achieved_lambda = achieved_request_rate * w_b
        achieved_rho_contribution = achieved_lambda / v_m
        target_request_rate = target["target_request_rate"]
        achieved_rate_relative_error = (
            abs(achieved_request_rate - target_request_rate) / target_request_rate
            if target_request_rate
            else None
        )
        if (
            achieved_rate_relative_error is not None
            and achieved_rate_relative_error > max_achieved_rate_relative_error
        ):
            extra_invalidation_reasons.append(
                "achieved_rate_relative_error_exceeds_threshold:"
                f"{bucket_name}:{achieved_rate_relative_error:.4f}>"
                f"{max_achieved_rate_relative_error}"
            )

        per_bucket_summary[bucket_name] = {
            "v_m_input": v_m,
            "w_b": w_b,
            "alpha_b": target["alpha"],
            "target_rho_contribution": target["rho_b"],
            "target_logical_token_rate": target["lambda_b"],
            "target_request_rate": target_request_rate,
            "requests_submitted": len(bucket_results),
            "achieved_request_count_in_window": achieved_request_count,
            "achieved_request_rate": achieved_request_rate,
            "achieved_logical_token_rate": achieved_lambda,
            "achieved_rho_contribution": achieved_rho_contribution,
            "achieved_rate_relative_error": achieved_rate_relative_error,
            "terminal_completions_in_measurement": len(terminal_in_measurement),
            "valid_terminal_completions_in_measurement": len(valid_terminal_in_measurement),
            "failure_counts": dict(sorted(failure_counts.items())),
        }

    scheduling_lag_summary = summarize_scheduling_lag(scheduling_lag_values)
    if (
        scheduling_lag_summary.get("available")
        and scheduling_lag_summary["p95"] > max_scheduling_lag_p95_seconds
    ):
        extra_invalidation_reasons.append(
            "scheduling_lag_p95_exceeds_threshold:"
            f"{scheduling_lag_summary['p95']:.4f}>{max_scheduling_lag_p95_seconds}"
        )

    rho_pred_target = compute_rho_from_lambdas(
        {b: t["lambda_b"] for b, t in bucket_targets.items()},
        {b: t["v_m"] for b, t in bucket_targets.items()},
    )
    rho_pred_achieved = compute_rho_from_lambdas(
        {b: s["achieved_logical_token_rate"] for b, s in per_bucket_summary.items()},
        {b: t["v_m"] for b, t in bucket_targets.items()},
    )

    summary = {
        "schema_version": POINT_SUMMARY_SCHEMA_VERSION,
        "run_id": run_id,
        "point_index": point_index,
        "target_rho": target_rho,
        "execution_skipped": False,
        "execution_skipped_reason": None,
        "point_start_s": point_start,
        "settling_start_s": point_start,
        "measurement_t0_s": t0,
        "measurement_t1_s": t1,
        "measurement_duration_s": timing.measurement_seconds,
        "client_concurrency_budget": client_concurrency_budget,
        "max_observed_outstanding": state.max_observed_outstanding,
        "outstanding_at_t0": outstanding_at_t0,
        "outstanding_at_t1": outstanding_at_t1,
        "outstanding_delta": outstanding_at_t1 - outstanding_at_t0,
        "outstanding_after_drain": outstanding_after_drain,
        "drain_duration_s": drain_duration,
        "drain_outcome": "drained" if drained else "timed_out",
        "requests_submitted_total": state.submitted_count,
        "scheduling_lag_seconds": scheduling_lag_summary,
        "buckets": per_bucket_summary,
        "rho_pred_target": rho_pred_target,
        "rho_pred_achieved": rho_pred_achieved,
        "engine_token_throughput": engine_token_throughput,
        "vllm_telemetry": vllm_telemetry_summary,
        "gpu_telemetry": gpu_telemetry_summary,
        "saturation_evidence": {
            "waiting": saturation_waiting_running["waiting"],
            "running": saturation_waiting_running["running"],
            "outstanding": {
                "at_t0": outstanding_at_t0,
                "at_t1": outstanding_at_t1,
                "delta": outstanding_at_t1 - outstanding_at_t0,
            },
            "engine": {
                "prompt_tokens_per_second": engine_token_throughput.get(
                    "prompt_tokens_per_second"
                ),
                "generation_tokens_per_second": engine_token_throughput.get(
                    "generation_tokens_per_second"
                ),
                "total_tokens_per_second": engine_token_throughput.get(
                    "total_tokens_per_second"
                ),
            },
        },
        "run_valid": len(extra_invalidation_reasons) == 0,
        "invalidation_reasons": extra_invalidation_reasons,
        "capacity_quantity_warning": (
            "V_M inputs and derived rho values are monolithic non-P/D "
            "logical total-token throughput evidence; NOT physical "
            "KV-release throughput, NOT isolated decoder V_D, and NOT "
            "SLO-safe operating capacity. Expected overload/backlog under "
            "rho>=1 (growing waiting/outstanding population) is NOT itself "
            "a run-invalid condition; only harness/precondition/client-"
            "generator failures are."
        ),
    }
    return summary, results_snapshot


# ---------------------------------------------------------------------------
# Experiment orchestration
# ---------------------------------------------------------------------------


@dataclass
class MixedExperimentConfig:
    """All inputs that affect one open-loop mixed-workload experiment."""

    profiling_jsonl: Path
    output_dir: Path
    base_url: str
    model: str
    tokenizer_revision: str
    capacities: dict[str, float] = field(default_factory=dict)
    raw_weights: dict[str, float] | None = None
    target_rho_values: tuple[float, ...] = DEFAULT_TARGET_RHO_VALUES
    vllm_version: str = "0.28.0"
    dtype: str = "float16"
    tensor_parallel_size: int = 1
    max_model_len: int = 1024
    generation_config: str = "vllm"
    prefix_caching: bool = False
    gpu_memory_utilization: float = pb.DEFAULT_GPU_MEMORY_UTILIZATION
    timing: MixedTimingConfig = field(default_factory=MixedTimingConfig)
    client_concurrency_budget: int = DEFAULT_CLIENT_CONCURRENCY_BUDGET
    max_scheduling_lag_p95_seconds: float = DEFAULT_MAX_SCHEDULING_LAG_P95_SECONDS
    max_achieved_rate_relative_error: float = DEFAULT_MAX_ACHIEVED_RATE_RELATIVE_ERROR
    waiting_trend_window_fraction: float = DEFAULT_WAITING_TREND_WINDOW_FRACTION
    phase_offset_step_seconds: float = DEFAULT_PHASE_OFFSET_STEP_SECONDS
    run_id: str = ""
    transport: Transport = smoke.post_completion
    probe_transport: Probe = pb.http_get
    metrics_transport: Probe = pb.http_get
    gpu_sampler: GpuSampler = pb.sample_gpu_telemetry
    fingerprint_runner: SubprocessRunner = subprocess.run
    clock: Clock = time.monotonic
    collect_telemetry: bool = True

    @property
    def alphas(self) -> dict[str, float]:
        raw = self.raw_weights if self.raw_weights is not None else {
            bucket: 1.0 for bucket in self.capacities
        }
        return normalize_weights(raw)

    def validate(self) -> None:
        if not self.model.strip():
            raise ProfilingError("model must not be empty")
        if not self.tokenizer_revision.strip():
            raise ProfilingError("tokenizer revision must not be empty")
        if not (0.0 < self.gpu_memory_utilization <= 1.0):
            raise ProfilingError("gpu_memory_utilization must be in (0.0, 1.0]")
        if not self.capacities:
            raise ProfilingError(
                "at least one bucket V_M capacity input must be supplied"
            )
        for bucket_name, capacity in self.capacities.items():
            pb.resolve_bucket(bucket_name)
            if not math.isfinite(capacity) or capacity <= 0:
                raise ProfilingError(
                    f"V_M capacity for bucket {bucket_name!r} must be finite "
                    "and positive"
                )
        if self.raw_weights is not None and set(self.raw_weights) != set(self.capacities):
            raise ProfilingError(
                "composition weight bucket names must exactly match capacity "
                "bucket names"
            )
        if not self.target_rho_values:
            raise ProfilingError("at least one target rho value must be supplied")
        for rho in self.target_rho_values:
            if not math.isfinite(rho) or rho <= 0:
                raise ProfilingError("target rho values must be finite and positive")
        if self.client_concurrency_budget <= 0:
            raise ProfilingError("client_concurrency_budget must be positive")
        if not (0 < self.waiting_trend_window_fraction <= 0.5):
            raise ProfilingError("waiting_trend_window_fraction must be in (0, 0.5]")
        if self.phase_offset_step_seconds < 0:
            raise ProfilingError("phase_offset_step_seconds must be non-negative")
        self.timing.validate()
        # Fail closed rather than guess: exercise the normalization/lookup
        # logic now so a malformed weight/capacity set is caught before any
        # HTTP traffic is generated.
        _ = self.alphas


def default_run_id() -> str:
    return "mixed-workload-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_manifest(
    config: MixedExperimentConfig,
    bucket_definitions: Mapping[str, generator.Bucket],
    dataset_sha256: str,
    total_profiling_record_count: int,
    server_identity: Mapping[str, Any],
) -> dict[str, Any]:
    alphas = config.alphas
    composition_plan = [
        {
            "target_rho": rho,
            "buckets": derive_bucket_targets(
                target_rho=rho,
                alphas=alphas,
                capacities=config.capacities,
                bucket_definitions=bucket_definitions,
            ),
        }
        for rho in config.target_rho_values
    ]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "phase": "mixed_workload_open_loop_composition_validation",
        "research_question": (
            "Can independently profiled V_M^(b) values (profile_bucket.py) "
            "predict the saturation boundary of a held-out MIXED workload "
            "via rho_pred = sum_b lambda'_b / V_M^(b)? Pure-bucket profiling "
            "being valid does NOT itself prove this; this experiment is the "
            "decisive abstraction-validation gate."
        ),
        "non_goals": [
            "not a predictor implementation",
            "not autoscaling or HPA/KEDA integration",
            "not an automatic composition-model pass/fail decision",
            "not a change to bucket definitions, W=L_in+L_out, or the "
            "request contract",
            "not physical KV-release throughput, isolated decoder V_D, or "
            "SLO-safe capacity",
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
                    "Declared by the operator invoking this harness; not "
                    "independently verified beyond /v1/models identity and a "
                    "best-effort /version probe."
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
            "total_profiling_record_count": total_profiling_record_count,
            "bucket_definitions": {
                name: {
                    "input_tokens": bucket.input_tokens,
                    "target_output_tokens": bucket.target_output_tokens,
                    "total_target_tokens": bucket.total_target_tokens,
                }
                for name, bucket in bucket_definitions.items()
            },
        },
        "work_coordinate": {
            "definition": "W_b = L_in + L_out (logical tokens per request for bucket b)",
            "note": (
                "W_b is read per bucket from the dataset's own bucket "
                "definition; it is never hard-coded or assumed equal across "
                "buckets, even though all three current buckets happen to "
                "share W=512."
            ),
        },
        "primary_capacity_estimator": {
            "name": "engine_token_throughput.total_tokens_per_second",
            "formula": (
                "(delta(vllm:prompt_tokens_total) + "
                "delta(vllm:generation_tokens_total)) / telemetry_duration_s, "
                "using the first and last valid ('status'=='ok') vLLM "
                "/metrics samples with timestamp in [T0,T1). Identical, "
                "unmodified contract to profile_bucket.py's mandatory "
                "engine-counter estimator; never replaced by completion "
                "throughput."
            ),
        },
        "capacities_supplied": dict(config.capacities),
        "composition_weights": {
            "raw_weights": (
                config.raw_weights
                if config.raw_weights is not None
                else {bucket: 1.0 for bucket in config.capacities}
            ),
            "normalized_alpha": alphas,
            "note": (
                "alpha_b describes bucket b's INTENDED contribution to "
                "rho_pred, NOT equal request counts and NOT equal raw token "
                "rates."
            ),
        },
        "composition_model": {
            "formula": "rho_pred = sum_b lambda'_b / V_M(b); lambda'_b = request_rate_b * W_b",
            "target_rho_values": list(config.target_rho_values),
            "composition_plan": composition_plan,
        },
        "timing": {
            "settling_seconds": config.timing.settling_seconds,
            "measurement_seconds": config.timing.measurement_seconds,
            "drain_timeout_seconds": config.timing.drain_timeout_seconds,
            "request_timeout_seconds": config.timing.request_timeout_seconds,
            "metrics_interval_seconds": config.timing.metrics_interval_seconds,
        },
        "load_generator": {
            "model": "deterministic_open_loop_absolute_time_per_bucket_arrival_schedule",
            "note": (
                "scheduled_time(k,b) = point_start + phase_b + k / "
                "request_rate_b. NEVER computed as previous_actual + "
                "interval (that would drift the requested rate under "
                "runtime delay). Each bucket is paced by an independent "
                "thread and an independent PromptCycle; a bucket's next "
                "arrival never waits for its previous arrival to complete."
            ),
            "phase_offset_step_seconds": config.phase_offset_step_seconds,
            "client_concurrency_budget": config.client_concurrency_budget,
            "client_concurrency_budget_note": (
                "Explicit HTTP worker-thread budget for the client's request "
                "executor; must be set safely above any expected accumulated "
                "server-side backlog for the planned rates/durations. If the "
                "number of outstanding client requests ever reaches this "
                "budget, the client itself -- not vLLM -- becomes the "
                "limiting factor, and the point is invalidated "
                "(client_concurrency_budget_exceeded)."
            ),
        },
        "validity_thresholds": {
            "max_scheduling_lag_p95_seconds": config.max_scheduling_lag_p95_seconds,
            "max_achieved_rate_relative_error": config.max_achieved_rate_relative_error,
            "note": (
                "Conservative, explicit, operator-configurable thresholds "
                "distinguishing a client-side load-generator saturation "
                "failure from genuine server-side backlog. Server backlog/"
                "overload (a growing waiting/outstanding population) is "
                "expected evidence, NOT itself an invalidation condition."
            ),
        },
        "waiting_trend_window_fraction": config.waiting_trend_window_fraction,
        "base_url": config.base_url,
        "host": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "node": platform.node(),
        },
        "generated_at_wall_clock_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_mixed_experiment(config: MixedExperimentConfig) -> dict[str, Any]:
    """Run the full target-rho sequence for the configured bucket mix."""

    config.validate()
    records, dataset_sha256 = smoke.load_profiling_records(
        config.profiling_jsonl, config.model, config.tokenizer_revision
    )
    bucket_definitions = {name: pb.resolve_bucket(name) for name in config.capacities}
    bucket_cycles = {
        name: pb.PromptCycle(pb.select_bucket_records(records, name))
        for name in config.capacities
    }
    endpoint = smoke.completions_endpoint(config.base_url)
    # Duck-typed reuse of profile_bucket.py's precondition/identity probes:
    # MixedExperimentConfig exposes the same attribute names
    # (model/base_url/probe_transport/fingerprint_runner/timing.
    # request_timeout_seconds) these functions read, so no local
    # reimplementation is needed.
    idle_check = pb.build_idle_check(config)
    server_identity = pb.probe_server_identity(config)
    alphas = config.alphas

    vllm_metrics: list[dict[str, Any]] = []
    gpu_metrics: list[dict[str, Any]] = []
    telemetry_lock = threading.Lock()

    point_summaries: list[dict[str, Any]] = []
    request_results: list[dict[str, Any]] = []
    previous_point_drained = True

    for point_index, target_rho in enumerate(config.target_rho_values):
        bucket_targets = derive_bucket_targets(
            target_rho=target_rho,
            alphas=alphas,
            capacities=config.capacities,
            bucket_definitions=bucket_definitions,
        )

        telemetry_config: pb.TelemetryConfig | None = None
        if config.collect_telemetry:

            def make_sink(sink_list: list[dict[str, Any]]):
                def _sink(entry: dict[str, Any]) -> None:
                    tagged = dict(entry)
                    tagged["run_id"] = config.run_id
                    tagged["point_index"] = point_index
                    tagged["target_rho"] = target_rho
                    with telemetry_lock:
                        sink_list.append(tagged)

                return _sink

            telemetry_config = pb.TelemetryConfig(
                metrics_endpoint=pb.metrics_endpoint(config.base_url),
                metrics_transport=config.metrics_transport,
                gpu_sampler=config.gpu_sampler,
                interval_seconds=config.timing.metrics_interval_seconds,
                request_timeout_seconds=config.timing.request_timeout_seconds,
                on_vllm_sample=make_sink(vllm_metrics),
                on_gpu_sample=make_sink(gpu_metrics),
            )

        summary, results = run_mixed_load_point(
            target_rho=target_rho,
            point_index=point_index,
            bucket_cycles=bucket_cycles,
            bucket_targets=bucket_targets,
            endpoint=endpoint,
            model=config.model,
            timing=config.timing,
            transport=config.transport,
            client_concurrency_budget=config.client_concurrency_budget,
            max_scheduling_lag_p95_seconds=config.max_scheduling_lag_p95_seconds,
            max_achieved_rate_relative_error=config.max_achieved_rate_relative_error,
            waiting_trend_window_fraction=config.waiting_trend_window_fraction,
            phase_offset_step_seconds=config.phase_offset_step_seconds,
            clock=config.clock,
            idle_check=idle_check,
            telemetry_config=telemetry_config,
            run_id=config.run_id,
            previous_point_drained=previous_point_drained,
        )
        previous_point_drained = summary["drain_outcome"] == "drained"
        point_summaries.append(summary)
        request_results.extend(results)

    manifest = build_manifest(
        config, bucket_definitions, dataset_sha256, len(records), server_identity
    )

    review_table = [
        {
            "target_rho": summary["target_rho"],
            "achieved_rho": summary["rho_pred_achieved"],
            "total_target_req_s": sum(
                b["target_request_rate"] for b in summary["buckets"].values()
            ),
            "total_achieved_req_s": sum(
                b["achieved_request_rate"] for b in summary["buckets"].values()
            ),
            "engine_tok_s": summary["engine_token_throughput"].get("total_tokens_per_second"),
            "waiting_mean": summary["saturation_evidence"]["waiting"].get("mean"),
            "waiting_max": summary["saturation_evidence"]["waiting"].get("max"),
            "waiting_trend_req_s": summary["saturation_evidence"]["waiting"].get(
                "trend_slope_per_second"
            ),
            "waiting_first_median": summary["saturation_evidence"]["waiting"].get(
                "first_window_median"
            ),
            "waiting_last_median": summary["saturation_evidence"]["waiting"].get(
                "last_window_median"
            ),
            "outstanding_t0": summary["outstanding_at_t0"],
            "outstanding_t1": summary["outstanding_at_t1"],
            "outstanding_delta": summary["outstanding_delta"],
            "running_mean": summary["saturation_evidence"]["running"].get("mean"),
            "running_max": summary["saturation_evidence"]["running"].get("max"),
            "preemptions_delta": (
                summary["vllm_telemetry"]["num_preemptions_total"]["delta"]
                if summary["vllm_telemetry"].get("num_preemptions_total", {}).get(
                    "available"
                )
                else None
            ),
            "scheduling_lag_p95": summary["scheduling_lag_seconds"].get("p95"),
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
            "V_M inputs and rho values are monolithic non-P/D total-token "
            "(L_in+L_out) throughput evidence. NOT physical KV-release "
            "throughput, NOT isolated decoder V_D, and NOT SLO-safe "
            "operating capacity."
        ),
        "human_review_warning": (
            "This harness does NOT decide whether the composition model "
            "held. rho_pred_target, rho_pred_achieved, waiting/running/"
            "outstanding trends, and run_valid are all raw evidence for "
            "HUMAN REVIEW. rho_pred materially below 1 suggests a stable/"
            "non-growing backlog is expected; rho_pred near 1 suggests "
            "onset/boundary behavior is expected; rho_pred above 1 suggests "
            "persistent accumulation is expected -- but confirming that "
            "expectation against the recorded evidence remains human "
            "analysis."
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
# Artifact writing (reuses profile_bucket.py's atomic writer unmodified)
# ---------------------------------------------------------------------------


def write_artifacts(output_dir: Path, bundle: Mapping[str, Any]) -> None:
    pb.write_artifacts(output_dir, bundle)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_named_float(value: str) -> tuple[str, float]:
    try:
        name, raw_number = value.split(":", maxsplit=1)
        number = float(raw_number)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected NAME:NUMBER, got {value!r}"
        ) from error
    if not name:
        raise argparse.ArgumentTypeError(f"expected NAME:NUMBER, got {value!r}")
    return name, number


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiling-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--vllm-version", default="0.28.0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--generation-config", default="vllm")
    parser.add_argument("--prefix-caching", action="store_true")
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=pb.DEFAULT_GPU_MEMORY_UTILIZATION
    )
    parser.add_argument(
        "--capacity",
        action="append",
        type=_parse_named_float,
        required=True,
        metavar="BUCKET:V_M",
        help=(
            "operator-supplied V_M(bucket) logical-token/s capacity input, "
            "e.g. --capacity input-heavy:2180; repeat once per bucket. "
            "Never hard-coded by this harness."
        ),
    )
    parser.add_argument(
        "--weight",
        action="append",
        type=_parse_named_float,
        metavar="BUCKET:WEIGHT",
        help=(
            "raw (unnormalized) composition weight alpha input for a "
            "bucket, e.g. --weight input-heavy:1; repeat once per bucket. "
            "Defaults to equal weight across every --capacity bucket if "
            "omitted."
        ),
    )
    parser.add_argument(
        "--target-rho",
        action="append",
        type=float,
        metavar="RHO",
        help=(
            "target predicted-utilization ratio; repeatable. Defaults to "
            f"{list(DEFAULT_TARGET_RHO_VALUES)} if omitted."
        ),
    )
    parser.add_argument("--settling-seconds", type=float, default=DEFAULT_SETTLING_SECONDS)
    parser.add_argument(
        "--measurement-seconds", type=float, default=DEFAULT_MEASUREMENT_SECONDS
    )
    parser.add_argument(
        "--drain-timeout-seconds", type=float, default=DEFAULT_DRAIN_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--request-timeout-seconds", type=float, default=DEFAULT_REQUEST_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--metrics-interval-seconds", type=float, default=DEFAULT_METRICS_INTERVAL_SECONDS
    )
    parser.add_argument(
        "--client-concurrency-budget", type=int, default=DEFAULT_CLIENT_CONCURRENCY_BUDGET
    )
    parser.add_argument(
        "--max-scheduling-lag-p95-seconds",
        type=float,
        default=DEFAULT_MAX_SCHEDULING_LAG_P95_SECONDS,
    )
    parser.add_argument(
        "--max-achieved-rate-relative-error",
        type=float,
        default=DEFAULT_MAX_ACHIEVED_RATE_RELATIVE_ERROR,
    )
    parser.add_argument(
        "--waiting-trend-window-fraction",
        type=float,
        default=DEFAULT_WAITING_TREND_WINDOW_FRACTION,
    )
    parser.add_argument(
        "--phase-offset-step-seconds",
        type=float,
        default=DEFAULT_PHASE_OFFSET_STEP_SECONDS,
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument("--no-telemetry", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> MixedExperimentConfig:
    capacities = dict(args.capacity)
    raw_weights = dict(args.weight) if args.weight else None
    target_rho_values = tuple(args.target_rho) if args.target_rho else DEFAULT_TARGET_RHO_VALUES
    return MixedExperimentConfig(
        profiling_jsonl=args.profiling_jsonl,
        output_dir=args.output_dir,
        base_url=args.base_url,
        model=args.model,
        tokenizer_revision=args.tokenizer_revision,
        capacities=capacities,
        raw_weights=raw_weights,
        target_rho_values=target_rho_values,
        vllm_version=args.vllm_version,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        generation_config=args.generation_config,
        prefix_caching=args.prefix_caching,
        gpu_memory_utilization=args.gpu_memory_utilization,
        timing=MixedTimingConfig(
            settling_seconds=args.settling_seconds,
            measurement_seconds=args.measurement_seconds,
            drain_timeout_seconds=args.drain_timeout_seconds,
            request_timeout_seconds=args.request_timeout_seconds,
            metrics_interval_seconds=args.metrics_interval_seconds,
        ),
        client_concurrency_budget=args.client_concurrency_budget,
        max_scheduling_lag_p95_seconds=args.max_scheduling_lag_p95_seconds,
        max_achieved_rate_relative_error=args.max_achieved_rate_relative_error,
        waiting_trend_window_fraction=args.waiting_trend_window_fraction,
        phase_offset_step_seconds=args.phase_offset_step_seconds,
        run_id=args.run_id or default_run_id(),
        collect_telemetry=not args.no_telemetry,
    )


def run_cli(config: MixedExperimentConfig) -> int:
    try:
        bundle = run_mixed_experiment(config)
        write_artifacts(config.output_dir, bundle)
    except (ProfilingError, smoke.SmokeError, OSError) as error:
        print(f"mixed-workload validation failed: {error}", file=sys.stderr)
        return 1

    print(f"run_id: {config.run_id}")
    print(f"artifacts written to: {config.output_dir}")
    for row in bundle["summary"]["review_table"]:
        engine_tok_s = row["engine_tok_s"]
        print(
            f"  target_rho={row['target_rho']:.3f} "
            f"achieved_rho={row['achieved_rho']:.3f} "
            f"engine_tok/s={engine_tok_s if engine_tok_s is None else f'{engine_tok_s:.2f}'} "
            f"waiting(mean/max)={row['waiting_mean']}/{row['waiting_max']} "
            f"outstanding(t0/t1)={row['outstanding_t0']}/{row['outstanding_t1']} "
            f"valid={row['run_valid']} reasons={row['invalidation_reasons']}"
        )
    if bundle["summary"]["any_point_invalid"]:
        print(
            "WARNING: at least one point is invalid; see invalidation_reasons "
            "before drawing any composition-model conclusion.",
            file=sys.stderr,
        )
    print(
        "NOTE: whether the composition model held is a HUMAN REVIEW "
        "decision; this tool does not decide it automatically."
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    config = config_from_args(args)
    return run_cli(config)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Thin backward-compatibility wrapper around the generic bucket profiler.

The balanced-bucket fixed-concurrency profiling methodology implemented here
has been VALIDATED on real Colab Tesla T4 hardware (see the #1546 evidence
record: two independent 180s confirmation runs reproduced C=48 and C=64
throughput, ~1228.8 and ~1274.3 logical token/s respectively). The
measurement logic itself now lives in ``profile_bucket.py`` so the exact
same, already-validated methodology can also be applied to the
``input-heavy`` and ``output-heavy`` buckets without duplicating any
profiling logic.

This module exists only so that pre-existing commands, scripts, and
documentation that invoke ``profile_balanced_bucket.py`` directly keep
working unchanged; it always profiles the ``balanced`` bucket and does not
expose ``--bucket`` on its CLI. New usage, and profiling of the other
approved buckets, should prefer::

    python profile_bucket.py --bucket balanced ...
    python profile_bucket.py --bucket input-heavy ...
    python profile_bucket.py --bucket output-heavy ...

Every name re-exported below is the identical object from
``profile_bucket``; there is no duplicated profiling implementation here.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import profile_bucket as _bucket_profiler
from profile_bucket import (
    BUCKETS_BY_NAME,
    BUCKET_VALIDATION_STATUS,
    DEFAULT_CONCURRENCY_LADDER,
    DEFAULT_DRAIN_TIMEOUT_SECONDS,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_MEASUREMENT_SECONDS,
    DEFAULT_METRICS_INTERVAL_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_SETTLING_SECONDS,
    EXPERIMENT_SUMMARY_SCHEMA_VERSION,
    GPU_METRICS_FILENAME,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    POINT_SUMMARIES_FILENAME,
    POINT_SUMMARY_SCHEMA_VERSION,
    REQUEST_RESULT_SCHEMA_VERSION,
    REQUEST_RESULTS_FILENAME,
    SUMMARY_FILENAME,
    VLLM_METRICS_FILENAME,
    ExperimentConfig,
    ProfilingError,
    PromptCycle,
    TelemetryConfig,
    TelemetrySampler,
    TimingConfig,
    Transport,
    build_idle_check,
    build_manifest,
    compute_adjacent_gain,
    execute_profiling_request,
    gpu_fingerprint,
    http_get,
    metrics_endpoint,
    models_endpoint,
    parse_prometheus_text,
    probe_server_identity,
    resolve_bucket,
    run_cli,
    run_experiment,
    run_load_point,
    sample_gpu_telemetry,
    select_known_metrics,
    summarize_gpu_telemetry_window,
    summarize_point,
    summarize_vllm_telemetry_window,
    version_endpoint,
    write_artifacts,
)

__all__ = [
    "BUCKETS_BY_NAME",
    "BUCKET_VALIDATION_STATUS",
    "DEFAULT_CONCURRENCY_LADDER",
    "DEFAULT_DRAIN_TIMEOUT_SECONDS",
    "DEFAULT_GPU_MEMORY_UTILIZATION",
    "DEFAULT_MEASUREMENT_SECONDS",
    "DEFAULT_METRICS_INTERVAL_SECONDS",
    "DEFAULT_REQUEST_TIMEOUT_SECONDS",
    "DEFAULT_SETTLING_SECONDS",
    "EXPERIMENT_SUMMARY_SCHEMA_VERSION",
    "GPU_METRICS_FILENAME",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "POINT_SUMMARIES_FILENAME",
    "POINT_SUMMARY_SCHEMA_VERSION",
    "REQUEST_RESULT_SCHEMA_VERSION",
    "REQUEST_RESULTS_FILENAME",
    "SUMMARY_FILENAME",
    "VLLM_METRICS_FILENAME",
    "ExperimentConfig",
    "ProfilingError",
    "PromptCycle",
    "TelemetryConfig",
    "TelemetrySampler",
    "TimingConfig",
    "Transport",
    "build_idle_check",
    "build_manifest",
    "compute_adjacent_gain",
    "execute_profiling_request",
    "gpu_fingerprint",
    "http_get",
    "metrics_endpoint",
    "models_endpoint",
    "parse_prometheus_text",
    "probe_server_identity",
    "resolve_bucket",
    "run_cli",
    "run_experiment",
    "run_load_point",
    "sample_gpu_telemetry",
    "select_known_metrics",
    "summarize_gpu_telemetry_window",
    "summarize_point",
    "summarize_vllm_telemetry_window",
    "version_endpoint",
    "write_artifacts",
    "BALANCED_BUCKET_NAME",
    "BALANCED_BUCKET",
    "select_balanced_records",
    "default_run_id",
    "build_argument_parser",
    "main",
]

BALANCED_BUCKET_NAME = "balanced"
BALANCED_BUCKET = BUCKETS_BY_NAME[BALANCED_BUCKET_NAME]


def select_balanced_records(
    records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Backward-compatible alias for ``select_bucket_records(records, "balanced")``."""

    return _bucket_profiler.select_bucket_records(records, BALANCED_BUCKET_NAME)


def default_run_id() -> str:
    return _bucket_profiler.default_run_id(BALANCED_BUCKET_NAME)


def build_argument_parser():
    """The same CLI surface as the original balanced-only profiler.

    ``--bucket`` is intentionally not exposed here: this wrapper always runs
    the already-validated balanced experiment. Use ``profile_bucket.py
    --bucket ...`` directly to select a different bucket.
    """

    import argparse
    from pathlib import Path

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
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help="operator-declared vLLM --gpu-memory-utilization used at launch",
    )
    parser.add_argument(
        "--concurrency",
        type=_bucket_profiler._parse_concurrency_ladder,
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
        bucket=BALANCED_BUCKET_NAME,
        vllm_version=args.vllm_version,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        generation_config=args.generation_config,
        prefix_caching=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
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
    return run_cli(config)


if __name__ == "__main__":
    raise SystemExit(main())

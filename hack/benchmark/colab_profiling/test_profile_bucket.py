"""CPU-only mocked tests for the #1546 generic fixed-concurrency profiler.

These tests validate the harness only (bucket selection, concurrency bounds,
window membership, invalidation, dynamic token-rate arithmetic, and derived
telemetry summaries). They do not and cannot answer any real research
question about input-heavy/output-heavy saturation; that requires a real
Colab/vLLM run, which has not been performed for those two buckets.
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_dataset as generator
import profile_bucket as profiler
import run_request_smoke as smoke


MODEL = "Qwen/Qwen2.5-3B"
REVISION = "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"

BUCKET_NAMES = ("balanced", "input-heavy", "output-heavy")


def make_records(records_per_bucket: int = 4) -> list[dict[str, Any]]:
    """Build validator-compliant profiling records for all three buckets."""

    records: list[dict[str, Any]] = []
    for bucket_index, bucket in enumerate(generator.DEFAULT_BUCKETS):
        for record_index in range(records_per_bucket):
            prompt_ids = [
                1000 + bucket_index * 10_000 + record_index * bucket.input_tokens + offset
                for offset in range(bucket.input_tokens)
            ]
            records.append(
                {
                    "bucket": bucket.name,
                    "generator_seed": 1546 + bucket_index,
                    "model_id": MODEL,
                    "prompt_hash": generator.prompt_hash(prompt_ids),
                    "prompt_token_count": bucket.input_tokens,
                    "prompt_token_ids": prompt_ids,
                    "request_id": f"profiling.{bucket.name}.{record_index:04d}",
                    "schema_version": generator.SCHEMA_VERSION,
                    "server_reported_completion_tokens": None,
                    "split": "profiling",
                    "target_output_tokens": bucket.target_output_tokens,
                    "tokenizer_revision": REVISION,
                    "total_target_tokens": bucket.total_target_tokens,
                }
            )
    return records


def write_dataset(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def successful_response(payload: Mapping[str, Any]) -> smoke.HttpResponse:
    prompt_ids = payload["prompt"]
    target = payload["max_tokens"]
    body = {
        "model": payload["model"],
        "choices": [
            {
                "index": 0,
                "text": "",
                "finish_reason": "length",
                "prompt_token_ids": prompt_ids,
                "token_ids": list(range(target)),
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": target,
            "total_tokens": len(prompt_ids) + target,
        },
    }
    return smoke.HttpResponse(200, json.dumps(body).encode("utf-8"))


class DelayedTransport:
    """A deterministic fake transport with a configurable per-request delay.

    Independently tracks concurrent in-flight calls so tests can verify the
    profiler's own concurrency accounting against ground truth measured
    inside the transport itself.
    """

    def __init__(self, delay_seconds: float = 0.0) -> None:
        self.delay_seconds = delay_seconds
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.call_count = 0

    def __call__(
        self, endpoint: str, payload: Mapping[str, Any], timeout: float
    ) -> smoke.HttpResponse:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.call_count += 1
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        with self._lock:
            self.active -= 1
        return successful_response(payload)


class FailNthTransport(DelayedTransport):
    def __init__(self, fail_on_call: int, delay_seconds: float = 0.0) -> None:
        super().__init__(delay_seconds)
        self.fail_on_call = fail_on_call

    def __call__(
        self, endpoint: str, payload: Mapping[str, Any], timeout: float
    ) -> smoke.HttpResponse:
        with self._lock:
            self.call_count += 1
            call_number = self.call_count
        if call_number == self.fail_on_call:
            return smoke.HttpResponse(503, b"")
        return super().__call__(endpoint, payload, timeout)


def always_ok_idle_check() -> tuple[bool, None]:
    return True, None


# ---------------------------------------------------------------------------
# Bucket selection / genericity
# ---------------------------------------------------------------------------


class BucketSelectionTests(unittest.TestCase):
    def test_each_approved_bucket_selects_only_its_own_records(self) -> None:
        records = make_records()
        for bucket_name in BUCKET_NAMES:
            selected = profiler.select_bucket_records(records, bucket_name)
            self.assertTrue(selected)
            self.assertTrue(all(r["bucket"] == bucket_name for r in selected))
            self.assertEqual(
                len(selected), sum(1 for r in records if r["bucket"] == bucket_name)
            )

    def test_no_cross_bucket_records_in_selection(self) -> None:
        records = make_records()
        for bucket_name in BUCKET_NAMES:
            selected = profiler.select_bucket_records(records, bucket_name)
            other_names = {name for name in BUCKET_NAMES if name != bucket_name}
            self.assertFalse(any(r["bucket"] in other_names for r in selected))

    def test_unknown_bucket_is_rejected(self) -> None:
        records = make_records()
        with self.assertRaises(profiler.ProfilingError):
            profiler.select_bucket_records(records, "does-not-exist")
        with self.assertRaises(profiler.ProfilingError):
            profiler.resolve_bucket("does-not-exist")

    def test_missing_records_for_an_approved_bucket_is_rejected(self) -> None:
        records = [r for r in make_records() if r["bucket"] != "input-heavy"]
        with self.assertRaises(profiler.ProfilingError):
            profiler.select_bucket_records(records, "input-heavy")

    def test_bucket_definitions_come_from_the_dataset_module(self) -> None:
        # This is a genericity guard: profile_bucket must not hard-code
        # independent bucket geometry that could drift from
        # generate_dataset.DEFAULT_BUCKETS.
        self.assertEqual(
            profiler.BUCKETS_BY_NAME,
            {bucket.name: bucket for bucket in generator.DEFAULT_BUCKETS},
        )


class CliRejectsUnknownBucketTests(unittest.TestCase):
    def test_argparse_rejects_an_unknown_bucket_choice(self) -> None:
        parser = profiler.build_argument_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--profiling-jsonl",
                    "x.jsonl",
                    "--output-dir",
                    "out",
                    "--bucket",
                    "not-a-real-bucket",
                    "--model",
                    MODEL,
                    "--tokenizer-revision",
                    REVISION,
                ]
            )


# ---------------------------------------------------------------------------
# Dynamic per-bucket request contract
# ---------------------------------------------------------------------------


class DynamicRequestContractTests(unittest.TestCase):
    def test_prompt_and_completion_token_contract_follow_the_selected_bucket(
        self,
    ) -> None:
        all_records = make_records(1)
        for bucket in generator.DEFAULT_BUCKETS:
            record = next(r for r in all_records if r["bucket"] == bucket.name)
            transport = DelayedTransport()
            result = profiler.execute_profiling_request(
                record,
                "http://x/v1/completions",
                MODEL,
                10,
                transport,
                time.monotonic,
            )
            self.assertTrue(result["passed"], msg=f"bucket={bucket.name}")
            self.assertEqual(result["expected_prompt_tokens"], bucket.input_tokens)
            self.assertEqual(
                result["target_completion_tokens"], bucket.target_output_tokens
            )
            self.assertEqual(result["observed_prompt_tokens"], bucket.input_tokens)
            self.assertEqual(
                result["observed_completion_tokens"], bucket.target_output_tokens
            )
            self.assertEqual(
                result["observed_total_tokens"], bucket.total_target_tokens
            )

    def test_wrong_bucket_completion_length_fails_contract(self) -> None:
        record = next(
            r for r in make_records(1) if r["bucket"] == "output-heavy"
        )

        def wrong_length_transport(endpoint, payload, timeout):
            response = successful_response(payload)
            body = json.loads(response.body)
            body["usage"]["completion_tokens"] -= 1
            return smoke.HttpResponse(200, json.dumps(body).encode("utf-8"))

        result = profiler.execute_profiling_request(
            record, "http://x/v1/completions", MODEL, 10, wrong_length_transport, time.monotonic
        )
        self.assertFalse(result["passed"])
        self.assertIn(
            "completion_token_count",
            {f["reason"] for f in result["failure_reasons"]},
        )


# ---------------------------------------------------------------------------
# Dynamic token-rate invariant (per-bucket, not hard-coded 512)
# ---------------------------------------------------------------------------


class SummarizePointGenericityTests(unittest.TestCase):
    def _in_window_pass(self, prompt_tokens: int, completion_tokens: int):
        return {
            "passed": True,
            "in_measurement_window": True,
            "observed_prompt_tokens": prompt_tokens,
            "observed_completion_tokens": completion_tokens,
            "failure_reasons": [],
        }

    def test_invariant_uses_each_current_buckets_total_target_tokens(self) -> None:
        for bucket in generator.DEFAULT_BUCKETS:
            results = [
                self._in_window_pass(bucket.input_tokens, bucket.target_output_tokens)
                for _ in range(10)
            ]
            summary = profiler.summarize_point(
                bucket=bucket,
                concurrency=2,
                results=results,
                max_observed_concurrency=2,
                t_start=0.0,
                t0=1.0,
                t1=2.0,
                measurement_seconds=1.0,
                submitted_count=10,
                outstanding_at_t1=0,
                outstanding_after_drain=0,
                drain_duration=0.0,
                drained=True,
            )
            self.assertEqual(summary["bucket"], bucket.name)
            self.assertAlmostEqual(
                summary["completed_total_tokens_per_second"],
                bucket.total_target_tokens * summary["completed_requests_per_second"],
            )
            self.assertTrue(summary["token_rate_invariant"]["holds"])
            self.assertEqual(
                summary["token_rate_invariant"]["expected_total_tokens_per_request"],
                bucket.total_target_tokens,
            )

    def test_invariant_is_not_hard_coded_to_512(self) -> None:
        # A synthetic, non-approved bucket with a different total-work
        # figure proves the arithmetic is genuinely derived from
        # bucket.total_target_tokens rather than a hard-coded constant.
        custom_bucket = generator.Bucket(
            name="custom-test-only", input_tokens=100, target_output_tokens=50
        )
        self.assertEqual(custom_bucket.total_target_tokens, 150)
        results = [self._in_window_pass(100, 50) for _ in range(4)]
        summary = profiler.summarize_point(
            bucket=custom_bucket,
            concurrency=1,
            results=results,
            max_observed_concurrency=1,
            t_start=0.0,
            t0=0.0,
            t1=2.0,
            measurement_seconds=2.0,
            submitted_count=4,
            outstanding_at_t1=0,
            outstanding_after_drain=0,
            drain_duration=0.0,
            drained=True,
        )
        self.assertAlmostEqual(summary["completed_requests_per_second"], 2.0)
        self.assertAlmostEqual(summary["completed_total_tokens_per_second"], 300.0)
        self.assertTrue(summary["token_rate_invariant"]["holds"])

    def test_generic_point_summary_contains_selected_bucket_identity(self) -> None:
        bucket = generator.DEFAULT_BUCKETS[0]
        summary = profiler.summarize_point(
            bucket=bucket,
            concurrency=1,
            results=[],
            max_observed_concurrency=1,
            t_start=0.0,
            t0=0.0,
            t1=1.0,
            measurement_seconds=1.0,
            submitted_count=0,
            outstanding_at_t1=0,
            outstanding_after_drain=0,
            drain_duration=0.0,
            drained=True,
        )
        self.assertEqual(summary["bucket"], bucket.name)


# ---------------------------------------------------------------------------
# vLLM measurement-window telemetry summaries
# ---------------------------------------------------------------------------


def vllm_sample(
    timestamp: float,
    running: float | None = None,
    waiting: float | None = None,
    kv: float | None = None,
    preemptions: float | None = None,
    status: str = "ok",
) -> dict[str, Any]:
    known_metrics: dict[str, Any] = {}

    def entry(value: float | None) -> dict[str, Any]:
        if value is None:
            return {"present": False, "samples": []}
        return {"present": True, "samples": [{"name": "x", "labels": {}, "value": value}]}

    known_metrics["vllm:num_requests_running"] = entry(running)
    known_metrics["vllm:num_requests_waiting"] = entry(waiting)
    known_metrics["vllm:kv_cache_usage_perc"] = entry(kv)
    known_metrics["vllm:num_preemptions_total"] = entry(preemptions)
    return {
        "timestamp": timestamp,
        "status": status,
        "known_metrics": known_metrics,
    }


class VllmTelemetrySummaryTests(unittest.TestCase):
    def test_running_waiting_kv_avg_and_max(self) -> None:
        samples = [
            vllm_sample(1.0, running=2, waiting=0, kv=0.1),
            vllm_sample(1.5, running=4, waiting=1, kv=0.3),
            vllm_sample(1.9, running=6, waiting=2, kv=0.5),
        ]
        summary = profiler.summarize_vllm_telemetry_window(samples, t0=1.0, t1=2.0)
        self.assertTrue(summary["available"])
        self.assertEqual(summary["measurement_sample_count"], 3)
        self.assertAlmostEqual(summary["num_requests_running"]["avg"], 4.0)
        self.assertEqual(summary["num_requests_running"]["max"], 6)
        self.assertAlmostEqual(summary["num_requests_waiting"]["avg"], 1.0)
        self.assertEqual(summary["num_requests_waiting"]["max"], 2)
        self.assertAlmostEqual(summary["kv_cache_usage_perc"]["avg"], 0.3)
        self.assertEqual(summary["kv_cache_usage_perc"]["max"], 0.5)

    def test_preemption_start_end_delta(self) -> None:
        samples = [
            vllm_sample(1.0, preemptions=5),
            vllm_sample(1.5, preemptions=7),
            vllm_sample(1.9, preemptions=9),
        ]
        summary = profiler.summarize_vllm_telemetry_window(samples, t0=1.0, t1=2.0)
        preemption = summary["num_preemptions_total"]
        self.assertTrue(preemption["available"])
        self.assertEqual(preemption["start"], 5)
        self.assertEqual(preemption["end"], 9)
        self.assertEqual(preemption["delta"], 4)

    def test_kv_cache_metric_name_fallback(self) -> None:
        # Older/newer vLLM releases may expose gpu_cache_usage_perc instead
        # of kv_cache_usage_perc; the summary must still find it.
        sample = vllm_sample(1.0)
        sample["known_metrics"]["vllm:gpu_cache_usage_perc"] = {
            "present": True,
            "samples": [{"name": "x", "labels": {}, "value": 0.42}],
        }
        summary = profiler.summarize_vllm_telemetry_window([sample], t0=1.0, t1=2.0)
        self.assertTrue(summary["kv_cache_usage_perc"]["available"])
        self.assertAlmostEqual(summary["kv_cache_usage_perc"]["avg"], 0.42)

    def test_missing_metric_is_explicit_not_zero(self) -> None:
        samples = [vllm_sample(1.0, running=3)]  # waiting/kv/preemptions absent
        summary = profiler.summarize_vllm_telemetry_window(samples, t0=1.0, t1=2.0)
        self.assertTrue(summary["num_requests_running"]["available"])
        self.assertEqual(summary["num_requests_waiting"], {"available": False})
        self.assertEqual(summary["kv_cache_usage_perc"], {"available": False})
        self.assertEqual(summary["num_preemptions_total"], {"available": False})
        # Explicitly not zero:
        self.assertNotEqual(summary["num_requests_waiting"].get("avg"), 0)

    def test_non_ok_samples_are_excluded(self) -> None:
        samples = [
            vllm_sample(1.0, running=100, status="transport_error"),
            vllm_sample(1.5, running=3, status="ok"),
        ]
        summary = profiler.summarize_vllm_telemetry_window(samples, t0=1.0, t1=2.0)
        self.assertEqual(summary["measurement_sample_count"], 1)
        self.assertAlmostEqual(summary["num_requests_running"]["avg"], 3.0)

    def test_samples_outside_window_do_not_leak_in(self) -> None:
        # Simulates telemetry accumulated across TWO different points/
        # concurrencies in one combined list; only the [t0, t1) slice for
        # THIS point's window may contribute to this point's summary.
        other_point_samples = [vllm_sample(t, running=999) for t in (0.0, 0.5, 0.8)]
        this_point_samples = [vllm_sample(t, running=5) for t in (1.0, 1.4, 1.9)]
        later_point_samples = [vllm_sample(t, running=1) for t in (2.0, 2.5)]
        combined = other_point_samples + this_point_samples + later_point_samples

        summary = profiler.summarize_vllm_telemetry_window(combined, t0=1.0, t1=2.0)
        self.assertEqual(summary["measurement_sample_count"], 3)
        self.assertAlmostEqual(summary["num_requests_running"]["avg"], 5.0)
        self.assertEqual(summary["num_requests_running"]["max"], 5)

    def test_no_samples_in_window_is_explicit(self) -> None:
        summary = profiler.summarize_vllm_telemetry_window([], t0=1.0, t1=2.0)
        self.assertFalse(summary["available"])
        self.assertEqual(summary["measurement_sample_count"], 0)


# ---------------------------------------------------------------------------
# GPU measurement-window telemetry summaries
# ---------------------------------------------------------------------------


def gpu_sample(
    timestamp: float,
    utilization: str = "10",
    memory_used: str = "512",
    temperature: str = "45",
    power: str = "55.0",
    throttle: str = "0x0000000000000000",
    available: bool = True,
) -> dict[str, Any]:
    if not available:
        return {"timestamp": timestamp, "available": False, "error": "no gpu"}
    return {
        "timestamp": timestamp,
        "available": True,
        "gpus": [
            {
                "utilization.gpu": utilization,
                "memory.used": memory_used,
                "memory.total": "16384",
                "temperature.gpu": temperature,
                "power.draw": power,
                "clocks.sm": "1500",
                "clocks_throttle_reasons.active": throttle,
            }
        ],
    }


class GpuTelemetrySummaryTests(unittest.TestCase):
    def test_utilization_memory_temperature_power_avg_and_max(self) -> None:
        samples = [
            gpu_sample(1.0, utilization="10", memory_used="500", temperature="40", power="50"),
            gpu_sample(1.5, utilization="20", memory_used="600", temperature="45", power="60"),
            gpu_sample(1.9, utilization="30", memory_used="700", temperature="50", power="70"),
        ]
        summary = profiler.summarize_gpu_telemetry_window(samples, t0=1.0, t1=2.0)
        self.assertTrue(summary["available"])
        self.assertAlmostEqual(summary["utilization.gpu"]["avg"], 20.0)
        self.assertEqual(summary["utilization.gpu"]["max"], 30.0)
        self.assertAlmostEqual(summary["memory.used"]["avg"], 600.0)
        self.assertAlmostEqual(summary["temperature.gpu"]["avg"], 45.0)
        self.assertAlmostEqual(summary["power.draw"]["avg"], 60.0)

    def test_throttle_nonzero_count_and_distinct_values(self) -> None:
        samples = [
            gpu_sample(1.0, throttle="0x0000000000000000"),
            gpu_sample(1.2, throttle="0x0000000000000004"),
            gpu_sample(1.4, throttle="0x0000000000000004"),
            gpu_sample(1.6, throttle="0x0000000000000010"),
            gpu_sample(1.8, throttle="0x0000000000000000"),
        ]
        summary = profiler.summarize_gpu_telemetry_window(samples, t0=1.0, t1=2.0)
        throttle = summary["clocks_throttle_reasons.active"]
        self.assertEqual(throttle["total_sample_count"], 5)
        self.assertEqual(throttle["nonzero_sample_count"], 3)
        self.assertEqual(
            throttle["distinct_nonzero_values"],
            ["0x0000000000000004", "0x0000000000000010"],
        )

    def test_all_zero_throttle_reports_zero_nonzero_count(self) -> None:
        samples = [gpu_sample(1.0), gpu_sample(1.5)]
        summary = profiler.summarize_gpu_telemetry_window(samples, t0=1.0, t1=2.0)
        throttle = summary["clocks_throttle_reasons.active"]
        self.assertEqual(throttle["nonzero_sample_count"], 0)
        self.assertEqual(throttle["distinct_nonzero_values"], [])

    def test_unavailable_gpu_samples_are_explicit(self) -> None:
        samples = [gpu_sample(1.0, available=False), gpu_sample(1.5, available=False)]
        summary = profiler.summarize_gpu_telemetry_window(samples, t0=1.0, t1=2.0)
        self.assertFalse(summary["available"])
        self.assertEqual(summary["measurement_sample_count"], 2)

    def test_samples_outside_window_do_not_leak_in(self) -> None:
        earlier = [gpu_sample(t, utilization="99") for t in (0.0, 0.5)]
        current = [gpu_sample(t, utilization="10") for t in (1.0, 1.5)]
        later = [gpu_sample(t, utilization="1") for t in (2.0,)]
        summary = profiler.summarize_gpu_telemetry_window(
            earlier + current + later, t0=1.0, t1=2.0
        )
        self.assertEqual(summary["measurement_sample_count"], 2)
        self.assertAlmostEqual(summary["utilization.gpu"]["avg"], 10.0)


# ---------------------------------------------------------------------------
# Existing concurrency/admission/window/drain behavior (still passing)
# ---------------------------------------------------------------------------


class RunLoadPointTests(unittest.TestCase):
    """These tests exercise the real threaded closed-loop scheduler."""

    def setUp(self) -> None:
        all_records = make_records(64)
        self.bucket = generator.DEFAULT_BUCKETS[1]  # "balanced"
        self.assertEqual(self.bucket.name, "balanced")
        self.records = profiler.select_bucket_records(all_records, self.bucket.name)
        self.cycle = profiler.PromptCycle(self.records)
        self.endpoint = "http://127.0.0.1:8000/v1/completions"

    def test_concurrency_never_exceeds_target(self) -> None:
        transport = DelayedTransport(delay_seconds=0.02)
        timing = profiler.TimingConfig(
            settling_seconds=0.05,
            measurement_seconds=0.15,
            drain_timeout_seconds=2.0,
            metrics_interval_seconds=1.0,
            request_timeout_seconds=5.0,
        )
        summary, results = profiler.run_load_point(
            concurrency=4,
            cycle=self.cycle,
            endpoint=self.endpoint,
            model=MODEL,
            bucket=self.bucket,
            timing=timing,
            transport=transport,
            idle_check=always_ok_idle_check,
            run_id="test-run",
        )
        self.assertLessEqual(transport.max_active, 4)
        self.assertLessEqual(summary["max_observed_concurrency"], 4)
        self.assertEqual(summary["max_observed_concurrency"], transport.max_active)
        self.assertGreater(summary["completed_requests_in_window"], 0)
        self.assertTrue(summary["token_rate_invariant"]["holds"])
        self.assertEqual(summary["bucket"], "balanced")
        self.assertTrue(all(result["concurrency"] == 4 for result in results))
        self.assertTrue(all(result["run_id"] == "test-run" for result in results))

    def test_no_admission_after_t1_and_drain_excludes_from_numerator(self) -> None:
        transport = DelayedTransport(delay_seconds=0.3)
        timing = profiler.TimingConfig(
            settling_seconds=0.02,
            measurement_seconds=0.03,
            drain_timeout_seconds=5.0,
            metrics_interval_seconds=1.0,
            request_timeout_seconds=5.0,
        )
        summary, results = profiler.run_load_point(
            concurrency=2,
            cycle=self.cycle,
            endpoint=self.endpoint,
            model=MODEL,
            bucket=self.bucket,
            timing=timing,
            transport=transport,
            idle_check=always_ok_idle_check,
            run_id="test-run",
        )
        self.assertEqual(summary["requests_submitted"], 2)
        self.assertEqual(len(results), 2)
        self.assertEqual(summary["completed_requests_in_window"], 0)
        self.assertEqual(summary["drain_outcome"], "drained")
        self.assertEqual(summary["outstanding_after_drain"], 0)
        self.assertTrue(all(result["phase"] == "drain" for result in results))
        self.assertTrue(all(not result["in_measurement_window"] for result in results))
        self.assertEqual(transport.call_count, 2)

    def test_bounded_drain_timeout_marks_point_invalid(self) -> None:
        transport = DelayedTransport(delay_seconds=0.3)
        timing = profiler.TimingConfig(
            settling_seconds=0.01,
            measurement_seconds=0.01,
            drain_timeout_seconds=0.02,
            metrics_interval_seconds=1.0,
            request_timeout_seconds=5.0,
        )
        summary, results = profiler.run_load_point(
            concurrency=1,
            cycle=self.cycle,
            endpoint=self.endpoint,
            model=MODEL,
            bucket=self.bucket,
            timing=timing,
            transport=transport,
            idle_check=always_ok_idle_check,
            run_id="test-run",
        )
        self.assertEqual(summary["drain_outcome"], "timed_out")
        self.assertIn("drain_timeout", summary["invalidation_reasons"])
        self.assertFalse(summary["run_valid"])

    def test_request_contract_failure_invalidates_the_point(self) -> None:
        transport = FailNthTransport(fail_on_call=1, delay_seconds=0.01)
        timing = profiler.TimingConfig(
            settling_seconds=0.02,
            measurement_seconds=0.1,
            drain_timeout_seconds=2.0,
            metrics_interval_seconds=1.0,
            request_timeout_seconds=5.0,
        )
        summary, results = profiler.run_load_point(
            concurrency=2,
            cycle=self.cycle,
            endpoint=self.endpoint,
            model=MODEL,
            bucket=self.bucket,
            timing=timing,
            transport=transport,
            idle_check=always_ok_idle_check,
            run_id="test-run",
        )
        self.assertFalse(summary["run_valid"])
        self.assertIn("request_failures_present", summary["invalidation_reasons"])
        self.assertTrue(any(not r["passed"] for r in results))

    def test_precondition_failure_invalidates_the_point(self) -> None:
        transport = DelayedTransport(delay_seconds=0.0)
        timing = profiler.TimingConfig(
            settling_seconds=0.01,
            measurement_seconds=0.02,
            drain_timeout_seconds=1.0,
            metrics_interval_seconds=1.0,
            request_timeout_seconds=5.0,
        )

        def failing_idle_check():
            return False, "model_identity_mismatch"

        summary, _ = profiler.run_load_point(
            concurrency=1,
            cycle=self.cycle,
            endpoint=self.endpoint,
            model=MODEL,
            bucket=self.bucket,
            timing=timing,
            transport=transport,
            idle_check=failing_idle_check,
            run_id="test-run",
        )
        self.assertFalse(summary["run_valid"])
        self.assertTrue(
            any(
                reason.startswith("precondition_failed")
                for reason in summary["invalidation_reasons"]
            )
        )

    def test_successful_precondition_still_executes_normally(self) -> None:
        # Guard against a fail-closed fix accidentally also short-circuiting
        # the healthy path.
        transport = DelayedTransport(delay_seconds=0.01)
        timing = profiler.TimingConfig(
            settling_seconds=0.02,
            measurement_seconds=0.05,
            drain_timeout_seconds=2.0,
            metrics_interval_seconds=1.0,
            request_timeout_seconds=5.0,
        )
        summary, results = profiler.run_load_point(
            concurrency=2,
            cycle=self.cycle,
            endpoint=self.endpoint,
            model=MODEL,
            bucket=self.bucket,
            timing=timing,
            transport=transport,
            idle_check=always_ok_idle_check,
            run_id="test-run",
        )
        self.assertFalse(summary["execution_skipped"])
        self.assertGreater(transport.call_count, 0)
        self.assertGreater(summary["requests_submitted"], 0)
        self.assertGreater(summary["max_observed_concurrency"], 0)
        self.assertGreater(len(results), 0)

    def test_measurement_window_membership_by_terminal_timestamp(self) -> None:
        transport = DelayedTransport(delay_seconds=0.01)
        timing = profiler.TimingConfig(
            settling_seconds=0.05,
            measurement_seconds=0.2,
            drain_timeout_seconds=2.0,
            metrics_interval_seconds=1.0,
            request_timeout_seconds=5.0,
        )
        summary, results = profiler.run_load_point(
            concurrency=3,
            cycle=self.cycle,
            endpoint=self.endpoint,
            model=MODEL,
            bucket=self.bucket,
            timing=timing,
            transport=transport,
            idle_check=always_ok_idle_check,
            run_id="test-run",
        )
        t0 = summary["measurement_t0_s"]
        t1 = summary["measurement_t1_s"]
        settling_results = [r for r in results if r["phase"] == "settling"]
        measurement_results = [r for r in results if r["phase"] == "measurement"]
        self.assertTrue(settling_results, "expected at least one settling completion")
        self.assertTrue(measurement_results, "expected at least one measurement completion")
        for result in settling_results:
            self.assertLess(result["terminal_monotonic_s"], t0)
        for result in measurement_results:
            self.assertGreaterEqual(result["terminal_monotonic_s"], t0)
            self.assertLess(result["terminal_monotonic_s"], t1)
        self.assertEqual(
            summary["completed_requests_in_window"], len(measurement_results)
        )

    def test_run_load_point_attaches_real_telemetry_summaries(self) -> None:
        transport = DelayedTransport(delay_seconds=0.01)
        timing = profiler.TimingConfig(
            settling_seconds=0.02,
            measurement_seconds=0.1,
            drain_timeout_seconds=2.0,
            metrics_interval_seconds=0.02,
            request_timeout_seconds=5.0,
        )

        def metrics_transport(endpoint, timeout):
            body = (
                b"vllm:num_requests_running 2.0\n"
                b"vllm:num_requests_waiting 0.0\n"
                b"vllm:kv_cache_usage_perc 0.25\n"
                b"vllm:num_preemptions_total 0.0\n"
            )
            return smoke.HttpResponse(200, body)

        telemetry_config = profiler.TelemetryConfig(
            metrics_endpoint="http://x/metrics",
            metrics_transport=metrics_transport,
            gpu_sampler=lambda: {"available": False, "error": "no gpu in tests"},
            interval_seconds=0.02,
            request_timeout_seconds=5.0,
        )
        summary, _ = profiler.run_load_point(
            concurrency=1,
            cycle=self.cycle,
            endpoint=self.endpoint,
            model=MODEL,
            bucket=self.bucket,
            timing=timing,
            transport=transport,
            idle_check=always_ok_idle_check,
            telemetry_config=telemetry_config,
            run_id="test-run",
        )
        self.assertTrue(summary["vllm_telemetry"]["available"])
        self.assertTrue(summary["vllm_telemetry"]["num_requests_running"]["available"])
        self.assertFalse(summary["gpu_telemetry"]["available"])

    def test_no_telemetry_config_is_explicit_not_missing(self) -> None:
        transport = DelayedTransport(delay_seconds=0.0)
        timing = profiler.TimingConfig(
            settling_seconds=0.01,
            measurement_seconds=0.02,
            drain_timeout_seconds=1.0,
            metrics_interval_seconds=1.0,
            request_timeout_seconds=5.0,
        )
        summary, _ = profiler.run_load_point(
            concurrency=1,
            cycle=self.cycle,
            endpoint=self.endpoint,
            model=MODEL,
            bucket=self.bucket,
            timing=timing,
            transport=transport,
            idle_check=always_ok_idle_check,
            telemetry_config=None,
            run_id="test-run",
        )
        self.assertFalse(summary["vllm_telemetry"]["available"])
        self.assertFalse(summary["gpu_telemetry"]["available"])


# ---------------------------------------------------------------------------
# Fail-closed precondition behavior (real Colab defect regression coverage)
#
# A fresh-runtime input-heavy repeatability run once started a C=48 point
# while the server was briefly unreachable. The precondition probe correctly
# failed, but run_load_point() still entered the closed-loop executor and
# generated ~25,069 immediately-failing HTTP requests before the bounded
# window elapsed. A failed mandatory precondition must prevent ALL load
# generation for that point, not merely mark it invalid afterward.
# ---------------------------------------------------------------------------


def failing_idle_check_with_reason(reason: str) -> profiler.IdleCheck:
    def _idle_check() -> tuple[bool, str]:
        return False, reason

    return _idle_check


class FailedPreconditionFailsClosedTests(unittest.TestCase):
    def setUp(self) -> None:
        all_records = make_records(64)
        self.bucket = generator.DEFAULT_BUCKETS[1]  # "balanced"
        self.records = profiler.select_bucket_records(all_records, self.bucket.name)
        self.cycle = profiler.PromptCycle(self.records)
        self.endpoint = "http://127.0.0.1:8000/v1/completions"
        # Generous timing that would otherwise sustain many closed-loop
        # cycles if execution were (incorrectly) allowed to proceed.
        self.timing = profiler.TimingConfig(
            settling_seconds=0.05,
            measurement_seconds=0.2,
            drain_timeout_seconds=1.0,
            metrics_interval_seconds=0.02,
            request_timeout_seconds=5.0,
        )

    def _run(self, concurrency: int, transport, telemetry_config=None):
        return profiler.run_load_point(
            concurrency=concurrency,
            cycle=self.cycle,
            endpoint=self.endpoint,
            model=MODEL,
            bucket=self.bucket,
            timing=self.timing,
            transport=transport,
            idle_check=failing_idle_check_with_reason("Connection refused"),
            telemetry_config=telemetry_config,
            run_id="test-run",
        )

    def test_zero_transport_calls_on_failed_precondition(self) -> None:
        transport = DelayedTransport(delay_seconds=0.0)
        summary, results = self._run(concurrency=48, transport=transport)
        self.assertEqual(transport.call_count, 0)
        self.assertEqual(results, [])

    def test_requests_submitted_and_max_concurrency_are_zero(self) -> None:
        transport = DelayedTransport(delay_seconds=0.0)
        summary, _ = self._run(concurrency=48, transport=transport)
        self.assertEqual(summary["requests_submitted"], 0)
        self.assertEqual(summary["max_observed_concurrency"], 0)
        self.assertEqual(summary["completed_requests_in_window"], 0)
        self.assertEqual(summary["completed_requests_per_second"], 0)
        self.assertEqual(summary["completed_total_tokens_per_second"], 0)
        self.assertEqual(summary["outstanding_at_t1"], 0)
        self.assertEqual(summary["outstanding_after_drain"], 0)

    def test_point_is_invalid_with_concrete_reason_retained(self) -> None:
        transport = DelayedTransport(delay_seconds=0.0)
        summary, _ = self._run(concurrency=48, transport=transport)
        self.assertFalse(summary["run_valid"])
        self.assertIn(
            "precondition_failed:Connection refused", summary["invalidation_reasons"]
        )
        self.assertTrue(summary["execution_skipped"])
        self.assertEqual(
            summary["execution_skipped_reason"], "precondition_failed:Connection refused"
        )

    def test_no_settling_or_measurement_execution_occurs(self) -> None:
        # Independent proof, beyond the transport call counter: no
        # per-request result records were ever produced at all, for any
        # phase (settling/measurement/drain).
        transport = DelayedTransport(delay_seconds=0.0)
        _, results = self._run(concurrency=48, transport=transport)
        self.assertEqual(len(results), 0)

    def test_telemetry_is_not_started_when_precondition_fails(self) -> None:
        vllm_calls = {"count": 0}
        gpu_calls = {"count": 0}

        def counting_metrics_transport(endpoint, timeout):
            vllm_calls["count"] += 1
            return smoke.HttpResponse(200, b"vllm:num_requests_running 0\n")

        def counting_gpu_sampler():
            gpu_calls["count"] += 1
            return {"available": False, "error": "should never be called"}

        telemetry_config = profiler.TelemetryConfig(
            metrics_endpoint="http://x/metrics",
            metrics_transport=counting_metrics_transport,
            gpu_sampler=counting_gpu_sampler,
            interval_seconds=0.01,
            request_timeout_seconds=5.0,
        )
        transport = DelayedTransport(delay_seconds=0.0)
        summary, _ = self._run(
            concurrency=48, transport=transport, telemetry_config=telemetry_config
        )
        # Give any (incorrectly) started background sampler thread a chance
        # to have fired at least once before asserting it never did.
        time.sleep(0.1)
        self.assertEqual(vllm_calls["count"], 0)
        self.assertEqual(gpu_calls["count"], 0)
        self.assertFalse(summary["vllm_telemetry"]["available"])
        self.assertFalse(summary["gpu_telemetry"]["available"])
        self.assertEqual(
            summary["vllm_telemetry"]["reason"], "execution_skipped_precondition_failed"
        )
        self.assertEqual(
            summary["gpu_telemetry"]["reason"], "execution_skipped_precondition_failed"
        )

    def test_fails_closed_regardless_of_requested_concurrency(self) -> None:
        for concurrency in (1, 4, 48, 64):
            with self.subTest(concurrency=concurrency):
                transport = DelayedTransport(delay_seconds=0.0)
                summary, results = self._run(concurrency=concurrency, transport=transport)
                self.assertEqual(transport.call_count, 0)
                self.assertEqual(len(results), 0)
                self.assertEqual(summary["requests_submitted"], 0)
                self.assertEqual(summary["max_observed_concurrency"], 0)
                self.assertFalse(summary["run_valid"])

    def test_no_idle_check_configured_still_executes_normally(self) -> None:
        # Fail-closed behavior must only trigger when a precondition check
        # is actually configured and it actually fails.
        transport = DelayedTransport(delay_seconds=0.01)
        summary, results = profiler.run_load_point(
            concurrency=2,
            cycle=self.cycle,
            endpoint=self.endpoint,
            model=MODEL,
            bucket=self.bucket,
            timing=self.timing,
            transport=transport,
            idle_check=None,
            run_id="test-run",
        )
        self.assertFalse(summary["execution_skipped"])
        self.assertGreater(transport.call_count, 0)
        self.assertGreater(summary["requests_submitted"], 0)


# ---------------------------------------------------------------------------
# Manifest reproducibility fields
# ---------------------------------------------------------------------------


class ManifestTests(unittest.TestCase):
    def _build_config(self, **overrides: Any) -> profiler.ExperimentConfig:
        defaults = dict(
            profiling_jsonl=Path("unused.jsonl"),
            output_dir=Path("unused-output"),
            base_url="http://127.0.0.1:8000",
            model=MODEL,
            tokenizer_revision=REVISION,
            bucket="balanced",
        )
        defaults.update(overrides)
        return profiler.ExperimentConfig(**defaults)

    def test_gpu_memory_utilization_is_written_to_manifest(self) -> None:
        config = self._build_config(gpu_memory_utilization=0.90, run_id="r1")
        manifest = profiler.build_manifest(
            config,
            profiler.BUCKETS_BY_NAME["balanced"],
            dataset_sha256="sha256:deadbeef",
            bucket_record_count=64,
            server_identity={"vllm_version": None, "gpu_fingerprint": {"available": False}},
        )
        declared = manifest["vllm"]["operator_declared_runtime_configuration"]
        self.assertEqual(declared["gpu_memory_utilization"], 0.90)
        self.assertIn(
            "--gpu-memory-utilization 0.9",
            manifest["runtime_launch_assumptions"]["recommended_launch_flags"],
        )

    def test_prefix_caching_false_is_written_to_manifest(self) -> None:
        config = self._build_config(prefix_caching=False, run_id="r1")
        manifest = profiler.build_manifest(
            config,
            profiler.BUCKETS_BY_NAME["balanced"],
            dataset_sha256="sha256:deadbeef",
            bucket_record_count=64,
            server_identity={"vllm_version": None, "gpu_fingerprint": {"available": False}},
        )
        declared = manifest["vllm"]["operator_declared_runtime_configuration"]
        self.assertIs(declared["prefix_caching"], False)
        self.assertIn(
            "--no-enable-prefix-caching",
            manifest["runtime_launch_assumptions"]["recommended_launch_flags"],
        )

    def test_operator_declared_values_are_labeled_as_such(self) -> None:
        config = self._build_config(run_id="r1")
        manifest = profiler.build_manifest(
            config,
            profiler.BUCKETS_BY_NAME["balanced"],
            dataset_sha256="sha256:deadbeef",
            bucket_record_count=64,
            server_identity={"vllm_version": None, "gpu_fingerprint": {"available": False}},
        )
        note = manifest["vllm"]["operator_declared_runtime_configuration"]["note"]
        self.assertIn("Declared by the operator", note)

    def test_manifest_records_selected_bucket_and_validation_status(self) -> None:
        config = self._build_config(bucket="input-heavy", run_id="r1")
        manifest = profiler.build_manifest(
            config,
            profiler.BUCKETS_BY_NAME["input-heavy"],
            dataset_sha256="sha256:deadbeef",
            bucket_record_count=64,
            server_identity={"vllm_version": None, "gpu_fingerprint": {"available": False}},
        )
        self.assertEqual(manifest["dataset"]["bucket_definition"]["name"], "input-heavy")
        self.assertIn("NOT YET VALIDATED", manifest["bucket_validation_status"])

    def test_config_validation_rejects_bad_gpu_memory_utilization(self) -> None:
        config = self._build_config(gpu_memory_utilization=1.5)
        with self.assertRaises(profiler.ProfilingError):
            config.validate()
        config = self._build_config(gpu_memory_utilization=0.0)
        with self.assertRaises(profiler.ProfilingError):
            config.validate()

    def test_config_validation_rejects_unknown_bucket(self) -> None:
        config = self._build_config(bucket="does-not-exist")
        with self.assertRaises(profiler.ProfilingError):
            config.validate()

    def test_config_validation_rejects_non_increasing_ladder(self) -> None:
        config = self._build_config(concurrency_ladder=(2, 1))
        with self.assertRaises(profiler.ProfilingError):
            config.validate()


# ---------------------------------------------------------------------------
# End-to-end orchestration / artifact writing, per bucket
# ---------------------------------------------------------------------------


class RunExperimentIntegrationTests(unittest.TestCase):
    def _run_full_experiment(self, bucket_name: str, output_dir: Path, dataset: Path):
        transport = DelayedTransport(delay_seconds=0.01)

        def probe_transport(endpoint, timeout):
            if endpoint.endswith("/v1/models"):
                body = json.dumps({"data": [{"id": MODEL}]}).encode("utf-8")
                return smoke.HttpResponse(200, body)
            if endpoint.endswith("/version"):
                return smoke.HttpResponse(200, b'{"version": "0.28.0"}')
            return smoke.HttpResponse(404, b"")

        def metrics_transport(endpoint, timeout):
            return smoke.HttpResponse(
                200,
                b"vllm:num_requests_running 1\nvllm:num_preemptions_total 0\n",
            )

        def gpu_sampler():
            return {"available": False, "error": "no gpu in unit tests"}

        def fingerprint_runner(*args, **kwargs):
            raise FileNotFoundError("nvidia-smi not present in unit tests")

        config = profiler.ExperimentConfig(
            profiling_jsonl=dataset,
            output_dir=output_dir,
            base_url="http://127.0.0.1:8000",
            model=MODEL,
            tokenizer_revision=REVISION,
            bucket=bucket_name,
            prefix_caching=False,
            gpu_memory_utilization=0.90,
            concurrency_ladder=(1, 2),
            timing=profiler.TimingConfig(
                settling_seconds=0.02,
                measurement_seconds=0.05,
                drain_timeout_seconds=1.0,
                metrics_interval_seconds=0.02,
                request_timeout_seconds=5.0,
            ),
            run_id=f"integration-test-{bucket_name}",
            transport=transport,
            probe_transport=probe_transport,
            metrics_transport=metrics_transport,
            gpu_sampler=gpu_sampler,
            fingerprint_runner=fingerprint_runner,
        )

        bundle = profiler.run_experiment(config)
        profiler.write_artifacts(output_dir, bundle)
        return bundle

    def test_full_experiment_for_each_bucket_writes_expected_artifacts(self) -> None:
        for bucket_name in BUCKET_NAMES:
            with self.subTest(bucket=bucket_name):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    dataset = root / "profiling.jsonl"
                    output_dir = root / "profiling-results"
                    write_dataset(dataset, make_records(8))

                    self._run_full_experiment(bucket_name, output_dir, dataset)

                    self.assertTrue((output_dir / profiler.MANIFEST_FILENAME).is_file())
                    self.assertTrue(
                        (output_dir / profiler.REQUEST_RESULTS_FILENAME).is_file()
                    )
                    self.assertTrue(
                        (output_dir / profiler.POINT_SUMMARIES_FILENAME).is_file()
                    )
                    self.assertTrue((output_dir / profiler.SUMMARY_FILENAME).is_file())
                    self.assertTrue(
                        (output_dir / profiler.VLLM_METRICS_FILENAME).is_file()
                    )
                    self.assertTrue((output_dir / profiler.GPU_METRICS_FILENAME).is_file())

                    manifest = json.loads(
                        (output_dir / profiler.MANIFEST_FILENAME).read_text("utf-8")
                    )
                    self.assertEqual(
                        manifest["dataset"]["bucket_definition"]["name"], bucket_name
                    )
                    self.assertEqual(
                        manifest["vllm"]["operator_declared_runtime_configuration"][
                            "gpu_memory_utilization"
                        ],
                        0.90,
                    )
                    self.assertIs(
                        manifest["vllm"]["operator_declared_runtime_configuration"][
                            "prefix_caching"
                        ],
                        False,
                    )

                    point_summaries = [
                        json.loads(line)
                        for line in (output_dir / profiler.POINT_SUMMARIES_FILENAME)
                        .read_text("utf-8")
                        .splitlines()
                    ]
                    self.assertEqual(len(point_summaries), 2)
                    self.assertTrue(
                        all(p["bucket"] == bucket_name for p in point_summaries)
                    )
                    self.assertIsNone(point_summaries[0]["adjacent_throughput_gain"])

                    summary = json.loads(
                        (output_dir / profiler.SUMMARY_FILENAME).read_text("utf-8")
                    )
                    self.assertEqual(summary["bucket"], bucket_name)
                    self.assertEqual(len(summary["review_table"]), 2)
                    self.assertTrue(
                        all(row["bucket"] == bucket_name for row in summary["review_table"])
                    )
                    self.assertIn("plateau_acceptance_warning", summary)

                    request_results = [
                        json.loads(line)
                        for line in (output_dir / profiler.REQUEST_RESULTS_FILENAME)
                        .read_text("utf-8")
                        .splitlines()
                    ]
                    self.assertTrue(request_results)
                    self.assertTrue(
                        all(r["bucket"] == bucket_name for r in request_results)
                    )

    def test_output_directory_must_not_already_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            existing = root / "already-there"
            existing.mkdir()
            with self.assertRaises(profiler.ProfilingError):
                profiler.write_artifacts(
                    existing,
                    {
                        "manifest": {},
                        "request_results": [],
                        "point_summaries": [],
                        "summary": {},
                        "vllm_metrics": [],
                        "gpu_metrics": [],
                    },
                )


if __name__ == "__main__":
    unittest.main()

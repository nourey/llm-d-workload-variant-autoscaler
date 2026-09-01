"""CPU-only mocked tests for the #1546 balanced-bucket profiling harness.

These tests validate the harness only (concurrency bounds, window
membership, invalidation, token-rate arithmetic). They do not and cannot
answer the real research question; that requires a real Colab/vLLM run.
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
import profile_balanced_bucket as profiler
import run_request_smoke as smoke


MODEL = "Qwen/Qwen2.5-3B"
REVISION = "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"


def balanced_bucket() -> generator.Bucket:
    return next(b for b in generator.DEFAULT_BUCKETS if b.name == "balanced")


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


class SelectBalancedRecordsTests(unittest.TestCase):
    def test_only_balanced_records_are_selected(self) -> None:
        records = make_records()
        selected = profiler.select_balanced_records(records)
        self.assertTrue(selected)
        self.assertTrue(all(record["bucket"] == "balanced" for record in selected))
        self.assertEqual(
            len(selected), sum(1 for r in records if r["bucket"] == "balanced")
        )

    def test_no_balanced_records_raises(self) -> None:
        records = [r for r in make_records() if r["bucket"] != "balanced"]
        with self.assertRaises(profiler.ProfilingError):
            profiler.select_balanced_records(records)


class PromptCycleTests(unittest.TestCase):
    def test_cycle_is_deterministic_sorted_and_wraps(self) -> None:
        records = [
            {"request_id": "b", "value": 2},
            {"request_id": "a", "value": 1},
            {"request_id": "c", "value": 3},
        ]
        cycle = profiler.PromptCycle(records)
        picked_ids = [cycle.next()[0]["request_id"] for _ in range(7)]
        self.assertEqual(picked_ids, ["a", "b", "c", "a", "b", "c", "a"])

    def test_sequence_numbers_increase_monotonically(self) -> None:
        cycle = profiler.PromptCycle([{"request_id": "only"}])
        sequences = [cycle.next()[1] for _ in range(5)]
        self.assertEqual(sequences, [0, 1, 2, 3, 4])

    def test_empty_cycle_rejected(self) -> None:
        with self.assertRaises(profiler.ProfilingError):
            profiler.PromptCycle([])

    def test_cycle_is_thread_safe_with_no_duplicate_sequence_numbers(self) -> None:
        cycle = profiler.PromptCycle([{"request_id": f"r{i}"} for i in range(8)])
        collected: list[int] = []
        lock = threading.Lock()

        def worker() -> None:
            for _ in range(200):
                _, sequence = cycle.next()
                with lock:
                    collected.append(sequence)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(len(collected), len(set(collected)))
        self.assertEqual(sorted(collected), list(range(8 * 200)))


class ExecuteProfilingRequestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.record = make_records(1)[1]  # first "balanced" record
        self.assertEqual(self.record["bucket"], "balanced")

    def test_successful_request_passes(self) -> None:
        transport = DelayedTransport()
        result = profiler.execute_profiling_request(
            self.record,
            "http://x/v1/completions",
            MODEL,
            10,
            transport,
            time.monotonic,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["observed_prompt_tokens"], 256)
        self.assertEqual(result["observed_completion_tokens"], 256)
        self.assertEqual(result["observed_total_tokens"], 512)
        self.assertEqual(result["finish_reason"], "length")

    def test_invalid_contract_response_fails(self) -> None:
        def bad_transport(endpoint, payload, timeout):
            response = successful_response(payload)
            body = json.loads(response.body)
            body["usage"]["completion_tokens"] -= 1
            return smoke.HttpResponse(200, json.dumps(body).encode("utf-8"))

        result = profiler.execute_profiling_request(
            self.record, "http://x/v1/completions", MODEL, 10, bad_transport, time.monotonic
        )
        self.assertFalse(result["passed"])
        self.assertIn(
            "completion_token_count",
            {f["reason"] for f in result["failure_reasons"]},
        )

    def test_http_server_failure_is_recorded(self) -> None:
        def failing_transport(endpoint, payload, timeout):
            return smoke.HttpResponse(500, b"")

        result = profiler.execute_profiling_request(
            self.record, "http://x/v1/completions", MODEL, 10, failing_transport, time.monotonic
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["http_status"], 500)
        self.assertEqual(result["failure_reasons"][0]["reason"], "http_status")


class SummarizePointTests(unittest.TestCase):
    """Pure, non-threaded tests of the D12/D16 summary arithmetic."""

    def _in_window_pass(self, prompt_tokens=256, completion_tokens=256):
        return {
            "passed": True,
            "in_measurement_window": True,
            "observed_prompt_tokens": prompt_tokens,
            "observed_completion_tokens": completion_tokens,
            "failure_reasons": [],
        }

    def test_exact_token_rate_and_512x_invariant(self) -> None:
        results = [self._in_window_pass() for _ in range(20)]
        summary = profiler.summarize_point(
            concurrency=4,
            results=results,
            max_observed_concurrency=4,
            t_start=0.0,
            t0=1.0,
            t1=3.0,
            measurement_seconds=2.0,
            submitted_count=20,
            outstanding_at_t1=0,
            outstanding_after_drain=0,
            drain_duration=0.0,
            drained=True,
        )
        self.assertEqual(summary["completed_requests_in_window"], 20)
        self.assertAlmostEqual(summary["completed_requests_per_second"], 10.0)
        self.assertAlmostEqual(summary["completed_total_tokens_per_second"], 5120.0)
        self.assertAlmostEqual(
            summary["completed_total_tokens_per_second"],
            512 * summary["completed_requests_per_second"],
        )
        self.assertTrue(summary["token_rate_invariant"]["holds"])
        self.assertTrue(summary["run_valid"])
        self.assertEqual(summary["invalidation_reasons"], [])

    def test_post_window_drain_excluded_from_numerator(self) -> None:
        in_window = self._in_window_pass()
        drain_completion = dict(self._in_window_pass())
        drain_completion["in_measurement_window"] = False
        results = [in_window, drain_completion]
        summary = profiler.summarize_point(
            concurrency=1,
            results=results,
            max_observed_concurrency=1,
            t_start=0.0,
            t0=1.0,
            t1=2.0,
            measurement_seconds=1.0,
            submitted_count=2,
            outstanding_at_t1=1,
            outstanding_after_drain=0,
            drain_duration=0.1,
            drained=True,
        )
        self.assertEqual(summary["completed_requests_in_window"], 1)
        self.assertAlmostEqual(summary["completed_requests_per_second"], 1.0)
        self.assertAlmostEqual(summary["completed_total_tokens_per_second"], 512.0)

    def test_failed_request_invalidates_point(self) -> None:
        results = [self._in_window_pass()]
        failed = self._in_window_pass()
        failed["passed"] = False
        failed["failure_reasons"] = [{"reason": "http_status", "detail": "x"}]
        results.append(failed)
        summary = profiler.summarize_point(
            concurrency=1,
            results=results,
            max_observed_concurrency=1,
            t_start=0.0,
            t0=0.0,
            t1=1.0,
            measurement_seconds=1.0,
            submitted_count=2,
            outstanding_at_t1=0,
            outstanding_after_drain=0,
            drain_duration=0.0,
            drained=True,
        )
        self.assertFalse(summary["run_valid"])
        self.assertIn("request_failures_present", summary["invalidation_reasons"])
        self.assertEqual(summary["failure_counts"], {"http_status": 1})

    def test_drain_timeout_invalidates_point(self) -> None:
        summary = profiler.summarize_point(
            concurrency=1,
            results=[],
            max_observed_concurrency=1,
            t_start=0.0,
            t0=0.0,
            t1=1.0,
            measurement_seconds=1.0,
            submitted_count=1,
            outstanding_at_t1=1,
            outstanding_after_drain=1,
            drain_duration=5.0,
            drained=False,
        )
        self.assertFalse(summary["run_valid"])
        self.assertIn("drain_timeout", summary["invalidation_reasons"])
        self.assertEqual(summary["drain_outcome"], "timed_out")

    def test_concurrency_not_achieved_invalidates_point(self) -> None:
        summary = profiler.summarize_point(
            concurrency=8,
            results=[],
            max_observed_concurrency=3,
            t_start=0.0,
            t0=0.0,
            t1=1.0,
            measurement_seconds=1.0,
            submitted_count=3,
            outstanding_at_t1=0,
            outstanding_after_drain=0,
            drain_duration=0.0,
            drained=True,
        )
        self.assertFalse(summary["run_valid"])
        self.assertIn("concurrency_not_achieved", summary["invalidation_reasons"])

    def test_deterministic_summary_given_same_inputs(self) -> None:
        results = [self._in_window_pass() for _ in range(5)]
        kwargs = dict(
            concurrency=2,
            results=results,
            max_observed_concurrency=2,
            t_start=0.0,
            t0=1.0,
            t1=2.0,
            measurement_seconds=1.0,
            submitted_count=5,
            outstanding_at_t1=0,
            outstanding_after_drain=0,
            drain_duration=0.0,
            drained=True,
        )
        first = profiler.summarize_point(**kwargs)
        second = profiler.summarize_point(**kwargs)
        self.assertEqual(first, second)


class AdjacentGainTests(unittest.TestCase):
    def test_no_prior_point_returns_none(self) -> None:
        current = {"completed_total_tokens_per_second": 1000.0}
        self.assertIsNone(profiler.compute_adjacent_gain(None, current))

    def test_gain_is_relative_change(self) -> None:
        previous = {"completed_total_tokens_per_second": 1000.0}
        current = {"completed_total_tokens_per_second": 1100.0}
        self.assertAlmostEqual(profiler.compute_adjacent_gain(previous, current), 0.1)

    def test_zero_previous_rate_returns_none(self) -> None:
        previous = {"completed_total_tokens_per_second": 0.0}
        current = {"completed_total_tokens_per_second": 10.0}
        self.assertIsNone(profiler.compute_adjacent_gain(previous, current))


class PrometheusParsingTests(unittest.TestCase):
    def test_parses_labels_and_values_and_skips_comments(self) -> None:
        text = (
            "# HELP vllm:num_requests_running running\n"
            "# TYPE vllm:num_requests_running gauge\n"
            'vllm:num_requests_running{model_name="Qwen/Qwen2.5-3B"} 4.0\n'
            "vllm:num_requests_waiting 0\n"
            "\n"
        )
        samples = profiler.parse_prometheus_text(text)
        names = {s["name"] for s in samples}
        self.assertIn("vllm:num_requests_running", names)
        self.assertIn("vllm:num_requests_waiting", names)
        running = next(s for s in samples if s["name"] == "vllm:num_requests_running")
        self.assertEqual(running["labels"], {"model_name": "Qwen/Qwen2.5-3B"})
        self.assertEqual(running["value"], 4.0)

    def test_select_known_metrics_marks_absence(self) -> None:
        samples = [{"name": "vllm:num_requests_running", "labels": {}, "value": 1.0}]
        known = profiler.select_known_metrics(samples)
        self.assertTrue(known["vllm:num_requests_running"]["present"])
        self.assertFalse(known["vllm:num_requests_waiting"]["present"])


class GpuTelemetryTests(unittest.TestCase):
    def test_missing_nvidia_smi_is_surfaced_not_fatal(self) -> None:
        def fake_run(*args, **kwargs):
            raise FileNotFoundError("nvidia-smi not found")

        result = profiler.sample_gpu_telemetry(run=fake_run)
        self.assertFalse(result["available"])
        self.assertIn("error", result)

    def test_successful_sample_shape(self) -> None:
        class FakeCompleted:
            returncode = 0
            stdout = "10, 512, 16384, 45, 55.0, 1500, Active\n"
            stderr = ""

        def fake_run(*args, **kwargs):
            return FakeCompleted()

        result = profiler.sample_gpu_telemetry(run=fake_run)
        self.assertTrue(result["available"])
        self.assertEqual(len(result["gpus"]), 1)
        self.assertEqual(result["gpus"][0]["utilization.gpu"], "10")


class TelemetrySamplerTests(unittest.TestCase):
    def test_sample_once_records_ok_and_gpu_entries(self) -> None:
        vllm_samples: list[dict[str, Any]] = []
        gpu_samples: list[dict[str, Any]] = []

        def metrics_transport(endpoint, timeout):
            return smoke.HttpResponse(200, b'vllm:num_requests_running{} 2.0\n')

        def gpu_sampler():
            return {"available": True, "gpus": []}

        sampler = profiler.TelemetrySampler(
            metrics_endpoint="http://x/metrics",
            metrics_transport=metrics_transport,
            gpu_sampler=gpu_sampler,
            clock=lambda: 42.0,
            interval_seconds=0.01,
            request_timeout_seconds=5,
            on_vllm_sample=vllm_samples.append,
            on_gpu_sample=gpu_samples.append,
        )
        sampler.sample_once()
        self.assertEqual(len(vllm_samples), 1)
        self.assertEqual(vllm_samples[0]["status"], "ok")
        self.assertEqual(vllm_samples[0]["timestamp"], 42.0)
        self.assertEqual(len(gpu_samples), 1)
        self.assertTrue(gpu_samples[0]["available"])

    def test_transport_error_is_recorded_not_raised(self) -> None:
        def failing_metrics_transport(endpoint, timeout):
            raise profiler.TransportError("connection refused")

        vllm_samples: list[dict[str, Any]] = []
        sampler = profiler.TelemetrySampler(
            metrics_endpoint="http://x/metrics",
            metrics_transport=failing_metrics_transport,
            gpu_sampler=lambda: {"available": False, "error": "no gpu"},
            clock=time.monotonic,
            interval_seconds=0.01,
            request_timeout_seconds=5,
            on_vllm_sample=vllm_samples.append,
            on_gpu_sample=lambda entry: None,
        )
        sampler.sample_once()
        self.assertEqual(vllm_samples[0]["status"], "transport_error")

    def test_start_and_stop_bound_the_sampling_thread(self) -> None:
        counter_lock = threading.Lock()
        counts = {"vllm": 0}

        def metrics_transport(endpoint, timeout):
            return smoke.HttpResponse(200, b"")

        def on_vllm_sample(entry):
            with counter_lock:
                counts["vllm"] += 1

        sampler = profiler.TelemetrySampler(
            metrics_endpoint="http://x/metrics",
            metrics_transport=metrics_transport,
            gpu_sampler=lambda: {"available": False, "error": "disabled"},
            clock=time.monotonic,
            interval_seconds=0.02,
            request_timeout_seconds=5,
            on_vllm_sample=on_vllm_sample,
            on_gpu_sample=lambda entry: None,
        )
        sampler.start()
        time.sleep(0.15)
        sampler.stop(join_timeout_seconds=2.0)
        observed_after_stop = counts["vllm"]
        time.sleep(0.1)
        self.assertGreaterEqual(observed_after_stop, 2)
        self.assertEqual(counts["vllm"], observed_after_stop)


class RunLoadPointTests(unittest.TestCase):
    """These tests exercise the real threaded closed-loop scheduler."""

    def setUp(self) -> None:
        all_records = make_records(64)
        self.records = profiler.select_balanced_records(all_records)
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
        self.assertTrue(all(result["concurrency"] == 4 for result in results))
        self.assertTrue(all(result["run_id"] == "test-run" for result in results))

    def test_no_admission_after_t1_and_drain_excludes_from_numerator(self) -> None:
        # Each request takes far longer than the settling+measurement window,
        # so only the initial fill is ever admitted, and every completion
        # necessarily lands in the drain phase (after T1).
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
        # No replacement requests were admitted after the initial fill.
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


class RunExperimentIntegrationTests(unittest.TestCase):
    """End-to-end test of the orchestration and artifact-writing layer."""

    def test_full_experiment_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset = root / "profiling.jsonl"
            output_dir = root / "profiling-results"
            write_dataset(dataset, make_records(8))
            transport = DelayedTransport(delay_seconds=0.01)

            def probe_transport(endpoint, timeout):
                if endpoint.endswith("/v1/models"):
                    body = json.dumps({"data": [{"id": MODEL}]}).encode("utf-8")
                    return smoke.HttpResponse(200, body)
                if endpoint.endswith("/version"):
                    return smoke.HttpResponse(200, b'{"version": "0.28.0"}')
                return smoke.HttpResponse(404, b"")

            def metrics_transport(endpoint, timeout):
                return smoke.HttpResponse(200, b"vllm:num_requests_running 0\n")

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
                concurrency_ladder=(1, 2),
                timing=profiler.TimingConfig(
                    settling_seconds=0.02,
                    measurement_seconds=0.05,
                    drain_timeout_seconds=1.0,
                    metrics_interval_seconds=0.02,
                    request_timeout_seconds=5.0,
                ),
                run_id="integration-test",
                transport=transport,
                probe_transport=probe_transport,
                metrics_transport=metrics_transport,
                gpu_sampler=gpu_sampler,
                fingerprint_runner=fingerprint_runner,
            )

            bundle = profiler.run_experiment(config)
            profiler.write_artifacts(output_dir, bundle)

            self.assertTrue((output_dir / profiler.MANIFEST_FILENAME).is_file())
            self.assertTrue((output_dir / profiler.REQUEST_RESULTS_FILENAME).is_file())
            self.assertTrue((output_dir / profiler.POINT_SUMMARIES_FILENAME).is_file())
            self.assertTrue((output_dir / profiler.SUMMARY_FILENAME).is_file())
            self.assertTrue(
                (output_dir / profiler.VLLM_METRICS_FILENAME).is_file()
            )
            self.assertTrue((output_dir / profiler.GPU_METRICS_FILENAME).is_file())

            manifest = json.loads(
                (output_dir / profiler.MANIFEST_FILENAME).read_text("utf-8")
            )
            self.assertEqual(manifest["dataset"]["bucket_definition"]["name"], "balanced")
            self.assertEqual(manifest["concurrency_ladder"], [1, 2])
            self.assertEqual(
                manifest["server_identity"]["served_model_ids"], [MODEL]
            )
            self.assertFalse(manifest["server_identity"]["gpu_fingerprint"]["available"])

            point_summaries = [
                json.loads(line)
                for line in (output_dir / profiler.POINT_SUMMARIES_FILENAME)
                .read_text("utf-8")
                .splitlines()
            ]
            self.assertEqual(len(point_summaries), 2)
            self.assertIsNone(point_summaries[0]["adjacent_throughput_gain"])

            summary = json.loads(
                (output_dir / profiler.SUMMARY_FILENAME).read_text("utf-8")
            )
            self.assertEqual(len(summary["review_table"]), 2)
            self.assertIn("plateau_acceptance_warning", summary)

            request_results = [
                json.loads(line)
                for line in (output_dir / profiler.REQUEST_RESULTS_FILENAME)
                .read_text("utf-8")
                .splitlines()
            ]
            self.assertTrue(request_results)
            self.assertTrue(all(r["bucket"] == "balanced" for r in request_results))

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

    def test_config_validation_rejects_non_increasing_ladder(self) -> None:
        config = profiler.ExperimentConfig(
            profiling_jsonl=Path("unused.jsonl"),
            output_dir=Path("unused-output"),
            base_url="http://127.0.0.1:8000",
            model=MODEL,
            tokenizer_revision=REVISION,
            concurrency_ladder=(2, 1),
        )
        with self.assertRaises(profiler.ProfilingError):
            config.validate()


if __name__ == "__main__":
    unittest.main()

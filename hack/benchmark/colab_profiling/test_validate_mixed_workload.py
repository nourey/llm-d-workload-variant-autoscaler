"""CPU-only tests for the #1546 open-loop mixed-workload validation harness.

Reuses the existing fake-transport fixtures from ``test_profile_bucket.py``
(the pure-bucket profiler's own test suite) rather than duplicating them.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_dataset as generator
import profile_bucket as pb
import run_request_smoke as smoke
import validate_mixed_workload as vmw
from test_profile_bucket import (
    MODEL,
    REVISION,
    DelayedTransport,
    IncrementingMetricsTransport,
    always_ok_idle_check,
    make_records,
    write_dataset,
)


BUCKET_DEFINITIONS = {bucket.name: bucket for bucket in generator.DEFAULT_BUCKETS}


# ---------------------------------------------------------------------------
# Composition math (#1546 Part N: 1-6)
# ---------------------------------------------------------------------------


class NormalizeWeightsTests(unittest.TestCase):
    def test_weights_normalize_to_one(self) -> None:
        alphas = vmw.normalize_weights({"a": 1.0, "b": 1.0, "c": 1.0})
        self.assertAlmostEqual(sum(alphas.values()), 1.0)
        self.assertAlmostEqual(alphas["a"], 1 / 3)
        self.assertAlmostEqual(alphas["b"], 1 / 3)
        self.assertAlmostEqual(alphas["c"], 1 / 3)

    def test_unequal_raw_weights_normalize_proportionally(self) -> None:
        alphas = vmw.normalize_weights({"a": 1.0, "b": 3.0})
        self.assertAlmostEqual(alphas["a"], 0.25)
        self.assertAlmostEqual(alphas["b"], 0.75)

    def test_empty_weights_rejected(self) -> None:
        with self.assertRaises(vmw.ProfilingError):
            vmw.normalize_weights({})

    def test_negative_or_nonfinite_weight_rejected(self) -> None:
        with self.assertRaises(vmw.ProfilingError):
            vmw.normalize_weights({"a": -1.0})
        with self.assertRaises(vmw.ProfilingError):
            vmw.normalize_weights({"a": float("nan")})

    def test_all_zero_weights_rejected(self) -> None:
        with self.assertRaises(vmw.ProfilingError):
            vmw.normalize_weights({"a": 0.0, "b": 0.0})


class DeriveBucketTargetsTests(unittest.TestCase):
    def test_target_rho_decomposition_and_request_rate_derivation(self) -> None:
        targets = vmw.derive_bucket_targets(
            target_rho=1.0,
            alphas={"input-heavy": 1 / 3, "balanced": 1 / 3, "output-heavy": 1 / 3},
            capacities={"input-heavy": 2180.0, "balanced": 1965.0, "output-heavy": 1880.0},
            bucket_definitions=BUCKET_DEFINITIONS,
        )
        # #1546 Part L worked example.
        self.assertAlmostEqual(targets["input-heavy"]["rho_b"], 1 / 3)
        self.assertAlmostEqual(targets["input-heavy"]["lambda_b"], (1 / 3) * 2180.0)
        self.assertAlmostEqual(
            targets["input-heavy"]["target_request_rate"], ((1 / 3) * 2180.0) / 512
        )
        self.assertAlmostEqual(targets["balanced"]["lambda_b"], (1 / 3) * 1965.0)
        self.assertAlmostEqual(targets["output-heavy"]["lambda_b"], (1 / 3) * 1880.0)

    def test_different_w_b_values_are_handled_independently(self) -> None:
        custom_definitions = {
            "short": generator.Bucket("short", input_tokens=50, target_output_tokens=50),
            "long": generator.Bucket("long", input_tokens=900, target_output_tokens=900),
        }
        targets = vmw.derive_bucket_targets(
            target_rho=1.0,
            alphas={"short": 0.5, "long": 0.5},
            capacities={"short": 1000.0, "long": 1000.0},
            bucket_definitions=custom_definitions,
        )
        self.assertEqual(targets["short"]["w"], 100)
        self.assertEqual(targets["long"]["w"], 1800)
        # Same lambda (0.5 * 1000 = 500) but very different request rates
        # because W differs.
        self.assertAlmostEqual(targets["short"]["target_request_rate"], 500 / 100)
        self.assertAlmostEqual(targets["long"]["target_request_rate"], 500 / 1800)

    def test_missing_capacity_fails_closed(self) -> None:
        with self.assertRaises(vmw.ProfilingError):
            vmw.derive_bucket_targets(
                target_rho=1.0,
                alphas={"balanced": 1.0},
                capacities={},
                bucket_definitions=BUCKET_DEFINITIONS,
            )

    def test_zero_or_negative_capacity_fails_closed(self) -> None:
        with self.assertRaises(vmw.ProfilingError):
            vmw.derive_bucket_targets(
                target_rho=1.0,
                alphas={"balanced": 1.0},
                capacities={"balanced": 0.0},
                bucket_definitions=BUCKET_DEFINITIONS,
            )
        with self.assertRaises(vmw.ProfilingError):
            vmw.derive_bucket_targets(
                target_rho=1.0,
                alphas={"balanced": 1.0},
                capacities={"balanced": -5.0},
                bucket_definitions=BUCKET_DEFINITIONS,
            )

    def test_nonfinite_capacity_fails_closed(self) -> None:
        with self.assertRaises(vmw.ProfilingError):
            vmw.derive_bucket_targets(
                target_rho=1.0,
                alphas={"balanced": 1.0},
                capacities={"balanced": float("inf")},
                bucket_definitions=BUCKET_DEFINITIONS,
            )

    def test_nonpositive_target_rho_fails_closed(self) -> None:
        with self.assertRaises(vmw.ProfilingError):
            vmw.derive_bucket_targets(
                target_rho=0.0,
                alphas={"balanced": 1.0},
                capacities={"balanced": 100.0},
                bucket_definitions=BUCKET_DEFINITIONS,
            )


class ComputeRhoFromLambdasTests(unittest.TestCase):
    def test_rho_pred_target_reconstructs_requested_rho(self) -> None:
        capacities = {"input-heavy": 2180.0, "balanced": 1965.0, "output-heavy": 1880.0}
        for target_rho in (0.70, 1.00, 1.15):
            targets = vmw.derive_bucket_targets(
                target_rho=target_rho,
                alphas={"input-heavy": 1 / 3, "balanced": 1 / 3, "output-heavy": 1 / 3},
                capacities=capacities,
                bucket_definitions=BUCKET_DEFINITIONS,
            )
            rho_pred = vmw.compute_rho_from_lambdas(
                {b: t["lambda_b"] for b, t in targets.items()}, capacities
            )
            self.assertAlmostEqual(rho_pred, target_rho, places=9)

    def test_rho_pred_achieved_from_actual_arrivals(self) -> None:
        # Achieved lambdas independent of the target plan.
        capacities = {"a": 100.0, "b": 200.0}
        achieved_lambdas = {"a": 50.0, "b": 100.0}
        rho = vmw.compute_rho_from_lambdas(achieved_lambdas, capacities)
        self.assertAlmostEqual(rho, 50 / 100 + 100 / 200)


# ---------------------------------------------------------------------------
# Deterministic open-loop arrival scheduling (#1546 Part N: 7-9)
# ---------------------------------------------------------------------------


class DeterministicArrivalScheduleTests(unittest.TestCase):
    def test_absolute_time_schedule_is_deterministic(self) -> None:
        schedule = vmw.DeterministicArrivalSchedule(
            origin=100.0, phase_seconds=0.5, request_rate_per_second=10.0
        )
        self.assertAlmostEqual(schedule.scheduled_time(0), 100.5)
        self.assertAlmostEqual(schedule.scheduled_time(1), 100.6)
        self.assertAlmostEqual(schedule.scheduled_time(10), 101.5)

    def test_schedule_never_drifts_with_simulated_delay(self) -> None:
        # Even if we pretend a huge, varying delay occurred before each
        # index is computed, scheduled_time(k) for a FIXED k never changes:
        # it depends only on (origin, phase, rate, k), never on "now".
        schedule = vmw.DeterministicArrivalSchedule(
            origin=0.0, phase_seconds=0.0, request_rate_per_second=4.0
        )
        first_call = schedule.scheduled_time(7)
        second_call = schedule.scheduled_time(7)
        self.assertEqual(first_call, second_call)
        self.assertEqual(first_call, 7 / 4.0)

    def test_nonpositive_rate_rejected(self) -> None:
        with self.assertRaises(vmw.ProfilingError):
            vmw.DeterministicArrivalSchedule(
                origin=0.0, phase_seconds=0.0, request_rate_per_second=0.0
            )

    def test_negative_phase_rejected(self) -> None:
        with self.assertRaises(vmw.ProfilingError):
            vmw.DeterministicArrivalSchedule(
                origin=0.0, phase_seconds=-1.0, request_rate_per_second=1.0
            )


# ---------------------------------------------------------------------------
# Percentile / trend helpers
# ---------------------------------------------------------------------------


class SummarizeSchedulingLagTests(unittest.TestCase):
    def test_basic_statistics(self) -> None:
        summary = vmw.summarize_scheduling_lag([0.1, 0.2, 0.3, 0.4, None])
        self.assertTrue(summary["available"])
        self.assertEqual(summary["count"], 4)
        self.assertAlmostEqual(summary["mean"], 0.25)
        self.assertAlmostEqual(summary["max"], 0.4)
        self.assertAlmostEqual(summary["p50"], 0.25)

    def test_empty_is_explicit(self) -> None:
        summary = vmw.summarize_scheduling_lag([])
        self.assertFalse(summary["available"])
        self.assertEqual(summary["count"], 0)


class TimeSeriesTrendTests(unittest.TestCase):
    def test_waiting_trend_computation(self) -> None:
        # Perfectly linear growth: waiting = 10 + 2*t.
        series = [(float(t), 10.0 + 2.0 * t) for t in range(10)]
        summary = vmw.summarize_time_series_with_trend(
            series, t0=0.0, t1=9.0, first_last_window_fraction=0.2
        )
        self.assertTrue(summary["available"])
        self.assertAlmostEqual(summary["trend_slope_per_second"], 2.0, places=6)

    def test_first_last_window_median(self) -> None:
        series = [(0.0, 1.0), (1.0, 2.0), (8.0, 9.0), (9.0, 10.0)]
        summary = vmw.summarize_time_series_with_trend(
            series, t0=0.0, t1=10.0, first_last_window_fraction=0.2
        )
        # first 20% of [0,10) is [0,2): values 1.0, 2.0 -> median 1.5
        # last 20% is [8,10): values 9.0, 10.0 -> median 9.5
        self.assertAlmostEqual(summary["first_window_median"], 1.5)
        self.assertAlmostEqual(summary["last_window_median"], 9.5)
        self.assertAlmostEqual(summary["last_minus_first_median"], 8.0)
        self.assertEqual(summary["first_last_window_fraction"], 0.2)

    def test_empty_series_is_explicit(self) -> None:
        summary = vmw.summarize_time_series_with_trend(
            [], t0=0.0, t1=10.0, first_last_window_fraction=0.2
        )
        self.assertFalse(summary["available"])


class RunningSeriesTests(unittest.TestCase):
    def test_ceiling_fraction_is_observed_not_hardcoded(self) -> None:
        series = [(0.0, 10.0), (1.0, 42.0), (2.0, 42.0), (3.0, 20.0)]
        summary = vmw.summarize_running_series(series)
        self.assertEqual(summary["max"], 42.0)
        self.assertAlmostEqual(summary["fraction_of_samples_at_observed_ceiling"], 2 / 4)

    def test_empty_series_is_explicit(self) -> None:
        self.assertFalse(vmw.summarize_running_series([])["available"])


# ---------------------------------------------------------------------------
# Config validation (fail-closed capacity/weight inputs)
# ---------------------------------------------------------------------------


class MixedExperimentConfigValidationTests(unittest.TestCase):
    def _base_kwargs(self, **overrides: Any) -> dict[str, Any]:
        kwargs = dict(
            profiling_jsonl=Path("unused.jsonl"),
            output_dir=Path("unused-output"),
            base_url="http://127.0.0.1:8000",
            model=MODEL,
            tokenizer_revision=REVISION,
            capacities={"balanced": 1965.0},
        )
        kwargs.update(overrides)
        return kwargs

    def test_missing_capacities_fails_closed(self) -> None:
        config = vmw.MixedExperimentConfig(**self._base_kwargs(capacities={}))
        with self.assertRaises(vmw.ProfilingError):
            config.validate()

    def test_zero_capacity_fails_closed(self) -> None:
        config = vmw.MixedExperimentConfig(
            **self._base_kwargs(capacities={"balanced": 0.0})
        )
        with self.assertRaises(vmw.ProfilingError):
            config.validate()

    def test_unknown_bucket_capacity_fails_closed(self) -> None:
        config = vmw.MixedExperimentConfig(
            **self._base_kwargs(capacities={"does-not-exist": 100.0})
        )
        with self.assertRaises(vmw.ProfilingError):
            config.validate()

    def test_weight_bucket_set_must_match_capacity_bucket_set(self) -> None:
        config = vmw.MixedExperimentConfig(
            **self._base_kwargs(
                capacities={"balanced": 100.0},
                raw_weights={"input-heavy": 1.0},
            )
        )
        with self.assertRaises(vmw.ProfilingError):
            config.validate()

    def test_no_target_rho_fails_closed(self) -> None:
        config = vmw.MixedExperimentConfig(
            **self._base_kwargs(target_rho_values=())
        )
        with self.assertRaises(vmw.ProfilingError):
            config.validate()

    def test_negative_target_rho_fails_closed(self) -> None:
        config = vmw.MixedExperimentConfig(
            **self._base_kwargs(target_rho_values=(-1.0,))
        )
        with self.assertRaises(vmw.ProfilingError):
            config.validate()

    def test_default_alphas_are_equal_when_no_weights_supplied(self) -> None:
        config = vmw.MixedExperimentConfig(
            **self._base_kwargs(
                capacities={"input-heavy": 1.0, "balanced": 1.0, "output-heavy": 1.0}
            )
        )
        alphas = config.alphas
        self.assertAlmostEqual(alphas["input-heavy"], 1 / 3)
        self.assertAlmostEqual(alphas["balanced"], 1 / 3)
        self.assertAlmostEqual(alphas["output-heavy"], 1 / 3)


# ---------------------------------------------------------------------------
# Integration: real threaded open-loop load point
# ---------------------------------------------------------------------------


def _make_bucket_cycles(records_per_bucket: int = 32) -> dict[str, pb.PromptCycle]:
    records = make_records(records_per_bucket)
    return {
        bucket.name: pb.PromptCycle(pb.select_bucket_records(records, bucket.name))
        for bucket in generator.DEFAULT_BUCKETS
    }


class RunMixedLoadPointIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bucket_cycles = _make_bucket_cycles()
        self.endpoint = "http://127.0.0.1:8000/v1/completions"
        # Deliberately small, fast, round-number capacities so target
        # request rates are easy to reason about in tests: with W=512,
        # alpha=1/3 and target_rho=1.0, rate_b = capacity_b / (3*512).
        self.capacities = {"input-heavy": 15360.0, "balanced": 15360.0, "output-heavy": 15360.0}
        self.bucket_definitions = {b.name: b for b in generator.DEFAULT_BUCKETS}

    def _targets(self, target_rho: float = 1.0) -> dict[str, dict[str, float]]:
        alphas = {"input-heavy": 1 / 3, "balanced": 1 / 3, "output-heavy": 1 / 3}
        return vmw.derive_bucket_targets(
            target_rho=target_rho,
            alphas=alphas,
            capacities=self.capacities,
            bucket_definitions=self.bucket_definitions,
        )

    def _telemetry_config(self, timing: "vmw.MixedTimingConfig") -> pb.TelemetryConfig:
        # A real vLLM /metrics endpoint is required for the mandatory
        # engine estimator; use the same deterministic incrementing fake
        # the pure-bucket profiler's own test suite uses.
        return pb.TelemetryConfig(
            metrics_endpoint="http://x/metrics",
            metrics_transport=IncrementingMetricsTransport(),
            gpu_sampler=lambda: {"available": False, "error": "no gpu in tests"},
            interval_seconds=timing.metrics_interval_seconds,
            request_timeout_seconds=timing.request_timeout_seconds,
        )

    def test_bucket_streams_are_independently_paced(self) -> None:
        # Give output-heavy 4x the rate of the other two by using a skewed
        # weight, and check the actual measured inter-arrival gap per
        # bucket reflects that ratio, independent of the other buckets.
        alphas = vmw.normalize_weights(
            {"input-heavy": 1.0, "balanced": 1.0, "output-heavy": 4.0}
        )
        targets = vmw.derive_bucket_targets(
            target_rho=1.0,
            alphas=alphas,
            capacities=self.capacities,
            bucket_definitions=self.bucket_definitions,
        )
        transport = DelayedTransport(delay_seconds=0.0)
        timing = vmw.MixedTimingConfig(
            settling_seconds=0.05,
            measurement_seconds=0.6,
            drain_timeout_seconds=3.0,
            request_timeout_seconds=5.0,
            metrics_interval_seconds=1.0,
        )
        summary, results = vmw.run_mixed_load_point(
            target_rho=1.0,
            point_index=0,
            bucket_cycles=self.bucket_cycles,
            bucket_targets=targets,
            endpoint=self.endpoint,
            model=MODEL,
            timing=timing,
            transport=transport,
            client_concurrency_budget=512,
            max_scheduling_lag_p95_seconds=1.0,
            max_achieved_rate_relative_error=0.5,
            waiting_trend_window_fraction=0.2,
            phase_offset_step_seconds=0.01,
            idle_check=always_ok_idle_check,
            run_id="mixed-test",
        )
        for bucket_name in ("input-heavy", "balanced", "output-heavy"):
            bucket_results = sorted(
                (r for r in results if r["bucket"] == bucket_name),
                key=lambda r: r["sequence"],
            )
            self.assertGreaterEqual(len(bucket_results), 3, msg=bucket_name)
            scheduled_times = [r["scheduled_monotonic_s"] for r in bucket_results]
            gaps = [b - a for a, b in zip(scheduled_times, scheduled_times[1:])]
            expected_gap = 1.0 / targets[bucket_name]["target_request_rate"]
            for gap in gaps:
                self.assertAlmostEqual(gap, expected_gap, places=9)
        # output-heavy was scheduled ~4x more densely than input-heavy.
        self.assertLess(
            1.0 / targets["output-heavy"]["target_request_rate"],
            1.0 / targets["input-heavy"]["target_request_rate"],
        )

    def test_submission_counts_measured_using_t0_t1_window(self) -> None:
        transport = DelayedTransport(delay_seconds=0.0)
        timing = vmw.MixedTimingConfig(
            settling_seconds=0.1,
            measurement_seconds=0.2,
            drain_timeout_seconds=3.0,
            request_timeout_seconds=5.0,
            metrics_interval_seconds=0.02,
        )
        targets = self._targets(target_rho=1.0)
        summary, results = vmw.run_mixed_load_point(
            target_rho=1.0,
            point_index=0,
            bucket_cycles=self.bucket_cycles,
            bucket_targets=targets,
            endpoint=self.endpoint,
            model=MODEL,
            timing=timing,
            transport=transport,
            client_concurrency_budget=512,
            max_scheduling_lag_p95_seconds=1.0,
            max_achieved_rate_relative_error=0.5,
            waiting_trend_window_fraction=0.2,
            phase_offset_step_seconds=0.01,
            idle_check=always_ok_idle_check,
            telemetry_config=self._telemetry_config(timing),
            run_id="mixed-test",
        )
        t0 = summary["measurement_t0_s"]
        t1 = summary["measurement_t1_s"]
        for bucket_name, bucket_summary in summary["buckets"].items():
            manual_count = sum(
                1
                for r in results
                if r["bucket"] == bucket_name
                and r.get("submit_monotonic_s") is not None
                and t0 <= r["submit_monotonic_s"] < t1
            )
            self.assertEqual(
                bucket_summary["achieved_request_count_in_window"], manual_count
            )
        # With a fast/instant transport, achieved rate should track target
        # rate closely.
        for bucket_summary in summary["buckets"].values():
            self.assertLess(bucket_summary["achieved_rate_relative_error"], 0.5)
        self.assertTrue(summary["run_valid"], msg=summary["invalidation_reasons"])

    def test_client_concurrency_budget_exceeded_invalidates(self) -> None:
        # A slow transport plus a tiny client budget forces client-side
        # thread-pool queueing -- exactly the failure this must catch.
        transport = DelayedTransport(delay_seconds=0.5)
        timing = vmw.MixedTimingConfig(
            settling_seconds=0.05,
            measurement_seconds=0.3,
            drain_timeout_seconds=3.0,
            request_timeout_seconds=5.0,
            metrics_interval_seconds=1.0,
        )
        targets = self._targets(target_rho=1.0)
        summary, _ = vmw.run_mixed_load_point(
            target_rho=1.0,
            point_index=0,
            bucket_cycles=self.bucket_cycles,
            bucket_targets=targets,
            endpoint=self.endpoint,
            model=MODEL,
            timing=timing,
            transport=transport,
            client_concurrency_budget=2,  # deliberately tiny
            max_scheduling_lag_p95_seconds=10.0,
            max_achieved_rate_relative_error=10.0,
            waiting_trend_window_fraction=0.2,
            phase_offset_step_seconds=0.01,
            idle_check=always_ok_idle_check,
            run_id="mixed-test",
        )
        self.assertFalse(summary["run_valid"])
        self.assertTrue(
            any(
                reason.startswith("client_concurrency_budget_exceeded")
                for reason in summary["invalidation_reasons"]
            )
        )

    def test_server_backlog_alone_does_not_invalidate(self) -> None:
        # Slow transport (simulated server backlog) but a generous client
        # budget: the client never queues internally, so outstanding grows
        # (real backlog evidence) without inflating scheduling lag or
        # tripping the client-budget guard.
        transport = DelayedTransport(delay_seconds=0.3)
        timing = vmw.MixedTimingConfig(
            settling_seconds=0.05,
            measurement_seconds=0.3,
            drain_timeout_seconds=5.0,
            request_timeout_seconds=5.0,
            metrics_interval_seconds=0.02,
        )
        targets = self._targets(target_rho=1.0)
        summary, _ = vmw.run_mixed_load_point(
            target_rho=1.0,
            point_index=0,
            bucket_cycles=self.bucket_cycles,
            bucket_targets=targets,
            endpoint=self.endpoint,
            model=MODEL,
            timing=timing,
            transport=transport,
            client_concurrency_budget=4096,
            max_scheduling_lag_p95_seconds=1.0,
            max_achieved_rate_relative_error=0.5,
            waiting_trend_window_fraction=0.2,
            phase_offset_step_seconds=0.01,
            idle_check=always_ok_idle_check,
            telemetry_config=self._telemetry_config(timing),
            run_id="mixed-test",
        )
        self.assertGreater(summary["outstanding_at_t1"], 0)
        self.assertTrue(summary["run_valid"], msg=summary["invalidation_reasons"])

    def test_scheduling_lag_threshold_invalidates(self) -> None:
        transport = DelayedTransport(delay_seconds=0.0)
        timing = vmw.MixedTimingConfig(
            settling_seconds=0.05,
            measurement_seconds=0.2,
            drain_timeout_seconds=3.0,
            request_timeout_seconds=5.0,
            metrics_interval_seconds=1.0,
        )
        targets = self._targets(target_rho=1.0)
        summary, _ = vmw.run_mixed_load_point(
            target_rho=1.0,
            point_index=0,
            bucket_cycles=self.bucket_cycles,
            bucket_targets=targets,
            endpoint=self.endpoint,
            model=MODEL,
            timing=timing,
            transport=transport,
            client_concurrency_budget=512,
            max_scheduling_lag_p95_seconds=0.0,  # impossible to satisfy
            max_achieved_rate_relative_error=10.0,
            waiting_trend_window_fraction=0.2,
            phase_offset_step_seconds=0.01,
            idle_check=always_ok_idle_check,
            run_id="mixed-test",
        )
        self.assertFalse(summary["run_valid"])
        self.assertTrue(
            any(
                reason.startswith("scheduling_lag_p95_exceeds_threshold")
                for reason in summary["invalidation_reasons"]
            )
        )

    def test_achieved_rate_error_threshold_invalidates(self) -> None:
        # With target_request_rate == 10/s (inter-arrival exactly 0.1s), a
        # measurement window that is NOT an exact multiple of 0.1s (0.25s)
        # forces a genuine discretization gap between the target rate and
        # whatever whole number of arrivals actually lands in the window
        # (2 or 3 arrivals -> 8/s or 12/s, both >20% off target) -- a real,
        # reproducible achieved-rate deviation, not a threshold of exactly
        # zero (which a perfectly deterministic scheduler can satisfy).
        transport = DelayedTransport(delay_seconds=0.0)
        timing = vmw.MixedTimingConfig(
            settling_seconds=0.05,
            measurement_seconds=0.25,
            drain_timeout_seconds=3.0,
            request_timeout_seconds=5.0,
            metrics_interval_seconds=0.02,
        )
        targets = self._targets(target_rho=1.0)
        summary, _ = vmw.run_mixed_load_point(
            target_rho=1.0,
            point_index=0,
            bucket_cycles=self.bucket_cycles,
            bucket_targets=targets,
            endpoint=self.endpoint,
            model=MODEL,
            timing=timing,
            transport=transport,
            client_concurrency_budget=512,
            max_scheduling_lag_p95_seconds=10.0,
            max_achieved_rate_relative_error=0.01,
            waiting_trend_window_fraction=0.2,
            phase_offset_step_seconds=0.01,
            idle_check=always_ok_idle_check,
            telemetry_config=self._telemetry_config(timing),
            run_id="mixed-test",
        )
        self.assertFalse(summary["run_valid"])
        self.assertTrue(
            any(
                reason.startswith("achieved_rate_relative_error_exceeds_threshold")
                for reason in summary["invalidation_reasons"]
            )
        )

    def test_drain_failure_fails_closed(self) -> None:
        transport = DelayedTransport(delay_seconds=2.0)
        timing = vmw.MixedTimingConfig(
            settling_seconds=0.02,
            measurement_seconds=0.05,
            drain_timeout_seconds=0.05,  # far shorter than the 2s transport delay
            request_timeout_seconds=5.0,
            metrics_interval_seconds=1.0,
        )
        targets = self._targets(target_rho=1.0)
        summary, _ = vmw.run_mixed_load_point(
            target_rho=1.0,
            point_index=0,
            bucket_cycles=self.bucket_cycles,
            bucket_targets=targets,
            endpoint=self.endpoint,
            model=MODEL,
            timing=timing,
            transport=transport,
            client_concurrency_budget=512,
            max_scheduling_lag_p95_seconds=10.0,
            max_achieved_rate_relative_error=10.0,
            waiting_trend_window_fraction=0.2,
            phase_offset_step_seconds=0.01,
            idle_check=always_ok_idle_check,
            run_id="mixed-test",
        )
        self.assertEqual(summary["drain_outcome"], "timed_out")
        self.assertIn("drain_timeout", summary["invalidation_reasons"])
        self.assertFalse(summary["run_valid"])

    def test_engine_estimator_uses_existing_contract_no_completion_fallback(self) -> None:
        transport = DelayedTransport(delay_seconds=0.0)

        def metrics_without_counters(endpoint, timeout):
            return smoke.HttpResponse(200, b"vllm:num_requests_running 1\n")

        timing = vmw.MixedTimingConfig(
            settling_seconds=0.02,
            measurement_seconds=0.1,
            drain_timeout_seconds=3.0,
            request_timeout_seconds=5.0,
            metrics_interval_seconds=0.02,
        )
        telemetry_config = pb.TelemetryConfig(
            metrics_endpoint="http://x/metrics",
            metrics_transport=metrics_without_counters,
            gpu_sampler=lambda: {"available": False, "error": "no gpu"},
            interval_seconds=timing.metrics_interval_seconds,
            request_timeout_seconds=5.0,
        )
        targets = self._targets(target_rho=1.0)
        summary, _ = vmw.run_mixed_load_point(
            target_rho=1.0,
            point_index=0,
            bucket_cycles=self.bucket_cycles,
            bucket_targets=targets,
            endpoint=self.endpoint,
            model=MODEL,
            timing=timing,
            transport=transport,
            client_concurrency_budget=512,
            max_scheduling_lag_p95_seconds=10.0,
            max_achieved_rate_relative_error=10.0,
            waiting_trend_window_fraction=0.2,
            phase_offset_step_seconds=0.01,
            idle_check=always_ok_idle_check,
            telemetry_config=telemetry_config,
            run_id="mixed-test",
        )
        self.assertFalse(summary["engine_token_throughput"]["available"])
        self.assertFalse(summary["run_valid"])
        self.assertTrue(
            any(
                reason.startswith("engine_token_throughput_unavailable")
                for reason in summary["invalidation_reasons"]
            )
        )
        # Secondary completion evidence exists but never rescues run_valid.
        self.assertGreaterEqual(
            sum(b["terminal_completions_in_measurement"] for b in summary["buckets"].values()),
            0,
        )

    def test_engine_estimator_available_with_real_counters(self) -> None:
        transport = DelayedTransport(delay_seconds=0.0)
        metrics_transport = IncrementingMetricsTransport()
        timing = vmw.MixedTimingConfig(
            settling_seconds=0.02,
            measurement_seconds=0.2,
            drain_timeout_seconds=3.0,
            request_timeout_seconds=5.0,
            metrics_interval_seconds=0.02,
        )
        telemetry_config = pb.TelemetryConfig(
            metrics_endpoint="http://x/metrics",
            metrics_transport=metrics_transport,
            gpu_sampler=lambda: {"available": False, "error": "no gpu"},
            interval_seconds=timing.metrics_interval_seconds,
            request_timeout_seconds=5.0,
        )
        targets = self._targets(target_rho=1.0)
        summary, _ = vmw.run_mixed_load_point(
            target_rho=1.0,
            point_index=0,
            bucket_cycles=self.bucket_cycles,
            bucket_targets=targets,
            endpoint=self.endpoint,
            model=MODEL,
            timing=timing,
            transport=transport,
            client_concurrency_budget=512,
            max_scheduling_lag_p95_seconds=10.0,
            max_achieved_rate_relative_error=0.5,
            waiting_trend_window_fraction=0.2,
            phase_offset_step_seconds=0.01,
            idle_check=always_ok_idle_check,
            telemetry_config=telemetry_config,
            run_id="mixed-test",
        )
        self.assertTrue(
            summary["engine_token_throughput"]["available"],
            msg=summary["engine_token_throughput"].get("unavailable_reason"),
        )
        self.assertGreater(
            summary["engine_token_throughput"]["total_tokens_per_second"], 0
        )
        self.assertTrue(summary["run_valid"], msg=summary["invalidation_reasons"])
        # rho_pred_target should equal 1.0 exactly (up to fp precision).
        self.assertAlmostEqual(summary["rho_pred_target"], 1.0, places=9)

    def test_outstanding_t0_t1_accounting(self) -> None:
        # Slow enough that requests admitted during settling are still
        # outstanding at T0, and requests admitted during measurement are
        # still outstanding at T1.
        transport = DelayedTransport(delay_seconds=1.0)
        timing = vmw.MixedTimingConfig(
            settling_seconds=0.05,
            measurement_seconds=0.1,
            drain_timeout_seconds=3.0,
            request_timeout_seconds=5.0,
            metrics_interval_seconds=1.0,
        )
        targets = self._targets(target_rho=1.0)
        summary, _ = vmw.run_mixed_load_point(
            target_rho=1.0,
            point_index=0,
            bucket_cycles=self.bucket_cycles,
            bucket_targets=targets,
            endpoint=self.endpoint,
            model=MODEL,
            timing=timing,
            transport=transport,
            client_concurrency_budget=512,
            max_scheduling_lag_p95_seconds=10.0,
            max_achieved_rate_relative_error=10.0,
            waiting_trend_window_fraction=0.2,
            phase_offset_step_seconds=0.01,
            idle_check=always_ok_idle_check,
            run_id="mixed-test",
        )
        self.assertGreater(summary["outstanding_at_t0"], 0)
        self.assertGreaterEqual(summary["outstanding_at_t1"], summary["outstanding_at_t0"])
        self.assertEqual(
            summary["outstanding_delta"],
            summary["outstanding_at_t1"] - summary["outstanding_at_t0"],
        )

    def test_failed_precondition_causes_zero_admission(self) -> None:
        transport = DelayedTransport(delay_seconds=0.0)

        def failing_idle_check():
            return False, "Connection refused"

        timing = vmw.MixedTimingConfig(
            settling_seconds=0.05,
            measurement_seconds=0.2,
            drain_timeout_seconds=1.0,
            request_timeout_seconds=5.0,
            metrics_interval_seconds=1.0,
        )
        targets = self._targets(target_rho=1.0)
        summary, results = vmw.run_mixed_load_point(
            target_rho=1.0,
            point_index=0,
            bucket_cycles=self.bucket_cycles,
            bucket_targets=targets,
            endpoint=self.endpoint,
            model=MODEL,
            timing=timing,
            transport=transport,
            client_concurrency_budget=512,
            max_scheduling_lag_p95_seconds=10.0,
            max_achieved_rate_relative_error=10.0,
            waiting_trend_window_fraction=0.2,
            phase_offset_step_seconds=0.01,
            idle_check=failing_idle_check,
            run_id="mixed-test",
        )
        self.assertEqual(transport.call_count, 0)
        self.assertEqual(results, [])
        self.assertTrue(summary["execution_skipped"])
        self.assertFalse(summary["run_valid"])
        self.assertIn(
            "precondition_failed:Connection refused", summary["invalidation_reasons"]
        )

    def test_previous_point_drain_failure_cascades(self) -> None:
        targets = self._targets(target_rho=1.0)
        timing = vmw.MixedTimingConfig(
            settling_seconds=0.02,
            measurement_seconds=0.05,
            drain_timeout_seconds=1.0,
            request_timeout_seconds=5.0,
            metrics_interval_seconds=1.0,
        )
        summary, results = vmw.run_mixed_load_point(
            target_rho=1.0,
            point_index=1,
            bucket_cycles=self.bucket_cycles,
            bucket_targets=targets,
            endpoint=self.endpoint,
            model=MODEL,
            timing=timing,
            transport=DelayedTransport(delay_seconds=0.0),
            client_concurrency_budget=512,
            max_scheduling_lag_p95_seconds=10.0,
            max_achieved_rate_relative_error=10.0,
            waiting_trend_window_fraction=0.2,
            phase_offset_step_seconds=0.01,
            idle_check=always_ok_idle_check,
            run_id="mixed-test",
            previous_point_drained=False,
        )
        self.assertTrue(summary["execution_skipped"])
        self.assertEqual(results, [])
        self.assertIn("previous_point_drain_failed", summary["invalidation_reasons"])
        self.assertFalse(summary["run_valid"])


# ---------------------------------------------------------------------------
# End-to-end experiment / artifact writing
# ---------------------------------------------------------------------------


class RunMixedExperimentIntegrationTests(unittest.TestCase):
    def test_full_experiment_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset = root / "profiling.jsonl"
            output_dir = root / "mixed-results"
            write_dataset(dataset, make_records(32))

            transport = DelayedTransport(delay_seconds=0.0)
            metrics_transport = IncrementingMetricsTransport()

            def probe_transport(endpoint, timeout):
                if endpoint.endswith("/v1/models"):
                    body = json.dumps({"data": [{"id": MODEL}]}).encode("utf-8")
                    return smoke.HttpResponse(200, body)
                if endpoint.endswith("/version"):
                    return smoke.HttpResponse(200, b'{"version": "0.28.0"}')
                return smoke.HttpResponse(404, b"")

            def gpu_sampler():
                return {"available": False, "error": "no gpu in unit tests"}

            def fingerprint_runner(*args, **kwargs):
                raise FileNotFoundError("nvidia-smi not present in unit tests")

            config = vmw.MixedExperimentConfig(
                profiling_jsonl=dataset,
                output_dir=output_dir,
                base_url="http://127.0.0.1:8000",
                model=MODEL,
                tokenizer_revision=REVISION,
                capacities={
                    "input-heavy": 15360.0,
                    "balanced": 15360.0,
                    "output-heavy": 15360.0,
                },
                target_rho_values=(0.5, 1.0),
                timing=vmw.MixedTimingConfig(
                    settling_seconds=0.02,
                    measurement_seconds=0.2,
                    drain_timeout_seconds=3.0,
                    request_timeout_seconds=5.0,
                    metrics_interval_seconds=0.02,
                ),
                client_concurrency_budget=512,
                run_id="mixed-e2e-test",
                transport=transport,
                probe_transport=probe_transport,
                metrics_transport=metrics_transport,
                gpu_sampler=gpu_sampler,
                fingerprint_runner=fingerprint_runner,
            )

            bundle = vmw.run_mixed_experiment(config)
            vmw.write_artifacts(output_dir, bundle)

            self.assertTrue((output_dir / vmw.MANIFEST_FILENAME).is_file())
            self.assertTrue((output_dir / vmw.REQUEST_RESULTS_FILENAME).is_file())
            self.assertTrue((output_dir / vmw.POINT_SUMMARIES_FILENAME).is_file())
            self.assertTrue((output_dir / vmw.SUMMARY_FILENAME).is_file())
            self.assertTrue((output_dir / vmw.VLLM_METRICS_FILENAME).is_file())
            self.assertTrue((output_dir / vmw.GPU_METRICS_FILENAME).is_file())

            manifest = json.loads((output_dir / vmw.MANIFEST_FILENAME).read_text("utf-8"))
            self.assertEqual(
                manifest["capacities_supplied"],
                {"input-heavy": 15360.0, "balanced": 15360.0, "output-heavy": 15360.0},
            )
            self.assertEqual(len(manifest["composition_model"]["composition_plan"]), 2)
            self.assertIn("client_concurrency_budget", manifest["load_generator"])

            point_summaries = [
                json.loads(line)
                for line in (output_dir / vmw.POINT_SUMMARIES_FILENAME)
                .read_text("utf-8")
                .splitlines()
            ]
            self.assertEqual(len(point_summaries), 2)
            self.assertAlmostEqual(point_summaries[0]["rho_pred_target"], 0.5, places=6)
            self.assertAlmostEqual(point_summaries[1]["rho_pred_target"], 1.0, places=6)

            summary = json.loads((output_dir / vmw.SUMMARY_FILENAME).read_text("utf-8"))
            self.assertEqual(len(summary["review_table"]), 2)
            expected_columns = {
                "target_rho",
                "achieved_rho",
                "total_target_req_s",
                "total_achieved_req_s",
                "engine_tok_s",
                "waiting_mean",
                "waiting_max",
                "waiting_trend_req_s",
                "waiting_first_median",
                "waiting_last_median",
                "outstanding_t0",
                "outstanding_t1",
                "outstanding_delta",
                "running_mean",
                "running_max",
                "preemptions_delta",
                "scheduling_lag_p95",
                "run_valid",
                "invalidation_reasons",
            }
            self.assertTrue(expected_columns.issubset(set(summary["review_table"][0])))
            self.assertIn("human_review_warning", summary)


if __name__ == "__main__":
    unittest.main()

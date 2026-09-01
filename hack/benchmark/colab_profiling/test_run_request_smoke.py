"""CPU-only mocked tests for the #1546 vLLM request-contract smoke client."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_dataset as generator
import run_request_smoke as smoke


MODEL = "Qwen/Qwen2.5-3B"
REVISION = "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"


def source_records(records_per_bucket: int = 3) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record_index in range(records_per_bucket):
        for bucket_index, bucket in enumerate(generator.DEFAULT_BUCKETS):
            prompt_ids = [
                1000 + bucket_index * 1000 + record_index * bucket.input_tokens + offset
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


class RecordingTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], float]] = []

    def __call__(
        self, endpoint: str, payload: Mapping[str, Any], timeout: float
    ) -> smoke.HttpResponse:
        copied_payload = copy.deepcopy(dict(payload))
        self.calls.append((endpoint, copied_payload, timeout))
        return successful_response(copied_payload)


class IncrementingClock:
    def __init__(self, start: int = 1_000_000, step: int = 25) -> None:
        self.value = start - step
        self.step = step

    def __call__(self) -> int:
        self.value += self.step
        return self.value


def write_dataset(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def valid_response_object(record: Mapping[str, Any]) -> dict[str, Any]:
    target = record["target_output_tokens"]
    return {
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "text": "",
                "finish_reason": "length",
                "prompt_token_ids": list(record["prompt_token_ids"]),
                "token_ids": list(range(target)),
            }
        ],
        "usage": {
            "prompt_tokens": record["prompt_token_count"],
            "completion_tokens": target,
            "total_tokens": record["prompt_token_count"] + target,
        },
    }


class RequestSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.record = source_records(1)[0]

    def failure_reasons(self, response: dict[str, Any]) -> set[str]:
        _, failures = smoke.validate_response(response, self.record, MODEL)
        return {failure["reason"] for failure in failures}

    def test_successful_bounded_sequential_smoke_and_exact_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset = root / "profiling.jsonl"
            output_dir = root / "smoke-results"
            write_dataset(dataset, source_records())
            transport = RecordingTransport()

            results, summary = smoke.execute_smoke(
                profiling_jsonl=dataset,
                output_dir=output_dir,
                base_url="http://127.0.0.1:8000/",
                model=MODEL,
                tokenizer_revision=REVISION,
                transport=transport,
                clock_ns=IncrementingClock(),
            )

            expected_ids = [
                f"profiling.{bucket.name}.{record_index:04d}"
                for bucket in generator.DEFAULT_BUCKETS
                for record_index in range(2)
            ]
            expected_source_records = smoke.select_records(source_records(), 2)
            self.assertEqual([result["request_id"] for result in results], expected_ids)
            self.assertEqual(
                summary["selection"]["selected_request_ids"], expected_ids
            )
            self.assertEqual(len(transport.calls), 6)
            self.assertTrue(summary["overall_pass"])
            self.assertEqual(summary["requests_passed"], 6)
            self.assertEqual(summary["requests_failed"], 0)
            self.assertEqual(summary["failures_by_reason"], {})
            self.assertEqual(
                summary["timing_scope"], "audit_only_not_throughput_or_profiling"
            )
            self.assertTrue((output_dir / smoke.RESULTS_FILENAME).is_file())
            self.assertTrue((output_dir / smoke.SUMMARY_FILENAME).is_file())
            written_results = [
                json.loads(line)
                for line in (output_dir / smoke.RESULTS_FILENAME)
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(written_results), 6)
            self.assertTrue(all(result["passed"] for result in written_results))
            self.assertTrue(
                all(
                    result["schema_version"] == smoke.RESULT_SCHEMA_VERSION
                    for result in written_results
                )
            )

            expected_keys = {
                "ignore_eos",
                "max_tokens",
                "min_tokens",
                "model",
                "n",
                "prompt",
                "return_token_ids",
                "stream",
                "temperature",
            }
            for call, result, source_record in zip(
                transport.calls, results, expected_source_records
            ):
                endpoint, payload, timeout = call
                self.assertEqual(endpoint, "http://127.0.0.1:8000/v1/completions")
                self.assertEqual(timeout, smoke.DEFAULT_TIMEOUT_SECONDS)
                self.assertEqual(set(payload), expected_keys)
                self.assertIsInstance(payload["prompt"], list)
                self.assertTrue(all(isinstance(token, int) for token in payload["prompt"]))
                self.assertEqual(payload["prompt"], source_record["prompt_token_ids"])
                self.assertEqual(payload["model"], MODEL)
                self.assertEqual(payload["n"], 1)
                self.assertIs(payload["stream"], False)
                self.assertEqual(payload["temperature"], 0)
                self.assertEqual(payload["max_tokens"], result["target_output_tokens"])
                self.assertEqual(payload["min_tokens"], result["target_output_tokens"])
                self.assertIs(payload["ignore_eos"], True)
                self.assertIs(payload["return_token_ids"], True)
                self.assertEqual(result["latency_ns"], 25)

    def test_wrong_prompt_token_count_fails(self) -> None:
        response = valid_response_object(self.record)
        response["usage"]["prompt_tokens"] += 1
        self.assertIn("prompt_token_count", self.failure_reasons(response))

    def test_wrong_completion_token_count_fails(self) -> None:
        response = valid_response_object(self.record)
        response["usage"]["completion_tokens"] -= 1
        self.assertIn("completion_token_count", self.failure_reasons(response))

    def test_wrong_total_token_count_fails(self) -> None:
        response = valid_response_object(self.record)
        response["usage"]["total_tokens"] += 1
        self.assertIn("total_token_count", self.failure_reasons(response))

    def test_non_length_finish_reason_fails(self) -> None:
        response = valid_response_object(self.record)
        response["choices"][0]["finish_reason"] = "stop"
        self.assertIn("finish_reason", self.failure_reasons(response))

    def test_http_failure_is_a_failed_audit_result(self) -> None:
        def transport(
            endpoint: str, payload: Mapping[str, Any], timeout: float
        ) -> smoke.HttpResponse:
            return smoke.HttpResponse(503, b"")

        result = smoke.execute_record(
            self.record,
            "http://127.0.0.1:8000/v1/completions",
            MODEL,
            10,
            transport=transport,
            clock_ns=IncrementingClock(),
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["http_status"], 503)
        self.assertEqual(result["failure_reasons"][0]["reason"], "http_status")

    def test_malformed_json_and_transport_error_fail(self) -> None:
        malformed = smoke.execute_record(
            self.record,
            "http://example/v1/completions",
            MODEL,
            10,
            transport=lambda endpoint, payload, timeout: smoke.HttpResponse(200, b"{"),
            clock_ns=IncrementingClock(),
        )
        self.assertEqual(malformed["failure_reasons"][0]["reason"], "response_json")

        def transport_error(
            endpoint: str, payload: Mapping[str, Any], timeout: float
        ) -> smoke.HttpResponse:
            raise smoke.TransportError("connection refused")

        failed_transport = smoke.execute_record(
            self.record,
            "http://example/v1/completions",
            MODEL,
            10,
            transport=transport_error,
            clock_ns=IncrementingClock(),
        )
        self.assertEqual(
            failed_transport["failure_reasons"][0]["reason"], "http_transport"
        )

    def test_duplicate_response_json_keys_fail(self) -> None:
        result = smoke.execute_record(
            self.record,
            "http://example/v1/completions",
            MODEL,
            10,
            transport=lambda endpoint, payload, timeout: smoke.HttpResponse(
                200, b'{"model":"first","model":"second"}'
            ),
            clock_ns=IncrementingClock(),
        )
        self.assertEqual(result["failure_reasons"][0]["reason"], "response_json")

    def test_exactly_one_choice_and_model_identity_are_required(self) -> None:
        response = valid_response_object(self.record)
        response["choices"].append(copy.deepcopy(response["choices"][0]))
        self.assertIn("choice_count", self.failure_reasons(response))

        response = valid_response_object(self.record)
        response["model"] = "wrong/model"
        self.assertIn("response_model", self.failure_reasons(response))

    def test_vllm_choice_level_token_id_evidence_is_validated(self) -> None:
        response = valid_response_object(self.record)
        response["choices"][0]["prompt_token_ids"][0] += 1
        self.assertIn("prompt_token_ids_content", self.failure_reasons(response))

        response = valid_response_object(self.record)
        response["choices"][0]["token_ids"].pop()
        self.assertIn("output_token_ids_length", self.failure_reasons(response))

        response = valid_response_object(self.record)
        response["choices"][0]["token_ids"] = "not-an-array"
        self.assertIn("token_ids_shape", self.failure_reasons(response))

    def test_omitted_optional_token_id_evidence_is_allowed(self) -> None:
        response = valid_response_object(self.record)
        del response["choices"][0]["prompt_token_ids"]
        del response["choices"][0]["token_ids"]
        evidence, failures = smoke.validate_response(response, self.record, MODEL)
        self.assertEqual(failures, [])
        self.assertIsNone(evidence["returned_prompt_token_ids"])
        self.assertIsNone(evidence["returned_output_token_ids"])

    def test_source_data_and_output_location_fail_closed_before_http(self) -> None:
        records = source_records()
        records[0]["prompt_token_count"] += 1
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset = root / "profiling.jsonl"
            write_dataset(dataset, records)
            transport = RecordingTransport()
            with self.assertRaisesRegex(smoke.SmokeError, "prompt_token_count"):
                smoke.execute_smoke(
                    dataset,
                    root / "results",
                    "http://127.0.0.1:8000",
                    MODEL,
                    REVISION,
                    transport=transport,
                )
            self.assertEqual(transport.calls, [])
            self.assertFalse((root / "results").exists())

            existing_output = root / "existing"
            existing_output.mkdir()
            with self.assertRaisesRegex(smoke.SmokeError, "already exists"):
                smoke.execute_smoke(
                    dataset,
                    existing_output,
                    "http://127.0.0.1:8000",
                    MODEL,
                    REVISION,
                    transport=transport,
                )

    def test_failed_request_run_publishes_an_explicit_failed_summary(self) -> None:
        class OneFailureTransport(RecordingTransport):
            def __call__(
                self, endpoint: str, payload: Mapping[str, Any], timeout: float
            ) -> smoke.HttpResponse:
                response = super().__call__(endpoint, payload, timeout)
                if len(self.calls) == 1:
                    decoded = json.loads(response.body)
                    decoded["usage"]["completion_tokens"] -= 1
                    return smoke.HttpResponse(
                        200, json.dumps(decoded).encode("utf-8")
                    )
                return response

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset = root / "profiling.jsonl"
            output_dir = root / "smoke-results"
            write_dataset(dataset, source_records())
            results, summary = smoke.execute_smoke(
                dataset,
                output_dir,
                "http://127.0.0.1:8000",
                MODEL,
                REVISION,
                transport=OneFailureTransport(),
                clock_ns=IncrementingClock(),
            )
            self.assertEqual(len(results), 6)
            self.assertFalse(summary["overall_pass"])
            self.assertEqual(summary["requests_attempted"], 6)
            self.assertEqual(summary["requests_passed"], 5)
            self.assertEqual(summary["requests_failed"], 1)
            self.assertEqual(
                summary["failures_by_reason"],
                {"completion_token_count": 1, "total_token_count": 1},
            )
            written_summary = json.loads(
                (output_dir / smoke.SUMMARY_FILENAME).read_text(encoding="utf-8")
            )
            self.assertFalse(written_summary["overall_pass"])

    def test_selection_is_bounded(self) -> None:
        records = source_records()
        selected = smoke.select_records(records, 2)
        self.assertEqual(len(selected), 6)
        with self.assertRaises(smoke.SmokeError):
            smoke.select_records(records, smoke.MAX_RECORDS_PER_BUCKET + 1)


if __name__ == "__main__":
    unittest.main()

"""CPU-only tests for the ``profile_balanced_bucket.py`` compatibility wrapper.

The full profiling methodology is exercised by ``test_profile_bucket.py``
against the generic implementation in ``profile_bucket.py``. These tests
only check that the backward-compatible wrapper still behaves exactly like
the original balanced-only profiler: no ``--bucket`` flag, always profiles
``balanced``, and produces the same artifact shape end to end.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_dataset as generator
import profile_balanced_bucket as profiler
import run_request_smoke as smoke


MODEL = "Qwen/Qwen2.5-3B"
REVISION = "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"


def make_records(records_per_bucket: int = 4) -> list[dict[str, Any]]:
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


class CompatibilityCliTests(unittest.TestCase):
    def test_bucket_flag_is_not_exposed(self) -> None:
        parser = profiler.build_argument_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--profiling-jsonl",
                    "x.jsonl",
                    "--output-dir",
                    "out",
                    "--bucket",
                    "balanced",
                    "--model",
                    MODEL,
                    "--tokenizer-revision",
                    REVISION,
                ]
            )

    def test_default_run_id_prefix_is_balanced(self) -> None:
        self.assertTrue(profiler.default_run_id().startswith("balanced-"))

    def test_balanced_bucket_constant_matches_generic_module(self) -> None:
        self.assertEqual(profiler.BALANCED_BUCKET.name, "balanced")
        self.assertEqual(profiler.BALANCED_BUCKET.input_tokens, 256)
        self.assertEqual(profiler.BALANCED_BUCKET.target_output_tokens, 256)

    def test_select_balanced_records_matches_generic_selection(self) -> None:
        records = make_records()
        selected = profiler.select_balanced_records(records)
        self.assertTrue(selected)
        self.assertTrue(all(r["bucket"] == "balanced" for r in selected))


class CompatibilityEndToEndTests(unittest.TestCase):
    def test_main_runs_the_balanced_bucket_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset = root / "profiling.jsonl"
            output_dir = root / "balanced-results"
            write_dataset(dataset, make_records(8))

            # execute through the public config/run/write path used by main(),
            # with fakes injected the same way the generic test suite does.
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

            def transport(endpoint, payload, timeout):
                return successful_response(payload)

            config = profiler.ExperimentConfig(
                profiling_jsonl=dataset,
                output_dir=output_dir,
                base_url="http://127.0.0.1:8000",
                model=MODEL,
                tokenizer_revision=REVISION,
                bucket=profiler.BALANCED_BUCKET_NAME,
                concurrency_ladder=(1, 2),
                timing=profiler.TimingConfig(
                    settling_seconds=0.02,
                    measurement_seconds=0.05,
                    drain_timeout_seconds=1.0,
                    metrics_interval_seconds=0.02,
                    request_timeout_seconds=5.0,
                ),
                run_id="compat-test",
                transport=transport,
                probe_transport=probe_transport,
                metrics_transport=metrics_transport,
                gpu_sampler=gpu_sampler,
                fingerprint_runner=fingerprint_runner,
            )

            exit_code = profiler.run_cli(config)
            self.assertEqual(exit_code, 0)

            manifest = json.loads(
                (output_dir / profiler.MANIFEST_FILENAME).read_text("utf-8")
            )
            self.assertEqual(manifest["dataset"]["bucket_definition"]["name"], "balanced")
            self.assertIn("VALIDATED", manifest["bucket_validation_status"])

            point_summaries = [
                json.loads(line)
                for line in (output_dir / profiler.POINT_SUMMARIES_FILENAME)
                .read_text("utf-8")
                .splitlines()
            ]
            self.assertTrue(all(p["bucket"] == "balanced" for p in point_summaries))
            self.assertTrue(all(p["token_rate_invariant"]["holds"] for p in point_summaries))


if __name__ == "__main__":
    unittest.main()

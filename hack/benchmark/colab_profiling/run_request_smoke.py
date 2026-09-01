#!/usr/bin/env python3
"""Run the bounded #1546 vLLM request-contract smoke test."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from generate_dataset import DEFAULT_BUCKETS, SCHEMA_VERSION, prompt_hash


RESULT_SCHEMA_VERSION = "llm-d-vllm-request-contract-smoke-result-v1"
SUMMARY_SCHEMA_VERSION = "llm-d-vllm-request-contract-smoke-summary-v1"
RESULTS_FILENAME = "request_results.jsonl"
SUMMARY_FILENAME = "summary.json"
DEFAULT_RECORDS_PER_BUCKET = 2
MAX_RECORDS_PER_BUCKET = 10
DEFAULT_TIMEOUT_SECONDS = 600.0
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


class SmokeError(ValueError):
    """Raised when smoke setup or source data violates the contract."""


class TransportError(RuntimeError):
    """Raised when an HTTP request cannot produce a bounded response."""


@dataclass(frozen=True)
class HttpResponse:
    """The bounded HTTP evidence used by response validation."""

    status_code: int
    body: bytes


Transport = Callable[[str, Mapping[str, Any], float], HttpResponse]
Clock = Callable[[], int]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def completions_endpoint(base_url: str) -> str:
    """Validate a server root URL and append the one authorized endpoint."""

    try:
        parsed = urllib.parse.urlsplit(base_url)
    except ValueError as error:
        raise SmokeError("base URL is malformed") from error
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise SmokeError("base URL must be an absolute http(s) server root")
    if parsed.hostname is None:
        raise SmokeError("base URL must contain a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise SmokeError("credentials must not be embedded in the base URL")
    if parsed.query or parsed.fragment:
        raise SmokeError("base URL must not contain a query or fragment")
    if parsed.path not in ("", "/"):
        raise SmokeError("base URL must be a server root without a path")
    return base_url.rstrip("/") + "/v1/completions"


def _checked_nonnegative_int(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SmokeError(f"{description} must be a non-negative integer")
    return value


def _validate_source_record(
    record: Any,
    line_number: int,
    model: str,
    tokenizer_revision: str,
) -> dict[str, Any]:
    label = f"profiling JSONL line {line_number}"
    if not isinstance(record, dict):
        raise SmokeError(f"{label} must be a JSON object")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise SmokeError(f"{label} has an unsupported schema_version")
    if record.get("split") != "profiling":
        raise SmokeError(f"{label} is not a profiling record")

    request_id = record.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise SmokeError(f"{label} has no request_id")

    bucket_name = record.get("bucket")
    bucket_by_name = {bucket.name: bucket for bucket in DEFAULT_BUCKETS}
    bucket = bucket_by_name.get(bucket_name)
    if bucket is None:
        raise SmokeError(f"{label} has unapproved bucket {bucket_name!r}")

    token_ids = record.get("prompt_token_ids")
    if not isinstance(token_ids, list):
        raise SmokeError(f"{label} prompt_token_ids must be an array")
    checked_token_ids = [
        _checked_nonnegative_int(token_id, f"{label} prompt token ID")
        for token_id in token_ids
    ]
    if len(checked_token_ids) != bucket.input_tokens:
        raise SmokeError(
            f"{label} has {len(checked_token_ids)} prompt tokens; "
            f"expected {bucket.input_tokens}"
        )
    if record.get("prompt_token_count") != len(checked_token_ids):
        raise SmokeError(f"{label} prompt_token_count does not match token IDs")
    if record.get("prompt_hash") != prompt_hash(checked_token_ids):
        raise SmokeError(f"{label} prompt_hash does not match token IDs")
    if record.get("target_output_tokens") != bucket.target_output_tokens:
        raise SmokeError(f"{label} target_output_tokens does not match its bucket")
    if record.get("total_target_tokens") != bucket.total_target_tokens:
        raise SmokeError(f"{label} total_target_tokens does not match its bucket")
    if record.get("model_id") != model:
        raise SmokeError(f"{label} model_id does not match --model")
    if record.get("tokenizer_revision") != tokenizer_revision:
        raise SmokeError(
            f"{label} tokenizer_revision does not match --tokenizer-revision"
        )
    if record.get("server_reported_completion_tokens") is not None:
        raise SmokeError(
            f"{label} is not clean source data: server completion usage is populated"
        )
    _checked_nonnegative_int(record.get("generator_seed"), f"{label} generator_seed")
    return record


def load_profiling_records(
    path: Path,
    model: str,
    tokenizer_revision: str,
) -> tuple[list[dict[str, Any]], str]:
    """Read and validate the complete profiling JSONL before any HTTP request."""

    try:
        payload = path.read_bytes()
    except OSError as error:
        raise SmokeError(f"could not read profiling JSONL {path}: {error}") from error
    if not payload:
        raise SmokeError("profiling JSONL is empty")

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SmokeError("profiling JSONL is not valid UTF-8") from error

    records: list[dict[str, Any]] = []
    request_ids: set[str] = set()
    prompt_hashes: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise SmokeError(f"profiling JSONL line {line_number} is blank")
        try:
            decoded = _strict_json_loads(line)
        except (json.JSONDecodeError, ValueError) as error:
            raise SmokeError(
                f"profiling JSONL line {line_number} is not strict JSON"
            ) from error
        record = _validate_source_record(
            decoded, line_number, model, tokenizer_revision
        )
        if record["request_id"] in request_ids:
            raise SmokeError(f"duplicate request_id {record['request_id']!r}")
        if record["prompt_hash"] in prompt_hashes:
            raise SmokeError(f"duplicate prompt_hash {record['prompt_hash']!r}")
        request_ids.add(record["request_id"])
        prompt_hashes.add(record["prompt_hash"])
        records.append(record)

    return records, _sha256(payload)


def select_records(
    records: Sequence[Mapping[str, Any]],
    records_per_bucket: int,
) -> list[Mapping[str, Any]]:
    """Select the first N records per bucket in the approved bucket order."""

    if (
        isinstance(records_per_bucket, bool)
        or not isinstance(records_per_bucket, int)
        or records_per_bucket <= 0
        or records_per_bucket > MAX_RECORDS_PER_BUCKET
    ):
        raise SmokeError(
            f"records per bucket must be between 1 and {MAX_RECORDS_PER_BUCKET}"
        )

    selected: list[Mapping[str, Any]] = []
    for bucket in DEFAULT_BUCKETS:
        bucket_records = [
            record for record in records if record.get("bucket") == bucket.name
        ]
        if len(bucket_records) < records_per_bucket:
            raise SmokeError(
                f"bucket {bucket.name!r} has {len(bucket_records)} records; "
                f"need {records_per_bucket}"
            )
        selected.extend(bucket_records[:records_per_bucket])
    return selected


def request_payload(record: Mapping[str, Any], model: str) -> dict[str, Any]:
    """Build the exact D18 request without decoding the prompt token IDs."""

    target_output_tokens = record["target_output_tokens"]
    return {
        "ignore_eos": True,
        "max_tokens": target_output_tokens,
        "min_tokens": target_output_tokens,
        "model": model,
        "n": 1,
        "prompt": list(record["prompt_token_ids"]),
        "return_token_ids": True,
        "stream": False,
        "temperature": 0,
    }


def post_completion(
    endpoint: str,
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> HttpResponse:
    """POST one non-streaming completion using only the Python standard library."""

    request = urllib.request.Request(
        endpoint,
        data=_canonical_json_bytes(payload),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
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
        raise TransportError(f"HTTP transport failed: {error}") from error


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    decoded: dict[str, Any] = {}
    for key, value in pairs:
        if key in decoded:
            raise ValueError(f"duplicate JSON object key {key!r}")
        decoded[key] = value
    return decoded


def _strict_json_loads(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_json_constant,
    )


def _decode_response_json(body: bytes) -> Any:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SmokeError("response body is not valid UTF-8") from error
    try:
        return _strict_json_loads(text)
    except (json.JSONDecodeError, ValueError) as error:
        raise SmokeError("response body is not strict JSON") from error


def _response_int(
    container: Mapping[str, Any],
    field: str,
    context: str,
    failures: list[dict[str, str]],
) -> int | None:
    value = container.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        failures.append(
            {
                "reason": "response_shape",
                "detail": f"{context}.{field} must be a non-negative integer",
            }
        )
        return None
    return value


def _optional_token_ids(
    choice: Mapping[str, Any],
    field: str,
    failures: list[dict[str, str]],
) -> list[int] | None:
    if field not in choice or choice[field] is None:
        return None
    value = choice[field]
    if not isinstance(value, list) or any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for token_id in value
    ):
        failures.append(
            {
                "reason": f"{field}_shape",
                "detail": f"choices[0].{field} is present but is not an integer array",
            }
        )
        return None
    return list(value)


def validate_response(
    response: Any,
    record: Mapping[str, Any],
    expected_model: str,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Extract audit evidence and return every response-contract failure."""

    evidence: dict[str, Any] = {
        "response_model": None,
        "server_reported_prompt_tokens": None,
        "server_reported_completion_tokens": None,
        "server_reported_total_tokens": None,
        "finish_reason": None,
        "returned_prompt_token_ids": None,
        "returned_output_token_ids": None,
    }
    failures: list[dict[str, str]] = []
    if not isinstance(response, dict):
        return evidence, [
            {
                "reason": "response_shape",
                "detail": "response root must be a JSON object",
            }
        ]

    response_model = response.get("model")
    if isinstance(response_model, str):
        evidence["response_model"] = response_model
    if response_model != expected_model:
        failures.append(
            {
                "reason": "response_model",
                "detail": f"response model {response_model!r} != {expected_model!r}",
            }
        )

    usage = response.get("usage")
    if not isinstance(usage, dict):
        failures.append(
            {"reason": "response_shape", "detail": "usage must be an object"}
        )
    else:
        prompt_tokens = _response_int(usage, "prompt_tokens", "usage", failures)
        completion_tokens = _response_int(
            usage, "completion_tokens", "usage", failures
        )
        total_tokens = _response_int(usage, "total_tokens", "usage", failures)
        evidence["server_reported_prompt_tokens"] = prompt_tokens
        evidence["server_reported_completion_tokens"] = completion_tokens
        evidence["server_reported_total_tokens"] = total_tokens

        if (
            prompt_tokens is not None
            and prompt_tokens != record["prompt_token_count"]
        ):
            failures.append(
                {
                    "reason": "prompt_token_count",
                    "detail": (
                        f"usage.prompt_tokens {prompt_tokens} != "
                        f"expected {record['prompt_token_count']}"
                    ),
                }
            )
        if (
            completion_tokens is not None
            and completion_tokens != record["target_output_tokens"]
        ):
            failures.append(
                {
                    "reason": "completion_token_count",
                    "detail": (
                        f"usage.completion_tokens {completion_tokens} != "
                        f"target {record['target_output_tokens']}"
                    ),
                }
            )
        if (
            prompt_tokens is not None
            and completion_tokens is not None
            and total_tokens is not None
            and total_tokens != prompt_tokens + completion_tokens
        ):
            failures.append(
                {
                    "reason": "total_token_count",
                    "detail": (
                        f"usage.total_tokens {total_tokens} != "
                        f"{prompt_tokens} + {completion_tokens}"
                    ),
                }
            )

    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        failures.append(
            {
                "reason": "choice_count",
                "detail": "response must contain exactly one choice",
            }
        )
        return evidence, failures
    choice = choices[0]
    if not isinstance(choice, dict):
        failures.append(
            {"reason": "response_shape", "detail": "choices[0] must be an object"}
        )
        return evidence, failures
    choice_index = choice.get("index")
    if (
        isinstance(choice_index, bool)
        or not isinstance(choice_index, int)
        or choice_index != 0
    ):
        failures.append(
            {"reason": "choice_index", "detail": "choices[0].index must be 0"}
        )

    finish_reason = choice.get("finish_reason")
    if isinstance(finish_reason, str):
        evidence["finish_reason"] = finish_reason
    if finish_reason != "length":
        failures.append(
            {
                "reason": "finish_reason",
                "detail": f"finish_reason {finish_reason!r} != 'length'",
            }
        )

    returned_prompt_ids = _optional_token_ids(
        choice, "prompt_token_ids", failures
    )
    returned_output_ids = _optional_token_ids(choice, "token_ids", failures)
    evidence["returned_prompt_token_ids"] = returned_prompt_ids
    evidence["returned_output_token_ids"] = returned_output_ids

    if returned_prompt_ids is not None:
        if len(returned_prompt_ids) != record["prompt_token_count"]:
            failures.append(
                {
                    "reason": "prompt_token_ids_length",
                    "detail": (
                        f"returned prompt_token_ids length {len(returned_prompt_ids)} != "
                        f"expected {record['prompt_token_count']}"
                    ),
                }
            )
        elif returned_prompt_ids != record["prompt_token_ids"]:
            failures.append(
                {
                    "reason": "prompt_token_ids_content",
                    "detail": "returned prompt_token_ids differ from the submitted IDs",
                }
            )
    if (
        returned_output_ids is not None
        and len(returned_output_ids) != record["target_output_tokens"]
    ):
        failures.append(
            {
                "reason": "output_token_ids_length",
                "detail": (
                    f"returned token_ids length {len(returned_output_ids)} != "
                    f"target {record['target_output_tokens']}"
                ),
            }
        )

    return evidence, failures


def execute_record(
    record: Mapping[str, Any],
    endpoint: str,
    model: str,
    timeout_seconds: float,
    transport: Transport = post_completion,
    clock_ns: Clock = time.perf_counter_ns,
) -> dict[str, Any]:
    """Execute one request and retain a fail-closed audit result."""

    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "request_id": record["request_id"],
        "bucket": record["bucket"],
        "prompt_hash": record["prompt_hash"],
        "expected_prompt_tokens": record["prompt_token_count"],
        "server_reported_prompt_tokens": None,
        "target_output_tokens": record["target_output_tokens"],
        "server_reported_completion_tokens": None,
        "server_reported_total_tokens": None,
        "finish_reason": None,
        "http_status": None,
        "result_status": "failed",
        "passed": False,
        "failure_reasons": [],
        "response_model": None,
        "returned_prompt_token_ids": None,
        "returned_output_token_ids": None,
        "submit_monotonic_ns": None,
        "terminal_monotonic_ns": None,
        "latency_ns": None,
    }
    payload = request_payload(record, model)
    submit_ns = clock_ns()
    result["submit_monotonic_ns"] = submit_ns
    try:
        response = transport(endpoint, payload, timeout_seconds)
        result["http_status"] = response.status_code
        if response.status_code != 200:
            result["failure_reasons"] = [
                {
                    "reason": "http_status",
                    "detail": f"expected HTTP 200, got {response.status_code}",
                }
            ]
        else:
            try:
                decoded = _decode_response_json(response.body)
            except SmokeError as error:
                result["failure_reasons"] = [
                    {"reason": "response_json", "detail": str(error)}
                ]
            else:
                evidence, failures = validate_response(decoded, record, model)
                result.update(evidence)
                result["failure_reasons"] = failures
    except (TransportError, OSError) as error:
        result["failure_reasons"] = [
            {"reason": "http_transport", "detail": str(error)}
        ]
    finally:
        terminal_ns = clock_ns()
        result["terminal_monotonic_ns"] = terminal_ns
        result["latency_ns"] = terminal_ns - submit_ns

    if not result["failure_reasons"]:
        result["passed"] = True
        result["result_status"] = "passed"
    return result


def run_selected_records(
    selected_records: Sequence[Mapping[str, Any]],
    endpoint: str,
    model: str,
    timeout_seconds: float,
    transport: Transport = post_completion,
    clock_ns: Clock = time.perf_counter_ns,
) -> list[dict[str, Any]]:
    """Execute the bounded sample sequentially, with no concurrency machinery."""

    return [
        execute_record(
            record,
            endpoint,
            model,
            timeout_seconds,
            transport=transport,
            clock_ns=clock_ns,
        )
        for record in selected_records
    ]


def build_summary(
    results: Sequence[Mapping[str, Any]],
    selected_records: Sequence[Mapping[str, Any]],
    source_dataset: Path,
    source_dataset_sha256: str,
    endpoint: str,
    model: str,
    tokenizer_revision: str,
    records_per_bucket: int,
    results_sha256: str,
) -> dict[str, Any]:
    failure_counts = Counter(
        failure["reason"]
        for result in results
        for failure in result["failure_reasons"]
    )
    passed = sum(1 for result in results if result["passed"])
    failed = len(results) - passed
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "overall_pass": failed == 0,
        "requests_attempted": len(results),
        "requests_passed": passed,
        "requests_failed": failed,
        "failures_by_reason": dict(sorted(failure_counts.items())),
        "execution_mode": "sequential_concurrency_1",
        "timing_scope": "audit_only_not_throughput_or_profiling",
        "endpoint": endpoint,
        "model": model,
        "tokenizer_revision": tokenizer_revision,
        "source_dataset": str(source_dataset),
        "source_dataset_sha256": source_dataset_sha256,
        "results_artifact": RESULTS_FILENAME,
        "results_sha256": results_sha256,
        "selection": {
            "strategy": "first_n_per_bucket_in_approved_order",
            "records_per_bucket": records_per_bucket,
            "bucket_order": [bucket.name for bucket in DEFAULT_BUCKETS],
            "selected_request_ids": [
                record["request_id"] for record in selected_records
            ],
        },
        "token_id_evidence": {
            "requests_with_returned_prompt_token_ids": sum(
                result["returned_prompt_token_ids"] is not None for result in results
            ),
            "requests_with_returned_output_token_ids": sum(
                result["returned_output_token_ids"] is not None for result in results
            ),
        },
    }


def _ensure_output_target(output_dir: Path, source_dataset: Path) -> None:
    if output_dir.exists():
        raise SmokeError(f"output directory already exists: {output_dir}")
    if output_dir.resolve() == source_dataset.resolve().parent:
        raise SmokeError("smoke output directory must be separate from source data")


def _write_output_directory(
    output_dir: Path,
    artifacts: Mapping[str, bytes],
) -> None:
    """Publish the complete result directory only after every file is durable."""

    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise SmokeError(f"output directory already exists: {output_dir}")
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=parent)
    )
    try:
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


def execute_smoke(
    profiling_jsonl: Path,
    output_dir: Path,
    base_url: str,
    model: str,
    tokenizer_revision: str,
    records_per_bucket: int = DEFAULT_RECORDS_PER_BUCKET,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    transport: Transport = post_completion,
    clock_ns: Clock = time.perf_counter_ns,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate inputs, run the bounded smoke, and atomically publish results."""

    if not model.strip():
        raise SmokeError("model must not be empty")
    if not tokenizer_revision.strip():
        raise SmokeError("tokenizer revision must not be empty")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise SmokeError("timeout seconds must be finite and positive")
    endpoint = completions_endpoint(base_url)
    _ensure_output_target(output_dir, profiling_jsonl)
    records, source_sha256 = load_profiling_records(
        profiling_jsonl, model, tokenizer_revision
    )
    selected = select_records(records, records_per_bucket)
    results = run_selected_records(
        selected,
        endpoint,
        model,
        timeout_seconds,
        transport=transport,
        clock_ns=clock_ns,
    )
    results_bytes = b"".join(
        _canonical_json_bytes(result) + b"\n" for result in results
    )
    summary = build_summary(
        results,
        selected,
        profiling_jsonl,
        source_sha256,
        endpoint,
        model,
        tokenizer_revision,
        records_per_bucket,
        _sha256(results_bytes),
    )
    summary_bytes = (
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_output_directory(
        output_dir,
        {RESULTS_FILENAME: results_bytes, SUMMARY_FILENAME: summary_bytes},
    )
    return results, summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiling-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="OpenAI-compatible server root; /v1/completions is appended",
    )
    parser.add_argument("--model", required=True, help="served model identity")
    parser.add_argument(
        "--tokenizer-revision",
        required=True,
        help="expected immutable revision recorded in the source dataset",
    )
    parser.add_argument(
        "--records-per-bucket",
        type=int,
        default=DEFAULT_RECORDS_PER_BUCKET,
        help=f"bounded sequential sample size, 1-{MAX_RECORDS_PER_BUCKET}",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="per-request HTTP timeout",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        _, summary = execute_smoke(
            profiling_jsonl=args.profiling_jsonl,
            output_dir=args.output_dir,
            base_url=args.base_url,
            model=args.model,
            tokenizer_revision=args.tokenizer_revision,
            records_per_bucket=args.records_per_bucket,
            timeout_seconds=args.timeout_seconds,
        )
    except (SmokeError, OSError) as error:
        print(f"request-contract smoke failed before completion: {error}", file=sys.stderr)
        return 1

    print(
        f"request-contract smoke: {summary['requests_passed']}/"
        f"{summary['requests_attempted']} passed"
    )
    print(f"results: {args.output_dir / RESULTS_FILENAME}")
    print(f"summary: {args.output_dir / SUMMARY_FILENAME}")
    return 0 if summary["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

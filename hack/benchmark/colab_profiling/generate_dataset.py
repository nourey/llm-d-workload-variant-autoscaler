#!/usr/bin/env python3
"""Generate deterministic token-ID request pools for #1546 Colab profiling."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "llm-d-colab-profiling-request-v1"
GENERATOR_VERSION = "1.0.0"
SPLITS = ("profiling", "heldout")
ARTIFACT_FILENAMES = {
    "profiling": "profiling.jsonl",
    "heldout": "heldout.jsonl",
}
PREFIX_UNIQUENESS_TOKENS = 16
MAX_GENERATION_ATTEMPTS_PER_PROMPT = 1000


@dataclass(frozen=True)
class Bucket:
    """A homogeneous target input/output length bucket."""

    name: str
    input_tokens: int
    target_output_tokens: int

    @property
    def total_target_tokens(self) -> int:
        return self.input_tokens + self.target_output_tokens


DEFAULT_BUCKETS = (
    Bucket("input-heavy", input_tokens=384, target_output_tokens=128),
    Bucket("balanced", input_tokens=256, target_output_tokens=256),
    Bucket("output-heavy", input_tokens=128, target_output_tokens=384),
)


@dataclass(frozen=True)
class GeneratorConfig:
    """All inputs that affect generated request records."""

    model_id: str
    requested_tokenizer_revision: str | None
    master_seed: int
    buckets: tuple[Bucket, ...]
    profiling_prompts_per_bucket: int
    heldout_prompts_per_bucket: int

    @property
    def counts(self) -> dict[str, int]:
        return {
            "profiling": self.profiling_prompts_per_bucket,
            "heldout": self.heldout_prompts_per_bucket,
        }


@dataclass(frozen=True)
class TokenizerMetadata:
    """Tokenizer identity and the validated ID sets used for generation."""

    tokenizer_id: str
    tokenizer_class: str
    requested_revision: str | None
    resolved_revision: str | None
    transformers_version: str
    vocabulary_sha256: str
    eligible_token_ids_sha256: str
    valid_token_ids: frozenset[int]
    special_token_ids: frozenset[int]
    eligible_token_ids: tuple[int, ...]

    @property
    def record_revision(self) -> str | None:
        return self.resolved_revision or self.requested_revision


class DatasetValidationError(ValueError):
    """Raised when generation or validation would violate the data contract."""


class Sha256RNG:
    """A small version-stable counter RNG with unbiased bounded sampling."""

    _DOMAIN = b"llm-d-colab-profiling-rng-v1\0"
    _MODULUS = 1 << 256

    def __init__(self, seed: int) -> None:
        if seed < 0 or seed >= 1 << 64:
            raise DatasetValidationError("derived RNG seed must fit in 64 bits")
        self._seed = seed.to_bytes(8, byteorder="big", signed=False)
        self._counter = 0

    def randbelow(self, upper_bound: int) -> int:
        if upper_bound <= 0:
            raise DatasetValidationError("RNG upper bound must be positive")
        acceptance_limit = self._MODULUS - (self._MODULUS % upper_bound)
        while True:
            counter = self._counter.to_bytes(16, byteorder="big", signed=False)
            self._counter += 1
            value = int.from_bytes(
                hashlib.sha256(self._DOMAIN + self._seed + counter).digest(),
                byteorder="big",
                signed=False,
            )
            if value < acceptance_limit:
                return value % upper_bound


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical JSON encoding used by every stable hash."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_hex(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def prompt_hash(prompt_token_ids: Sequence[int]) -> str:
    """Hash an exact token sequence without decoding or re-tokenizing it."""

    return sha256_hex(canonical_json_bytes(list(prompt_token_ids)))


def validate_config(config: GeneratorConfig) -> None:
    if not config.model_id.strip():
        raise DatasetValidationError("model_id must not be empty")
    if config.master_seed < 0:
        raise DatasetValidationError("master seed must be non-negative")
    if not config.buckets:
        raise DatasetValidationError("at least one bucket is required")
    if any(count <= 0 for count in config.counts.values()):
        raise DatasetValidationError("prompt counts must be positive")

    names: set[str] = set()
    for bucket in config.buckets:
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", bucket.name):
            raise DatasetValidationError(
                f"invalid bucket name {bucket.name!r}; use lowercase letters, digits, and hyphens"
            )
        if bucket.name in names:
            raise DatasetValidationError(f"duplicate bucket name: {bucket.name}")
        names.add(bucket.name)
        if bucket.input_tokens <= 0 or bucket.target_output_tokens <= 0:
            raise DatasetValidationError(
                f"bucket {bucket.name!r} token lengths must be positive"
            )


def _checked_token_id(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatasetValidationError(f"{description} is not a non-negative integer: {value!r}")
    return value


def tokenizer_metadata(
    tokenizer: Any,
    requested_revision: str | None,
    transformers_version: str,
) -> TokenizerMetadata:
    """Extract and validate the tokenizer vocabulary used as the ID authority."""

    vocabulary = tokenizer.get_vocab()
    if not isinstance(vocabulary, Mapping) or not vocabulary:
        raise DatasetValidationError("tokenizer.get_vocab() returned no vocabulary")

    vocabulary_entries: list[tuple[str, int]] = []
    valid_ids: set[int] = set()
    for token, raw_token_id in vocabulary.items():
        if not isinstance(token, str):
            raise DatasetValidationError(f"tokenizer vocabulary key is not text: {token!r}")
        token_id = _checked_token_id(raw_token_id, f"vocabulary ID for {token!r}")
        vocabulary_entries.append((token, token_id))
        valid_ids.add(token_id)

    special_ids: set[int] = set()
    for raw_token_id in getattr(tokenizer, "all_special_ids", ()) or ():
        special_ids.add(_checked_token_id(raw_token_id, "special token ID"))

    for raw_token_id, added_token in (
        getattr(tokenizer, "added_tokens_decoder", {}) or {}
    ).items():
        if getattr(added_token, "special", False):
            special_ids.add(_checked_token_id(raw_token_id, "special added-token ID"))

    eligible_ids = tuple(sorted(valid_ids - special_ids))
    if not eligible_ids:
        raise DatasetValidationError("tokenizer has no valid non-special token IDs")

    init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    resolved_revision = getattr(tokenizer, "_commit_hash", None)
    if not isinstance(resolved_revision, str) or not resolved_revision:
        resolved_revision = init_kwargs.get("_commit_hash")
    if not isinstance(resolved_revision, str) or not resolved_revision:
        resolved_revision = None

    tokenizer_id = getattr(tokenizer, "name_or_path", None)
    if not isinstance(tokenizer_id, str) or not tokenizer_id:
        tokenizer_id = tokenizer.__class__.__name__

    return TokenizerMetadata(
        tokenizer_id=tokenizer_id,
        tokenizer_class=tokenizer.__class__.__name__,
        requested_revision=requested_revision,
        resolved_revision=resolved_revision,
        transformers_version=transformers_version,
        vocabulary_sha256=sha256_hex(
            canonical_json_bytes(sorted(vocabulary_entries, key=lambda item: (item[1], item[0])))
        ),
        eligible_token_ids_sha256=sha256_hex(canonical_json_bytes(eligible_ids)),
        valid_token_ids=frozenset(valid_ids),
        special_token_ids=frozenset(special_ids),
        eligible_token_ids=eligible_ids,
    )


def derive_seed(master_seed: int, split: str, bucket: Bucket) -> int:
    """Derive an order-independent seed for one split and bucket."""

    material = {
        "bucket": bucket.name,
        "generator_version": GENERATOR_VERSION,
        "input_tokens": bucket.input_tokens,
        "master_seed": master_seed,
        "split": split,
        "target_output_tokens": bucket.target_output_tokens,
    }
    digest = hashlib.sha256(canonical_json_bytes(material)).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def derived_seeds(config: GeneratorConfig) -> dict[str, dict[str, int]]:
    return {
        split: {
            bucket.name: derive_seed(config.master_seed, split, bucket)
            for bucket in config.buckets
        }
        for split in SPLITS
    }


def _request_id(split: str, bucket: str, index: int, stable_prompt_hash: str) -> str:
    hash_prefix = stable_prompt_hash.removeprefix("sha256:")[:16]
    return f"{split}.{bucket}.{index:04d}.{hash_prefix}"


def generate_records(
    config: GeneratorConfig,
    metadata: TokenizerMetadata,
) -> list[dict[str, Any]]:
    """Generate records directly from eligible IDs, without text round trips."""

    validate_config(config)
    eligible_ids = metadata.eligible_token_ids
    records: list[dict[str, Any]] = []
    prompts_by_split: dict[str, set[tuple[int, ...]]] = {
        split: set() for split in SPLITS
    }
    all_prompts: set[tuple[int, ...]] = set()
    all_prefixes: set[tuple[int, ...]] = set()
    hashes_to_prompts: dict[str, tuple[int, ...]] = {}

    for split in SPLITS:
        for bucket in config.buckets:
            seed = derive_seed(config.master_seed, split, bucket)
            rng = Sha256RNG(seed)
            for index in range(config.counts[split]):
                for _ in range(MAX_GENERATION_ATTEMPTS_PER_PROMPT):
                    prompt = tuple(
                        eligible_ids[rng.randbelow(len(eligible_ids))]
                        for _ in range(bucket.input_tokens)
                    )
                    prefix = prompt[: min(PREFIX_UNIQUENESS_TOKENS, len(prompt))]
                    if prompt in all_prompts or prefix in all_prefixes:
                        continue

                    stable_prompt_hash = prompt_hash(prompt)
                    hash_owner = hashes_to_prompts.get(stable_prompt_hash)
                    if hash_owner is not None and hash_owner != prompt:
                        raise DatasetValidationError(
                            f"SHA-256 collision for prompt hash {stable_prompt_hash}"
                        )
                    break
                else:
                    raise DatasetValidationError(
                        f"could not generate a unique prompt for {split}/{bucket.name} "
                        f"after {MAX_GENERATION_ATTEMPTS_PER_PROMPT} attempts"
                    )

                prompts_by_split[split].add(prompt)
                all_prompts.add(prompt)
                all_prefixes.add(prefix)
                hashes_to_prompts[stable_prompt_hash] = prompt
                records.append(
                    {
                        "bucket": bucket.name,
                        "generator_seed": seed,
                        "model_id": config.model_id,
                        "prompt_hash": stable_prompt_hash,
                        "prompt_token_count": len(prompt),
                        "prompt_token_ids": list(prompt),
                        "request_id": _request_id(
                            split, bucket.name, index, stable_prompt_hash
                        ),
                        "schema_version": SCHEMA_VERSION,
                        "server_reported_completion_tokens": None,
                        "split": split,
                        "target_output_tokens": bucket.target_output_tokens,
                        "tokenizer_revision": metadata.record_revision,
                        "total_target_tokens": bucket.total_target_tokens,
                    }
                )

    return records


def validate_records(
    records: Sequence[Mapping[str, Any]],
    config: GeneratorConfig,
    metadata: TokenizerMetadata,
) -> None:
    """Fail closed if records violate any generation or split invariant."""

    validate_config(config)
    bucket_by_name = {bucket.name: bucket for bucket in config.buckets}
    expected_seeds = derived_seeds(config)
    observed_counts = {
        split: {bucket.name: 0 for bucket in config.buckets} for split in SPLITS
    }
    prompts_by_split: dict[str, set[tuple[int, ...]]] = {
        split: set() for split in SPLITS
    }
    request_ids: set[str] = set()
    hashes_to_prompts: dict[str, tuple[int, ...]] = {}
    prefixes: set[tuple[int, ...]] = set()

    for record_number, record in enumerate(records, start=1):
        label = f"record {record_number}"
        if record.get("schema_version") != SCHEMA_VERSION:
            raise DatasetValidationError(f"{label}: wrong schema_version")

        request_id = record.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise DatasetValidationError(f"{label}: missing request_id")
        if request_id in request_ids:
            raise DatasetValidationError(f"{label}: duplicate request_id {request_id!r}")
        request_ids.add(request_id)

        split = record.get("split")
        if split not in SPLITS:
            raise DatasetValidationError(f"{label}: invalid split {split!r}")
        bucket_name = record.get("bucket")
        bucket = bucket_by_name.get(bucket_name)
        if bucket is None:
            raise DatasetValidationError(f"{label}: invalid bucket {bucket_name!r}")

        raw_prompt = record.get("prompt_token_ids")
        if not isinstance(raw_prompt, list):
            raise DatasetValidationError(f"{label}: prompt_token_ids must be a list")
        prompt = tuple(
            _checked_token_id(token_id, f"{label} prompt token ID")
            for token_id in raw_prompt
        )
        if len(prompt) != bucket.input_tokens:
            raise DatasetValidationError(
                f"{label}: expected {bucket.input_tokens} prompt tokens, got {len(prompt)}"
            )
        if record.get("prompt_token_count") != len(prompt):
            raise DatasetValidationError(f"{label}: wrong prompt_token_count")

        invalid_ids = set(prompt) - metadata.valid_token_ids
        if invalid_ids:
            raise DatasetValidationError(
                f"{label}: invalid token IDs {sorted(invalid_ids)[:5]}"
            )
        used_special_ids = set(prompt) & metadata.special_token_ids
        if used_special_ids:
            raise DatasetValidationError(
                f"{label}: special token IDs are forbidden: {sorted(used_special_ids)[:5]}"
            )

        stable_prompt_hash = prompt_hash(prompt)
        if record.get("prompt_hash") != stable_prompt_hash:
            raise DatasetValidationError(f"{label}: prompt_hash does not match token IDs")
        hash_owner = hashes_to_prompts.get(stable_prompt_hash)
        if hash_owner is not None and hash_owner != prompt:
            raise DatasetValidationError(
                f"{label}: prompt hash collision {stable_prompt_hash}"
            )
        hashes_to_prompts[stable_prompt_hash] = prompt

        if prompt in prompts_by_split[split]:
            raise DatasetValidationError(f"{label}: duplicate prompt within {split}")
        other_split = "heldout" if split == "profiling" else "profiling"
        if prompt in prompts_by_split[other_split]:
            raise DatasetValidationError(
                f"{label}: prompt overlaps {split} and {other_split} splits"
            )
        prompts_by_split[split].add(prompt)

        prefix = prompt[: min(PREFIX_UNIQUENESS_TOKENS, len(prompt))]
        if prefix in prefixes:
            raise DatasetValidationError(
                f"{label}: repeated {len(prefix)}-token prompt prefix"
            )
        prefixes.add(prefix)

        if record.get("target_output_tokens") != bucket.target_output_tokens:
            raise DatasetValidationError(f"{label}: wrong target_output_tokens")
        if record.get("total_target_tokens") != bucket.total_target_tokens:
            raise DatasetValidationError(f"{label}: wrong total_target_tokens")
        if record.get("generator_seed") != expected_seeds[split][bucket.name]:
            raise DatasetValidationError(f"{label}: wrong generator_seed")
        if record.get("model_id") != config.model_id:
            raise DatasetValidationError(f"{label}: wrong model_id")
        if record.get("tokenizer_revision") != metadata.record_revision:
            raise DatasetValidationError(f"{label}: wrong tokenizer_revision")
        if record.get("server_reported_completion_tokens") is not None:
            raise DatasetValidationError(
                f"{label}: server_reported_completion_tokens must be null before execution"
            )

        observed_counts[split][bucket.name] += 1

    for split in SPLITS:
        for bucket in config.buckets:
            if observed_counts[split][bucket.name] != config.counts[split]:
                raise DatasetValidationError(
                    f"wrong count for {split}/{bucket.name}: "
                    f"expected {config.counts[split]}, "
                    f"got {observed_counts[split][bucket.name]}"
                )

    overlap = prompts_by_split["profiling"] & prompts_by_split["heldout"]
    if overlap:
        raise DatasetValidationError(
            f"profiling and heldout splits overlap by {len(overlap)} prompts"
        )


def _records_as_jsonl(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(record) + b"\n" for record in records)


def _split_artifacts(records: Sequence[Mapping[str, Any]]) -> dict[str, bytes]:
    return {
        ARTIFACT_FILENAMES[split]: _records_as_jsonl(
            [record for record in records if record["split"] == split]
        )
        for split in SPLITS
    }


def build_manifest(
    config: GeneratorConfig,
    metadata: TokenizerMetadata,
    artifacts: Mapping[str, bytes],
) -> dict[str, Any]:
    artifact_metadata = {
        filename: {
            "bytes": len(payload),
            "records": payload.count(b"\n"),
            "sha256": sha256_hex(payload),
        }
        for filename, payload in sorted(artifacts.items())
    }
    return {
        "artifacts": artifact_metadata,
        "bucket_definitions": {
            bucket.name: {
                "input_tokens": bucket.input_tokens,
                "target_output_tokens": bucket.target_output_tokens,
                "total_target_tokens": bucket.total_target_tokens,
            }
            for bucket in config.buckets
        },
        "counts_per_bucket_and_split": {
            split: {bucket.name: config.counts[split] for bucket in config.buckets}
            for split in SPLITS
        },
        "dataset_sha256": sha256_hex(canonical_json_bytes(artifact_metadata)),
        "generator_version": GENERATOR_VERSION,
        "model_id": config.model_id,
        "output_length_contract": {
            "acceptance_rule": (
                "server_reported_completion_tokens == target_output_tokens"
            ),
            "future_request_controls": {
                "ignore_eos": True,
                "max_tokens": "target_output_tokens",
                "min_tokens": "target_output_tokens",
            },
            "server_usage_source": "usage.completion_tokens",
            "server_reported_field": "server_reported_completion_tokens",
            "target_field": "target_output_tokens",
        },
        "prompt_construction": {
            "eligible_id_sampling": (
                "uniform_with_replacement_using_sha256_counter_rng_v1"
            ),
            "fixed_or_shared_prefix": False,
            "prompt_hash": (
                "SHA-256 of UTF-8 canonical JSON prompt_token_ids, encoded as sha256:<hex>"
            ),
            "prefix_uniqueness_tokens": PREFIX_UNIQUENESS_TOKENS,
            "text_round_trip": False,
        },
        "schema_version": SCHEMA_VERSION,
        "seeds": {
            "derived_by_split_and_bucket": derived_seeds(config),
            "derivation": "first 64 bits of SHA-256 over canonical generation coordinates",
            "master": config.master_seed,
        },
        "split_artifacts": ARTIFACT_FILENAMES,
        "tokenizer": {
            "class": metadata.tokenizer_class,
            "eligible_non_special_token_count": len(metadata.eligible_token_ids),
            "eligible_token_ids_sha256": metadata.eligible_token_ids_sha256,
            "id": metadata.tokenizer_id,
            "requested_revision": metadata.requested_revision,
            "resolved_revision": metadata.resolved_revision,
            "special_token_ids": sorted(metadata.special_token_ids),
            "transformers_version": metadata.transformers_version,
            "valid_token_id_count": len(metadata.valid_token_ids),
            "vocabulary_sha256": metadata.vocabulary_sha256,
        },
    }


def _write_artifacts(output_dir: Path, artifacts: Mapping[str, bytes]) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise DatasetValidationError(f"output path is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.iterdir())
    if existing:
        raise DatasetValidationError(
            f"output directory must be empty; found {existing[0].name!r}"
        )

    temporary_paths: list[tuple[Path, Path]] = []
    try:
        for filename, payload in artifacts.items():
            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{filename}.", suffix=".tmp", dir=output_dir
            )
            temporary_path = Path(temporary_name)
            temporary_paths.append((temporary_path, output_dir / filename))
            with os.fdopen(file_descriptor, "wb") as output_file:
                output_file.write(payload)
                output_file.flush()
                os.fsync(output_file.fileno())

        for temporary_path, final_path in temporary_paths:
            os.replace(temporary_path, final_path)
    finally:
        for temporary_path, _ in temporary_paths:
            temporary_path.unlink(missing_ok=True)


def generate_dataset(
    output_dir: Path,
    config: GeneratorConfig,
    metadata: TokenizerMetadata,
) -> dict[str, Any]:
    """Generate, regenerate for determinism, validate, and write all artifacts."""

    records = generate_records(config, metadata)
    validate_records(records, config, metadata)
    artifacts = _split_artifacts(records)

    regenerated_records = generate_records(config, metadata)
    validate_records(regenerated_records, config, metadata)
    regenerated_artifacts = _split_artifacts(regenerated_records)
    if artifacts != regenerated_artifacts:
        raise DatasetValidationError(
            "deterministic regeneration produced different JSONL artifacts"
        )

    manifest = build_manifest(config, metadata, artifacts)
    complete_artifacts = dict(artifacts)
    complete_artifacts["manifest.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_artifacts(output_dir, complete_artifacts)
    return manifest


def parse_bucket(value: str) -> Bucket:
    try:
        name, input_tokens, output_tokens = value.split(":", maxsplit=2)
        bucket = Bucket(name, int(input_tokens), int(output_tokens))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "bucket must have the form NAME:INPUT_TOKENS:OUTPUT_TOKENS"
        ) from error
    return bucket


def load_tokenizer(
    model_id: str,
    revision: str | None,
    local_files_only: bool,
) -> tuple[Any, str]:
    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError as error:
        raise DatasetValidationError(
            "transformers is required; install hack/benchmark/colab_profiling/requirements.txt"
        ) from error

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
    except Exception as error:
        raise DatasetValidationError(
            f"could not load tokenizer for {model_id!r} at revision {revision!r}: {error}"
        ) from error
    return tokenizer, transformers.__version__


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="model ID whose tokenizer is used")
    parser.add_argument(
        "--tokenizer-revision",
        default="main",
        help="tokenizer revision; use an immutable Hugging Face commit for archival runs",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1546, help="master RNG seed")
    parser.add_argument(
        "--profiling-prompts-per-bucket",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--heldout-prompts-per-bucket",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--bucket",
        action="append",
        type=parse_bucket,
        help=(
            "custom NAME:INPUT_TOKENS:OUTPUT_TOKENS bucket; repeat to replace "
            "the three defaults"
        ),
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="do not download missing tokenizer files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    config = GeneratorConfig(
        model_id=args.model,
        requested_tokenizer_revision=args.tokenizer_revision,
        master_seed=args.seed,
        buckets=tuple(args.bucket) if args.bucket else DEFAULT_BUCKETS,
        profiling_prompts_per_bucket=args.profiling_prompts_per_bucket,
        heldout_prompts_per_bucket=args.heldout_prompts_per_bucket,
    )

    try:
        validate_config(config)
        tokenizer, transformers_version = load_tokenizer(
            config.model_id,
            config.requested_tokenizer_revision,
            args.local_files_only,
        )
        metadata = tokenizer_metadata(
            tokenizer,
            config.requested_tokenizer_revision,
            transformers_version,
        )
        manifest = generate_dataset(args.output_dir, config, metadata)
    except (DatasetValidationError, OSError, RuntimeError) as error:
        print(f"dataset generation failed: {error}", file=sys.stderr)
        return 1

    total_records = sum(
        artifact["records"] for artifact in manifest["artifacts"].values()
    )
    print(f"wrote {total_records} validated request records to {args.output_dir}")
    print(f"dataset checksum: {manifest['dataset_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

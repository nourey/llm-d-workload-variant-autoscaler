"""Focused no-GPU tests for the #1546 prompt dataset generator."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_dataset as generator


class FakeAddedToken:
    def __init__(self, special: bool) -> None:
        self.special = special


class FakeTokenizer:
    name_or_path = "fake/model"
    all_special_ids = [0, 1]
    added_tokens_decoder = {2: FakeAddedToken(special=True)}
    init_kwargs = {"_commit_hash": "fake-tokenizer-commit"}

    def get_vocab(self) -> dict[str, int]:
        vocabulary = {
            "<bos>": 0,
            "<eos>": 1,
            "<added-special>": 2,
        }
        vocabulary.update({f"token-{token_id}": token_id for token_id in range(3, 515)})
        return vocabulary


def fake_metadata() -> generator.TokenizerMetadata:
    return generator.tokenizer_metadata(
        FakeTokenizer(),
        requested_revision="fake-requested-revision",
        transformers_version="test-transformers",
    )


def small_config() -> generator.GeneratorConfig:
    return generator.GeneratorConfig(
        model_id="fake/model",
        requested_tokenizer_revision="fake-requested-revision",
        master_seed=1546,
        buckets=generator.DEFAULT_BUCKETS,
        profiling_prompts_per_bucket=4,
        heldout_prompts_per_bucket=3,
    )


class DatasetGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = small_config()
        self.metadata = fake_metadata()
        self.records = generator.generate_records(self.config, self.metadata)

    def test_default_contract_and_split_invariants(self) -> None:
        generator.validate_records(self.records, self.config, self.metadata)

        defaults = {
            bucket.name: (
                bucket.input_tokens,
                bucket.target_output_tokens,
                bucket.total_target_tokens,
            )
            for bucket in generator.DEFAULT_BUCKETS
        }
        self.assertEqual(
            defaults,
            {
                "input-heavy": (384, 128, 512),
                "balanced": (256, 256, 512),
                "output-heavy": (128, 384, 512),
            },
        )

        prompts_by_split = {
            split: {
                tuple(record["prompt_token_ids"])
                for record in self.records
                if record["split"] == split
            }
            for split in generator.SPLITS
        }
        self.assertFalse(
            prompts_by_split["profiling"] & prompts_by_split["heldout"]
        )
        for record in self.records:
            self.assertEqual(
                len(record["prompt_token_ids"]), record["prompt_token_count"]
            )
            self.assertFalse(
                set(record["prompt_token_ids"]) & self.metadata.special_token_ids
            )
            self.assertIsNone(record["server_reported_completion_tokens"])

    def test_rng_has_a_stable_known_vector(self) -> None:
        rng = generator.Sha256RNG(1546)
        self.assertEqual(
            [rng.randbelow(1000) for _ in range(5)],
            [584, 970, 617, 698, 580],
        )

    def test_regeneration_is_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            first = Path(temporary_directory) / "first"
            second = Path(temporary_directory) / "second"
            first_manifest = generator.generate_dataset(
                first, self.config, self.metadata
            )
            second_manifest = generator.generate_dataset(
                second, self.config, self.metadata
            )

            for filename in ("profiling.jsonl", "heldout.jsonl", "manifest.json"):
                self.assertEqual(
                    (first / filename).read_bytes(),
                    (second / filename).read_bytes(),
                )

            self.assertEqual(first_manifest, second_manifest)
            for filename, artifact in first_manifest["artifacts"].items():
                checksum = "sha256:" + hashlib.sha256(
                    (first / filename).read_bytes()
                ).hexdigest()
                self.assertEqual(checksum, artifact["sha256"])

    def test_wrong_length_is_rejected(self) -> None:
        records = copy.deepcopy(self.records)
        records[0]["prompt_token_ids"].pop()
        with self.assertRaisesRegex(
            generator.DatasetValidationError, "expected 384 prompt tokens"
        ):
            generator.validate_records(records, self.config, self.metadata)

    def test_special_and_invalid_ids_are_rejected(self) -> None:
        for bad_token_id, message in ((0, "special token IDs"), (9999, "invalid token IDs")):
            with self.subTest(bad_token_id=bad_token_id):
                records = copy.deepcopy(self.records)
                records[0]["prompt_token_ids"][0] = bad_token_id
                with self.assertRaisesRegex(generator.DatasetValidationError, message):
                    generator.validate_records(records, self.config, self.metadata)

    def test_duplicate_within_split_is_rejected(self) -> None:
        records = copy.deepcopy(self.records)
        first, second = records[0], records[1]
        second["prompt_token_ids"] = list(first["prompt_token_ids"])
        second["prompt_token_count"] = first["prompt_token_count"]
        second["prompt_hash"] = first["prompt_hash"]
        with self.assertRaisesRegex(generator.DatasetValidationError, "duplicate prompt"):
            generator.validate_records(records, self.config, self.metadata)

    def test_cross_split_overlap_is_rejected(self) -> None:
        records = copy.deepcopy(self.records)
        profiling = next(
            record
            for record in records
            if record["split"] == "profiling" and record["bucket"] == "input-heavy"
        )
        heldout = next(
            record
            for record in records
            if record["split"] == "heldout" and record["bucket"] == "input-heavy"
        )
        heldout["prompt_token_ids"] = list(profiling["prompt_token_ids"])
        heldout["prompt_token_count"] = profiling["prompt_token_count"]
        heldout["prompt_hash"] = profiling["prompt_hash"]
        with self.assertRaisesRegex(
            generator.DatasetValidationError, "overlap"
        ):
            generator.validate_records(records, self.config, self.metadata)

    def test_bucket_metadata_and_hash_are_rejected_when_inconsistent(self) -> None:
        records = copy.deepcopy(self.records)
        records[0]["target_output_tokens"] += 1
        with self.assertRaisesRegex(
            generator.DatasetValidationError, "wrong target_output_tokens"
        ):
            generator.validate_records(records, self.config, self.metadata)

        records = copy.deepcopy(self.records)
        records[0]["prompt_hash"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(
            generator.DatasetValidationError, "prompt_hash does not match"
        ):
            generator.validate_records(records, self.config, self.metadata)

    def test_custom_buckets_replace_defaults(self) -> None:
        custom = generator.parse_bucket("length-check:128:128")
        self.assertEqual(custom, generator.Bucket("length-check", 128, 128))
        with self.assertRaises(generator.DatasetValidationError):
            generator.validate_config(
                generator.GeneratorConfig(
                    model_id="fake/model",
                    requested_tokenizer_revision="revision",
                    master_seed=1,
                    buckets=(custom, custom),
                    profiling_prompts_per_bucket=1,
                    heldout_prompts_per_bucket=1,
                )
            )

    def test_cli_defaults_are_modest_and_configurable(self) -> None:
        arguments = generator.build_argument_parser().parse_args(
            ["--model", "fake/model", "--output-dir", "dataset"]
        )
        self.assertEqual(arguments.profiling_prompts_per_bucket, 64)
        self.assertEqual(arguments.heldout_prompts_per_bucket, 32)
        self.assertIsNone(arguments.bucket)

    def test_manifest_exposes_output_acceptance_contract(self) -> None:
        artifacts = generator._split_artifacts(self.records)
        manifest = generator.build_manifest(self.config, self.metadata, artifacts)
        contract = manifest["output_length_contract"]
        self.assertEqual(contract["target_field"], "target_output_tokens")
        self.assertEqual(
            contract["server_reported_field"],
            "server_reported_completion_tokens",
        )
        self.assertIn("==", contract["acceptance_rule"])
        json.dumps(manifest)


if __name__ == "__main__":
    unittest.main()

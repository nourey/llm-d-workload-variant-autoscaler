# #1546 Colab profiling request data

This directory contains the controlled request-data generator and the bounded
request-contract smoke client for the #1546 single-GPU monolithic-vLLM
experiment. It does not implement a profiler, launch vLLM itself, collect
telemetry, calculate capacity, or implement autoscaling.

## Generate in Colab

After cloning the repository, install the explicit Hugging Face tokenizer
dependency and run:

```bash
python -m pip install -r hack/benchmark/colab_profiling/requirements.txt
python hack/benchmark/colab_profiling/generate_dataset.py \
  --model Qwen/Qwen2.5-3B \
  --tokenizer-revision 3aab1f1954e9cc14eb9509a215f9e5ca08227a9b \
  --output-dir /content/wva-1546-prompts
```

`transformers` downloads the configured model's tokenizer files from Hugging
Face. Public tokenizers need no credential. For a gated tokenizer, provide your
own `HF_TOKEN` through the Colab environment; never place it in this repository
or generated data. The command pins the approved immutable revision. The
manifest records both the requested revision and the resolved commit when
`transformers` exposes it.

The output directory must be empty. This prevents an old and a regenerated
dataset from being mixed accidentally.

## Defaults and configuration

The default dataset has 64 profiling and 32 held-out prompts per bucket:

| Bucket | Input tokens | Target output tokens | Total target tokens |
| --- | ---: | ---: | ---: |
| `input-heavy` | 384 | 128 | 512 |
| `balanced` | 256 | 256 | 512 |
| `output-heavy` | 128 | 384 | 512 |

This produces 288 requests and roughly 74,000 stored prompt tokens: enough to
rotate prompts during repeated profiling and later held-out checks, while
remaining a small JSONL artifact. Override the counts with
`--profiling-prompts-per-bucket` and `--heldout-prompts-per-bucket`.

Repeat `--bucket NAME:INPUT_TOKENS:OUTPUT_TOKENS` to replace the default bucket
set. That provides an isolated path for a later length-sensitivity dataset
without adding those lengths to the initial dataset. For example:

```bash
python hack/benchmark/colab_profiling/generate_dataset.py \
  --model Qwen/Qwen2.5-3B \
  --tokenizer-revision 3aab1f1954e9cc14eb9509a215f9e5ca08227a9b \
  --output-dir /content/wva-1546-length-check \
  --bucket short:128:128 \
  --bucket medium:256:256 \
  --bucket long:384:384
```

## Artifacts and contract

```text
<output-dir>/
├── manifest.json
├── profiling.jsonl
└── heldout.jsonl
```

Each JSONL record stores the exact `prompt_token_ids`; decoded text and
re-tokenized text are never used as truth. The record also contains:

- `schema_version`, stable `request_id`, `split`, and `bucket`;
- `prompt_token_count`, `prompt_hash`, and the derived `generator_seed`;
- `target_output_tokens` and `total_target_tokens`;
- `model_id` and `tokenizer_revision`;
- `server_reported_completion_tokens`, which is always `null` in generated
  request data.

`target_output_tokens` is the requested output contract. A future client must
set `max_tokens` and `min_tokens` to that value and set `ignore_eos=true`. It
must separately populate `server_reported_completion_tokens` from the server's
`usage.completion_tokens`; a profiling observation is valid only when the two
values are equal. The generator intentionally implements none of that request
execution; the bounded smoke client below implements only this contract gate.

The manifest records the model/tokenizer identity, tokenizer vocabulary and
eligible-ID fingerprints, master and derived seeds, bucket/count configuration,
JSONL checksums, and an aggregate dataset checksum. Generation validates all
record invariants, regenerates the complete dataset in memory, and requires the
second JSONL encoding to be byte-identical before writing.

## Local no-GPU tests

The tests use a fake tokenizer, so they need neither `transformers`, a network
connection, nor a GPU:

```bash
python -m unittest discover \
  -s hack/benchmark/colab_profiling \
  -p 'test_*.py' \
  -v
```

## Bounded vLLM request-contract smoke

This is a six-request, sequential contract check, not profiling. It sends the
first two profiling records from each approved bucket in the approved bucket
order and has no concurrency, RPS, load, or capacity logic.

In a Colab GPU runtime, install the approved vLLM version:

```bash
python -m pip install "vllm==0.28.0"
```

Start the approved single-GPU, monolithic server in a background shell cell:

```bash
nohup vllm serve Qwen/Qwen2.5-3B \
  --revision 3aab1f1954e9cc14eb9509a215f9e5ca08227a9b \
  --tokenizer-revision 3aab1f1954e9cc14eb9509a215f9e5ca08227a9b \
  --dtype float16 \
  --tensor-parallel-size 1 \
  --host 127.0.0.1 \
  --port 8000 \
  > /content/wva-1546-vllm.log 2>&1 &
```

Wait for readiness before running the smoke client:

```bash
for attempt in {1..120}; do
  if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null; then
    break
  fi
  sleep 5
done
curl -fsS http://127.0.0.1:8000/v1/models >/dev/null
```

Then run the exact bounded client:

```bash
python hack/benchmark/colab_profiling/run_request_smoke.py \
  --profiling-jsonl /content/wva-1546-prompts/profiling.jsonl \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen2.5-3B \
  --tokenizer-revision 3aab1f1954e9cc14eb9509a215f9e5ca08227a9b \
  --output-dir /content/wva-1546-request-smoke
```

The client uses only the Python standard library. It sends each integer
`prompt_token_ids` array directly as the completions `prompt` with `n=1`,
`stream=false`, `temperature=0`, equal `min_tokens`/`max_tokens`,
`ignore_eos=true`, and `return_token_ids=true`. It fails the overall run if any
HTTP, JSON, usage, finish-reason, model-identity, or returned-token-ID evidence
violates the contract.

Results are separate from source data and are published only after all six
requests finish:

```text
/content/wva-1546-request-smoke/
├── request_results.jsonl
└── summary.json
```

`request_results.jsonl` retains per-request expected and observed token counts,
choice-level `prompt_token_ids` and `token_ids` when vLLM returns them, contract
failures, and monotonic submit/terminal/latency nanoseconds. These timestamps
are audit timing only. They are not throughput, saturation, or profiling
evidence and must not be used to estimate capacity.

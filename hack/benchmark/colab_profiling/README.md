# #1546 Colab profiling request data

This directory contains the controlled request-data generator, the bounded
request-contract smoke client, and the balanced-bucket fixed-concurrency
profiling harness for the #1546 single-GPU monolithic-vLLM experiment. It
does not launch vLLM itself, and it does not implement autoscaling.

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

All tests in this directory use fake tokenizers, fake HTTP transports, and
fake GPU/telemetry samplers, so they need neither `transformers`, a network
connection, nor a GPU. This includes the dataset generator tests, the
request-contract smoke tests, and the balanced-bucket profiling harness
tests below:

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

## Balanced-bucket fixed-concurrency profiling (`profile_balanced_bucket.py`)

This is the bounded profiling harness authorized by
`.claude/work/1546/DECISIONS.md`. It answers one deliberately narrow
question:

> Does the real single-GPU monolithic vLLM runtime produce a stable,
> repeatable saturation plateau for the **balanced** bucket
> (`L_in = 256`, `L_out = 256`, `W = 512` logical tokens/request) as fixed
> client-side concurrency increases?

It profiles **only** the `balanced` bucket, using **only**
`profiling.jsonl` records (never `heldout.jsonl`). It does not profile the
`input-heavy` or `output-heavy` buckets, does not run held-out mixtures, and
does not implement autoscaling.

**This tool is non-P/D monolithic `V_M` evidence.** It reports a
*total-token* (prompt + completion) service rate under one fixed serving
process. It is **not**:

* physical freed-KV-block throughput,
* isolated decoder `V_D` (no P/D disaggregation is involved),
* production Wide-EP capacity,
* SLO-safe operating capacity.

**Plateau acceptance is a HUMAN REVIEW decision.** This tool never selects a
final `V_M`. It only produces a per-concurrency-point summary table
(completions/s, total-token/s, adjacent relative throughput gain, and
run validity) for a human plus ChatGPT to inspect after a real Colab run.

### Prerequisites

* A `profiling.jsonl` generated by `generate_dataset.py` (see above) for the
  approved model/tokenizer revision, containing the `balanced` bucket.
* A running vLLM 0.28.0 OpenAI-compatible server for
  `Qwen/Qwen2.5-3B` at the approved immutable revision, reachable over HTTP.
* No GPU is required to run the harness's own local tests (see below); a GPU
  *is* required to produce real profiling evidence.

### Recommended vLLM launch command

Use the same single-GPU, monolithic, `TP=1` launch as the request-contract
smoke, with the additional flags this experiment requires to keep the
capacity measurement uncontaminated (D5):

```bash
python -m pip install "vllm==0.28.0"

nohup vllm serve Qwen/Qwen2.5-3B \
  --revision 3aab1f1954e9cc14eb9509a215f9e5ca08227a9b \
  --tokenizer-revision 3aab1f1954e9cc14eb9509a215f9e5ca08227a9b \
  --dtype float16 \
  --tensor-parallel-size 1 \
  --max-model-len 1024 \
  --generation-config vllm \
  --no-enable-prefix-caching \
  --host 127.0.0.1 \
  --port 8000 \
  > /content/wva-1546-vllm.log 2>&1 &

for attempt in {1..120}; do
  if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null; then
    break
  fi
  sleep 5
done
curl -fsS http://127.0.0.1:8000/v1/models >/dev/null
```

Do not change these serving flags between concurrency points within one run;
any change defines a different capacity artifact (D5).

### Exact profiling invocation

Start with a conservative subset before attempting the full ladder (the
first real run should escalate conservatively per D7):

```bash
python hack/benchmark/colab_profiling/profile_balanced_bucket.py \
  --profiling-jsonl /content/wva-1546-prompts/profiling.jsonl \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen2.5-3B \
  --tokenizer-revision 3aab1f1954e9cc14eb9509a215f9e5ca08227a9b \
  --vllm-version 0.28.0 \
  --dtype float16 \
  --tensor-parallel-size 1 \
  --max-model-len 1024 \
  --generation-config vllm \
  --concurrency 1,2,4 \
  --settling-seconds 30 \
  --measurement-seconds 60 \
  --drain-timeout-seconds 120 \
  --metrics-interval-seconds 1 \
  --output-dir /content/wva-1546-balanced-profiling
```

Once the conservative subset looks healthy (no invalidation reasons, no
server errors, no thermal/runtime instability), extend `--concurrency` up to
the full candidate ladder `1,2,4,8,16,32` in a fresh `--output-dir`. The
implementation never runs an unbounded or automatically-escalating
concurrency search; every concurrency value must be requested explicitly.

Use `--no-telemetry` only for harness debugging; real profiling runs should
keep telemetry enabled so the artifacts can distinguish a real server-side
saturation plateau from a client-imposed one (D13/D16).

### Artifact layout

```text
/content/wva-1546-balanced-profiling/
├── experiment_manifest.json     # model/vLLM/tokenizer identity, dataset checksum,
│                                 # concurrency ladder, timing config, GPU fingerprint
├── request_results.jsonl        # one line per executed request (D11 fields)
├── point_summaries.jsonl        # one line per concurrency point (D12/D16 fields)
├── summary.json                 # review table + explicit non-goal/plateau warnings
└── telemetry/
    ├── vllm_metrics.jsonl       # periodic raw + selected-known /metrics samples
    └── gpu_metrics.jsonl        # periodic nvidia-smi samples (diagnostic)
```

Each concurrency point runs through explicit phases (D9): an idle/precondition
check against `/v1/models`, a settling/warm-up period, a fixed measurement
window `[T0, T1)`, admission stop at `T1`, a bounded drain, a post-run
precondition re-check, and finally the summary/artifact write. Only
requests whose **server-validated** completion lands inside `[T0, T1)` are
counted in the numerator (D3/D10); settling and drain completions are
retained only as bookkeeping.

For the balanced bucket, `completed_total_tokens_per_second` must equal
`512 * completed_requests_per_second` for every valid point; the harness
asserts this and marks a point invalid (`token_rate_invariant_violated`) if
it does not hold.

### Known limitations / risks not yet resolved by local tests

* The closed-loop scheduler uses a bounded Python thread pool (not
  `asyncio`) as the concurrency-limiting mechanism, since this directory's
  existing components deliberately use only the Python standard library. At
  higher concurrency values (16, 32) this has not yet been validated against
  real vLLM latency/jitter; only mocked HTTP transports have been exercised
  locally.
* A load point's *drain* is bounded by `--drain-timeout-seconds` for
  **invalidation purposes**, but the underlying HTTP call for any
  already-admitted request can still block up to
  `--request-timeout-seconds`. A stuck server can therefore make one load
  point take longer than `drain_timeout_seconds` before the process moves on
  to report `drain_timeout` invalidation.
* A load point is marked invalid whenever *any* request in that point
  (including settling/drain, not only the measurement window) fails the
  token contract. This is a conservative, documented interpretation of D15;
  it has not been validated against real vLLM warm-up behavior.
* vLLM 0.28.0's exact `/metrics` surface has not been observed on real
  hardware by this implementation; parsing is generic Prometheus-text
  parsing plus a best-effort list of commonly expected `vllm:*` metric
  names, each explicitly marked present/absent.
* None of this has been exercised against a real GPU, a real vLLM server, or
  real network conditions. All local tests use fake/mocked HTTP transports.

Local tests for this harness (`test_profile_balanced_bucket.py`) run with the
same command as the rest of this directory; see
[Local no-GPU tests](#local-no-gpu-tests) above.

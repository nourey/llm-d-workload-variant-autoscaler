# #1546 Colab profiling request data

This directory contains the controlled request-data generator, the bounded
request-contract smoke client, and the generic fixed-concurrency bucket
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

## Fixed-concurrency bucket profiling (`profile_bucket.py`)

This is the bounded profiling harness authorized by
`.claude/work/1546/DECISIONS.md`, generalized to run against any one of the
three approved dataset buckets. It answers one deliberately narrow question
per run:

> Does the real single-GPU monolithic vLLM runtime produce a stable,
> repeatable saturation plateau for the **selected bucket** as fixed
> client-side concurrency increases?

### Supported buckets and validation status

| Bucket | `L_in` | `L_out` | `W` (total target tokens) | Real-hardware status |
| --- | ---: | ---: | ---: | --- |
| `balanced` | 256 | 256 | 512 | **VALIDATED** on real Tesla T4 hardware (see below) |
| `input-heavy` | 384 | 128 | 512 | **NOT YET VALIDATED** |
| `output-heavy` | 128 | 384 | 512 | **NOT YET VALIDATED** |

**Balanced-bucket result (already validated, do not re-derive from local
tests):** two independent 180s confirmation runs, on independent Tesla T4
instances, under the identical serving configuration documented below,
reproduced:

```
C=48: 1228.8 logical token/s
C=64: 1274.3 logical token/s   (adjacent gain ~3.7%)
```

giving a provisional empirical monolithic capacity
`V_M^(balanced) ≈ 1.27k logical token/s`. This conclusion was reached by
HUMAN REVIEW of real Colab evidence, not by this repository's code.

`input-heavy` and `output-heavy` are runnable through the exact same,
already-validated measurement methodology, but **no real-hardware run has
been performed for them yet**. Do not treat any input-heavy/output-heavy
number as validated until a real Colab run has been reviewed the same way.

Every run profiles **exactly one** bucket, using **only**
`profiling.jsonl` records for that bucket (never `heldout.jsonl` and never
another bucket's records). This tool does not run held-out mixtures, does
not implement mixed-workload composition, and does not implement
autoscaling.

**This tool is non-P/D monolithic `V_M` evidence.** It reports a
*total-token* (prompt + completion) service rate under one fixed serving
process. It is **not**:

* physical freed-KV-block throughput,
* isolated decoder `V_D` (no P/D disaggregation is involved),
* production Wide-EP capacity,
* SLO-safe operating capacity.

**Plateau acceptance is a HUMAN REVIEW decision.** This tool never selects a
final `V_M`. It only produces a per-concurrency-point summary table
(completions/s, total-token/s, adjacent relative throughput gain, run
validity, and a preemption-delta signal) for a human plus ChatGPT to inspect
after a real Colab run.

### Prerequisites

* A `profiling.jsonl` generated by `generate_dataset.py` (see above) for the
  approved model/tokenizer revision, containing the bucket you intend to
  profile.
* A running vLLM 0.28.0 OpenAI-compatible server for
  `Qwen/Qwen2.5-3B` at the approved immutable revision, reachable over HTTP.
* No GPU is required to run the harness's own local tests (see below); a GPU
  *is* required to produce real profiling evidence.

### Exact runtime launch contract

Use the same single-GPU, monolithic, `TP=1` launch as the request-contract
smoke, with the additional flags this experiment requires to keep the
capacity measurement uncontaminated (D5). This is the exact configuration
under which the balanced result above was validated:

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
  --gpu-memory-utilization 0.90 \
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

Do not change these serving flags between concurrency points, or between
buckets, within one comparable experiment; any change defines a different
capacity artifact (D5). Pass the same `--gpu-memory-utilization` value to
the profiler (below) so it is recorded in the manifest; the profiler cannot
inspect the running server's actual flags and never modifies it.

### Running a single bucket

```bash
python hack/benchmark/colab_profiling/profile_bucket.py \
  --profiling-jsonl /content/wva-1546-prompts/profiling.jsonl \
  --base-url http://127.0.0.1:8000 \
  --bucket balanced \
  --model Qwen/Qwen2.5-3B \
  --tokenizer-revision 3aab1f1954e9cc14eb9509a215f9e5ca08227a9b \
  --vllm-version 0.28.0 \
  --dtype float16 \
  --tensor-parallel-size 1 \
  --max-model-len 1024 \
  --generation-config vllm \
  --gpu-memory-utilization 0.90 \
  --concurrency 1,2,4 \
  --settling-seconds 30 \
  --measurement-seconds 60 \
  --drain-timeout-seconds 120 \
  --metrics-interval-seconds 1 \
  --output-dir /content/wva-1546-balanced-profiling
```

`--bucket` is required and must be exactly one of `balanced`,
`input-heavy`, or `output-heavy`; any other value is rejected before any
HTTP request is made. To profile a different bucket, change only `--bucket`
and `--output-dir` (use a fresh, non-existent output directory each time —
the tool refuses to overwrite an existing directory):

```bash
python hack/benchmark/colab_profiling/profile_bucket.py \
  --profiling-jsonl /content/wva-1546-prompts/profiling.jsonl \
  --base-url http://127.0.0.1:8000 \
  --bucket input-heavy \
  --model Qwen/Qwen2.5-3B \
  --tokenizer-revision 3aab1f1954e9cc14eb9509a215f9e5ca08227a9b \
  --gpu-memory-utilization 0.90 \
  --concurrency 1,2,4 \
  --output-dir /content/wva-1546-input-heavy-profiling
```

Start with a conservative concurrency subset before attempting the full
ladder (the first real run should escalate conservatively per D7). Once a
subset looks healthy (no invalidation reasons, no server errors, no
thermal/runtime instability), extend `--concurrency` up to the full
candidate ladder `1,2,4,8,16,32` in a fresh `--output-dir`. The
implementation never runs an unbounded or automatically-escalating
concurrency search; every concurrency value must be requested explicitly.

Use `--no-telemetry` only for harness debugging; real profiling runs should
keep telemetry enabled so the artifacts can distinguish a real server-side
saturation plateau from a client-imposed one (D13/D16).

`profile_balanced_bucket.py` still exists as a thin, unchanged-behavior
compatibility wrapper: it exposes the identical CLI it always did (no
`--bucket` flag) and always profiles `balanced`. It contains no profiling
logic of its own — every call delegates to `profile_bucket.py`. Prefer
`profile_bucket.py --bucket balanced` for new usage; the wrapper is kept
only so existing commands/scripts/docs that invoke it directly keep working.

### Artifact layout

```text
/content/wva-1546-balanced-profiling/
├── experiment_manifest.json     # model/vLLM/tokenizer identity, selected bucket,
│                                 # dataset checksum, concurrency ladder, timing
│                                 # config, operator-declared serving config,
│                                 # GPU fingerprint
├── request_results.jsonl        # one line per executed request (D11 fields)
├── point_summaries.jsonl        # one line per concurrency point (D12/D16 fields
│                                 # plus per-point telemetry summaries)
├── summary.json                 # review table + explicit non-goal/plateau warnings
└── telemetry/
    ├── vllm_metrics.jsonl       # periodic raw + selected-known /metrics samples
    └── gpu_metrics.jsonl        # periodic nvidia-smi samples (diagnostic)
```

Every artifact explicitly names the selected `bucket` (manifest, each
request result, each point summary, and the summary/review table), so a
directory's contents are self-describing even out of context.

Each concurrency point runs through explicit phases (D9): an idle/precondition
check against `/v1/models`, a settling/warm-up period, a fixed measurement
window `[T0, T1)`, admission stop at `T1`, a bounded drain, a post-run
precondition re-check, and finally the summary/artifact write. Only
requests whose **server-validated** completion lands inside `[T0, T1)` are
counted in the numerator (D3/D10); settling and drain completions are
retained only as bookkeeping.

For every currently approved bucket, `total_target_tokens = 512`, so
`completed_total_tokens_per_second` must equal
`512 * completed_requests_per_second` for every valid point; the harness
computes this from the *selected bucket's own* `total_target_tokens` (not a
hard-coded constant) and marks a point invalid
(`token_rate_invariant_violated`) if it does not hold.

### Per-point telemetry summaries

In addition to the raw `telemetry/*.jsonl` archives (never removed), each
entry in `point_summaries.jsonl` now carries concise, measurement-window-only
derived summaries:

* `vllm_telemetry.num_requests_running` / `num_requests_waiting` /
  `kv_cache_usage_perc`: `{available, avg, max, sample_count}`, or
  `{"available": false}` if that metric never appeared — never silently
  reported as zero.
* `vllm_telemetry.num_preemptions_total`: `{available, start, end, delta}`
  over the measurement window. A nonzero `delta` is part of the
  saturation-validity review and must not be ignored.
* `gpu_telemetry`: `utilization.gpu`, `memory.used`, `temperature.gpu`, and
  `power.draw` avg/max, plus
  `clocks_throttle_reasons.active.{total_sample_count,nonzero_sample_count,distinct_nonzero_values}`.

These summaries are computed only from samples whose timestamp falls inside
that specific point's own `[T0, T1)` window, so telemetry from a different
concurrency or a different point can never leak into another point's
summary.

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
  names (with a `kv_cache_usage_perc`/`gpu_cache_usage_perc` name fallback),
  each explicitly marked present/absent.
* `--gpu-memory-utilization` and `--prefix-caching` (and `--dtype`,
  `--tensor-parallel-size`, `--max-model-len`, `--generation-config`) are
  **operator-declared** values recorded in the manifest as-is; the profiler
  cannot independently inspect the running server's actual launch flags
  beyond `/v1/models` identity and a best-effort `/version` probe, and it
  never modifies the running server.
* `input-heavy` and `output-heavy` have only been exercised with mocked HTTP
  transports; no real GPU run has been performed for either bucket.
* None of this has been exercised against a real network. All local tests
  use fake/mocked HTTP transports and fake GPU/telemetry samplers.

### Local tests

```bash
python -m unittest discover \
  -s hack/benchmark/colab_profiling \
  -p 'test_*.py' \
  -v
```

`test_profile_bucket.py` covers the generic implementation across all three
buckets (selection, dynamic request/invariant arithmetic, concurrency/window/
drain semantics, telemetry summaries, and manifest fields).
`test_profile_balanced_bucket.py` covers only the thin compatibility
wrapper (no `--bucket` flag, always `balanced`, same artifact shape).

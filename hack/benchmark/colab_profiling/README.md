# #1546 Colab profiling request data

This directory contains the controlled request-data generator, the bounded
request-contract smoke client, the generic fixed-concurrency bucket
profiling harness, and the open-loop mixed-workload composition-validation
harness for the #1546 single-GPU monolithic-vLLM experiment. It does not
launch vLLM itself, and it does not implement autoscaling.

## Experimental procedure

This section is a single end-to-end map of the whole #1546 experiment. Every
step below is explained in detail in its own section further down; read this
first, then jump to the linked section for exact commands/contracts.

```text
1. generate_dataset.py
   deterministic synthetic raw TOKEN-ID prompts, per work-shape bucket
                        |
                        v
2. run_request_smoke.py
   bounded 6-request sanity check of the exact request/response contract
                        |
                        v
3. profile_bucket.py   (repeated once per bucket: input-heavy / balanced / output-heavy)
   fixed CLOSED-LOOP concurrency ladder
     -> per-point engine-counter throughput
       -> human-reviewed saturated maximum  =>  V_M^(bucket)
                        |
                        v
4. (3) is repeated independently for all three work-shape buckets,
   producing three independent, accepted V_M^(bucket) values
                        |
                        v
5. validate_mixed_workload.py
   fixed OPEN-LOOP arrival rates for all buckets at once, derived from a
   target composition rho and the three V_M^(bucket) values from step (4)
                        |
                        v
6. observe waiting / running / outstanding accumulation over [T0, T1)
                        |
                        v
7. compare the predicted rho_pred ~= 1 boundary against where the observed
   evidence actually shows saturation/backlog onset
                        |
                        v
8. only once this composition gate is accepted does further research into
   predictors, workload classification, or autoscaling become meaningful
```

* Steps 1–2: see ["Exactly what is sent to the model"](#exactly-what-is-sent-to-the-model),
  ["The output contract"](#the-output-contract), and
  ["Generate in Colab"](#generate-in-colab) /
  ["Bounded vLLM request-contract smoke"](#bounded-vllm-request-contract-smoke).
* Steps 3–4: see ["How load is generated"](#how-load-is-generated),
  ["How `V_M` is measured"](#how-v_m-is-measured), and
  ["Fixed-concurrency bucket profiling (`profile_bucket.py`)"](#fixed-concurrency-bucket-profiling-profile_bucketpy).
* Steps 5–8: see ["How load is generated"](#how-load-is-generated) and
  ["Mixed-workload composition validation"](#mixed-workload-composition-validation).

## Exactly what is sent to the model

**`input-heavy`, `balanced`, and `output-heavy` are WORK-SHAPE classes, not
semantic classes.** This experiment does not use natural-language prompts as
experimental truth at all:

* `generate_dataset.py` loads the real Qwen tokenizer's vocabulary and
  builds, per bucket, deterministic pseudo-random sequences of valid,
  non-special token IDs of exactly the bucket's `input_tokens` length;
* those integer token IDs — never any decoded text — are stored directly in
  `profiling.jsonl` / `heldout.jsonl` as `prompt_token_ids`;
* every client in this directory (`run_request_smoke.py`, `profile_bucket.py`,
  `validate_mixed_workload.py`) sends that integer array **directly** as the
  `/v1/completions` request's `prompt` field; it is never decoded to text and
  re-tokenized before the request is sent.

Schematically (not a real generated record):

```json
{
  "bucket": "input-heavy",
  "prompt_token_ids": [4913, 210, 88831, "... 384 valid non-special token IDs total ..."],
  "prompt_token_count": 384,
  "target_output_tokens": 128,
  "total_target_tokens": 512
}
```

The point of this design is to isolate **input/output length SHAPE** from
semantic content variability: three buckets that hold total tokens
(`W = L_in + L_out = 512`) fixed while varying how those 512 tokens split
between the prompt and the generated output. Whatever text the model
happens to generate from a synthetic token-ID prompt is not meaningful and
is never inspected as part of this experiment (see
["The output contract"](#the-output-contract) below).

## The output contract

Every request forces the engine to execute **exactly `L_out` autoregressive
decode steps**, never more, never fewer, unless the request itself fails.
For a bucket record with `target_output_tokens = L_out`, the request payload
sets:

* `min_tokens = L_out`
* `max_tokens = L_out`
* `ignore_eos = true`
* `temperature = 0`
* `n = 1`
* `stream = false`
* `return_token_ids = true`

(see `request_payload()` in `run_request_smoke.py`, reused unmodified by
`profile_bucket.py` and `validate_mixed_workload.py`).

A response is only accepted as valid profiling evidence if all of the
following hold (`validate_response()` in `run_request_smoke.py`):

* `usage.prompt_tokens == L_in`
* `usage.completion_tokens == L_out`
* `usage.total_tokens == L_in + L_out`
* `finish_reason == "length"`
* the response `model` matches the requested model, and returned
  `prompt_token_ids`/output token-ID evidence is checked when the server
  returns it

**The semantic CONTENT of the generated text is irrelevant to this
experiment.** The controlled, measured quantity is the number of decode
steps / logical output tokens the engine actually processed — not what those
tokens mean.

## How load is generated

This directory uses **two deliberately different** load-generation
strategies for two different questions. Confusing the two invalidates the
experiment they belong to, so both are described here side by side.

### Pure-bucket profiling: CLOSED LOOP, fixed concurrency (`profile_bucket.py`)

At a target concurrency `C`:

1. the deterministic startup ramp admits the initial `C` requests (paced,
   not a zero-delay burst — see
   ["Deterministic initial-admission ramp"](#deterministic-initial-admission-ramp));
2. once all `C` are outstanding, settling begins, then the `[T0, T1)`
   measurement window;
3. throughout settling, measurement, and drain, **the instant one request
   completes its replacement is admitted immediately**, so client-side
   offered concurrency stays at approximately `C` the whole time;
4. `C` is swept across an explicit ladder (e.g. `1,2,4,8,16,32,...`);
5. engine-counter throughput is measured at each `C`;
6. a human reviews the resulting throughput-vs-`C` curve to identify the
   saturated maximum region — see
   ["How saturation and `V_M` acceptance are determined"](#how-saturation-and-v_m-acceptance-are-determined).

**Concurrency is the independent load variable here.** The generator never
offers more than `C` outstanding requests at once for that bucket.

### Mixed-workload validation: OPEN LOOP, fixed arrival rate (`validate_mixed_workload.py`)

There is no concurrency cap and no replacement-on-completion here at all:

* completing a request does **not** trigger the next one — each bucket has
  its own absolute-time arrival schedule fixed in advance:
  `scheduled_time(k, b) = point_start + phase_b + k / request_rate_b`;
* arrivals for a bucket keep firing on schedule even while previous
  requests from that same bucket are still outstanding — this is exactly
  what lets a genuine server-side backlog (growing `waiting`/`running`/
  outstanding populations) show up in the evidence instead of being
  silently absorbed by the client;
* **request arrival rate is the independent load variable here**, not
  concurrency;
* a large, explicit client thread-pool concurrency budget
  (`--client-concurrency-budget`, default 4096) exists purely so the client
  itself never becomes the bottleneck; if it is ever actually exhausted,
  the point is invalidated (`client_concurrency_budget_exceeded`) instead
  of that client-side queueing being mistaken for server-side saturation.

See ["Mixed-workload composition validation"](#mixed-workload-composition-validation)
below for the exact scheduling/invalidation contract.

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

## How `V_M` is measured

At a fixed concurrency point, the question being asked is simply:

> How many logical prompt+generation tokens did the vLLM engine actually
> process per second during the measurement window?

That question is answered from vLLM's own engine-side counters, not from
counting terminal (finished) requests:

```
V_hat_M =
  (
    delta(vllm:prompt_tokens_total)
    +
    delta(vllm:generation_tokens_total)
  )
  /
  delta t
```

using the first and last valid `/metrics` samples inside `[T0, T1)`. This is
more robust than terminal-completion accounting because a single request can
straddle `T0` or `T1`: terminal accounting assigns that request's *entire*
`L_in + L_out` work to whichever side of the boundary its one finish
timestamp happens to land on, while the engine counters reflect work as it
is actually processed, continuously, regardless of when any one request
finishes. See
["Primary capacity estimator"](#primary-capacity-estimator-engine-side-token-counter-throughput)
below for the exact fail-closed contract (missing/ambiguous/reset counters
always invalidate the point; there is no silent fallback).

`V_M` is **NOT**:

* requests/second,
* concurrency,
* physical KV-block-release throughput,
* isolated decoder `V_D` (no P/D disaggregation is involved here),
* SLO-safe operating capacity.

`V_M` **is**: the maximum sustainable monolithic logical total-token
processing rate, for one specific model/GPU/vLLM/serving-config, under one
specific bucket/work-shape regime.

## How saturation and `V_M` acceptance are determined

A single throughput number at a single concurrency is not `V_M`. Accepting a
bucket's `V_M` requires running a concurrency ladder and a human reviewing
the resulting evidence for a maximum/saturated region where:

* increasing offered concurrency no longer materially increases engine
  throughput;
* the running-request population approaches an observed engine ceiling
  (never assumed to be any particular fixed number, e.g. `256`, in
  general — see
  ["Primary capacity estimator"](#primary-capacity-estimator-engine-side-token-counter-throughput));
* additional offered concurrency instead shows up as growing waiting
  population;
* failures/preemptions/runtime instability are absent, or present and
  explicitly reviewed rather than ignored;
* one or more independent repeats reproduce the candidate saturated region
  closely enough.

Non-monotonic local dips at individual concurrency points (e.g. the
`balanced` and `output-heavy` `C=160` dips reproduced twice each — see
["Supported buckets and validation status"](#supported-buckets-and-validation-status))
can and do occur; they are not automatically treated as measurement
failures. **The profiler itself never selects, infers, or auto-accepts a
plateau.** It only ever produces a per-point evidence table
(`run_valid`/`invalidation_reasons` plus the raw throughput/telemetry
numbers); a human reviews that table and records the accepted `V_M` value —
currently `V_M^(input-heavy) ~= 2.18k`, `V_M^(balanced) ~= 1.97k`, and
`V_M^(output-heavy) ~= 1.88k` logical token/s (see
["Supported buckets and validation status"](#supported-buckets-and-validation-status)
for the full per-point evidence trail behind each value).

## Fixed-concurrency bucket profiling (`profile_bucket.py`)

This is the bounded profiling harness authorized by
`.claude/work/1546/DECISIONS.md`, generalized to run against any one of the
three approved dataset buckets. It answers one deliberately narrow question
per run:

> Does the real single-GPU monolithic vLLM runtime produce a stable,
> repeatable saturation plateau for the **selected bucket** as fixed
> client-side concurrency increases?

### Supported buckets and validation status

All three buckets are now HUMAN-REVIEWED and ACCEPTED, using the PRIMARY
engine-counter estimator (see
["Primary capacity estimator"](#primary-capacity-estimator-engine-side-token-counter-throughput)
below) — NOT the secondary, boundary-sensitive terminal-completion
estimator that an earlier iteration of this evidence mistakenly relied on.

| Bucket | `L_in` | `L_out` | `W` (total target tokens) | Accepted `V_M` (engine-counter, HUMAN-REVIEWED) |
| --- | ---: | ---: | ---: | --- |
| `input-heavy` | 384 | 128 | 512 | **≈ 2.18k logical token/s** |
| `balanced` | 256 | 256 | 512 | **≈ 1.97k logical token/s** |
| `output-heavy` | 128 | 384 | 512 | **≈ 1.88k logical token/s** |

Each bucket independently reproduced a local, non-monotonic dip at one
mid-ladder concurrency point on two independent runs (e.g. `output-heavy`
`C=160` ≈ 1529.9 then repeat ≈ 1443.3; `balanced` `C=160` ≈ 1623.2 then
repeat ≈ 1624.7). This is treated as reproducible real-runtime scheduler/
batching behavior, **not a measurement failure**, and does not invalidate
the observed maximum/saturated region used for the accepted value. Full
per-bucket engine-throughput evidence (every concurrency point, both
repeats, the `C=288` saturation probes) is recorded in
`profiler.BUCKET_VALIDATION_STATUS` in `profile_bucket.py`.

**Bucket capacity ACCEPTANCE (the table above) is a distinct HUMAN REVIEW
decision from per-point RUN VALIDITY** (`run_valid`/`invalidation_reasons`,
computed by this module for every point). This module never infers, auto-
selects, or auto-accepts a capacity value — it only ever fails a point
closed or reports raw evidence for a human to review.

Every run of `profile_bucket.py` profiles **exactly one** bucket, using
**only** `profiling.jsonl` records for that bucket (never `heldout.jsonl`
and never another bucket's records). This tool does not run held-out
mixtures and does not implement autoscaling. The next research gate --
whether these independently measured `V_M^(b)` values predict a MIXED
workload's saturation boundary -- is implemented separately in
`validate_mixed_workload.py`; see
["Mixed-workload composition validation"](#mixed-workload-composition-validation)
below.

**This tool is non-P/D monolithic `V_M` evidence.** It reports a
*total-token* (prompt + completion) service rate under one fixed serving
process. It is **not**:

* physical freed-KV-block throughput,
* isolated decoder `V_D` (no P/D disaggregation is involved),
* production Wide-EP capacity,
* SLO-safe operating capacity.

**Plateau acceptance is a HUMAN REVIEW decision.** This tool never selects a
final `V_M`. It only produces a per-concurrency-point summary table (PRIMARY
engine-side token/s, adjacent relative throughput gain computed from that
primary estimator, secondary/boundary-sensitive completion diagnostics,
running/waiting saturation-ceiling evidence, run validity, and a
preemption-delta signal) for a human plus ChatGPT to inspect after a real
Colab run. See
["Primary capacity estimator"](#primary-capacity-estimator-engine-side-token-counter-throughput)
below for the exact estimator and fail-closed contract.

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

For `output-heavy` at higher concurrency, keep the (default, enabled)
initial-admission ramp — see
["Deterministic initial-admission ramp"](#deterministic-initial-admission-ramp)
below — and inspect the `completion_clustering` block in
`point_summaries.jsonl` before drawing any conclusion:

```bash
python hack/benchmark/colab_profiling/profile_bucket.py \
  --profiling-jsonl /content/wva-1546-prompts/profiling.jsonl \
  --base-url http://127.0.0.1:8000 \
  --bucket output-heavy \
  --model Qwen/Qwen2.5-3B \
  --tokenizer-revision 3aab1f1954e9cc14eb9509a215f9e5ca08227a9b \
  --gpu-memory-utilization 0.90 \
  --concurrency 64,96,128,160,192 \
  --settling-seconds 60 \
  --measurement-seconds 180 \
  --ramp-admission-interval-seconds 0.05 \
  --output-dir /content/wva-1546-output-heavy-profiling
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
precondition re-check, and finally the summary/artifact write. **The
precondition check fails closed**: if it fails, the point does not start
telemetry, does not create request workers, and does not admit a single
request — it returns an immediately invalid, zero-execution point summary
(`requests_submitted = 0`, `max_observed_concurrency = 0`,
`completed_requests_in_window = 0`, `completed_requests_per_second = 0`,
`completed_total_tokens_per_second = 0`, `outstanding_at_t1 = 0`,
`outstanding_after_drain = 0`, `execution_skipped = true`) with the concrete
precondition failure reason preserved in `invalidation_reasons` and
`execution_skipped_reason`. Only requests whose **server-validated**
completion lands inside `[T0, T1)` are counted in the numerator (D3/D10);
settling and drain completions are retained only as bookkeeping.

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

### Primary capacity estimator: engine-side token-counter throughput

The PRIMARY `V_M` capacity estimator for every bucket is:

```
V_hat_M =
  (
    delta(vllm:prompt_tokens_total)
    +
    delta(vllm:generation_tokens_total)
  )
  /
  telemetry_duration_s
```

using the **first and last valid** (`status == "ok"`) vLLM `/metrics`
samples whose timestamp falls inside the exact measurement interval
`[T0, T1)`. This is recorded per point as the `engine_token_throughput`
block in `point_summaries.jsonl`, and it — not terminal-completion
throughput — is the basis for `adjacent_throughput_gain` and the review
table's headline numbers.

**Why this is preferred over assigning full request work by terminal
timestamp.** The previous (still-retained, see below) estimator counts a
request's entire `L_in + L_out` work at the instant its response
terminates. For short-output buckets this is a reasonable approximation,
but for long-output workloads (like `output-heavy`, `L_out = 384`) a
request's generation can span a large fraction of — or cross — the
`[T0, T1)` boundary. Terminal-completion accounting then assigns either
ALL or NONE of that request's tokens to the window, purely based on which
side of the boundary its one terminal timestamp lands on, while the engine
itself was actually processing that request's tokens smoothly throughout.
Real evidence showed this boundary effect is large: roughly -11% to +9%
differences between the two estimators across output-heavy concurrency
points. Engine counters increase as work is processed and do not require a
request to terminate inside the window at all, so they are not subject to
this aliasing.

**Fail-closed contract (mandatory, no silent fallback).** A point's
`engine_token_throughput.available` is `false` — and the point is
therefore marked invalid via an explicit
`engine_token_throughput_unavailable:<reason>` invalidation reason — if any
of:

* `vllm:prompt_tokens_total` or `vllm:generation_tokens_total` is missing
  from either bracket sample (`*_unavailable`);
* fewer than two valid, in-window telemetry samples exist
  (`fewer_than_two_valid_telemetry_samples_in_window`);
* a counter resolves to **more than one** distinct labeled series within
  one telemetry snapshot (`ambiguous_metric_series:<name>`) — this
  profiler assumes exactly one engine/model/TP rank, so an ambiguous
  series is never summed or guessed;
* the selected series' labels differ between the first and last bracket
  sample (`*_series_identity_changed`) — the identity of "the" counter
  must be stable across the window;
* either counter's last value is less than its first value
  (`counter_reset_detected`) — counters are monotonic; a decrease is
  treated as a reset/restart and is **never** wrapped or absolute-valued;
* the resulting telemetry duration is non-positive
  (`non_positive_telemetry_duration`).

The engine estimator is **never** silently replaced by the secondary
completion estimator when unavailable. Consequently, `--no-telemetry` (or
any run where the server does not expose `vllm:prompt_tokens_total`/
`vllm:generation_tokens_total`) makes every point invalid.

Each `engine_token_throughput` record also carries
`telemetry_first_timestamp`/`telemetry_last_timestamp`/
`telemetry_duration_s` (the **actual** telemetry bracket, never assumed to
coincide with `T0`/`T1` or with the nominal `measurement_seconds`),
`prompt_tokens_start/end/delta`, `generation_tokens_start/end/delta`,
`total_tokens_delta`, `prompt_tokens_per_second`,
`generation_tokens_per_second`, `total_tokens_per_second`, and
`selected_series` (the exact metric name and labels used, for
auditability) — or `unavailable_reason` when unavailable.

**The terminal-completion estimator is retained, not deleted, as a
secondary diagnostic.** `completed_requests_in_window`,
`completed_requests_per_second`, and `completed_total_tokens_per_second`
are still computed and recorded every point (labeled
`completion_throughput_role: "secondary_boundary_sensitive_completion_diagnostic..."`).
Each point also carries a `completion_vs_engine` block
(`completion_vs_engine_ratio`, `completion_vs_engine_percent_difference`)
comparing the two, purely for human review — it never influences
`run_valid` or `adjacent_throughput_gain`.

**Saturation/ceiling evidence.** Each point's `saturation_ceiling_evidence`
block records `concurrency_target`, `max_observed_running`, and
`max_observed_waiting` from the same telemetry. A persistent running-request
ceiling below `concurrency_target` together with a nonzero waiting
population is evidence the *server* (not the client) is limiting
concurrency — e.g. a `max_num_seqs`-style engine/config limit — and is
saturation/config-limit evidence for HUMAN REVIEW. This module does not
infer or hard-code any universal ceiling value from it and never declares a
plateau from it automatically.

### Output-heavy completion-wave risk (estimator quality vs. measurement validity)

Real output-heavy profiling exposed a suspicious pattern: several
concurrency points completed near-exact multiples of the target concurrency
`C` within one measurement window (e.g. `completed = 2*C`), with an
apparently *non-monotonic* throughput curve across concurrency, despite
**zero request failures, zero preemptions, GPU ~100% utilization, and
bounded drains** — i.e. every point was individually **valid** by every
existing check.

**Why equal-length, long-output workloads can phase-synchronize.** Every
bucket's records share one fixed `target_output_tokens` (that is the
bucket's definition), so this risk is not unique to `output-heavy` in
principle. What makes it acute for `output-heavy` specifically is the
combination of (a) the closed-loop scheduler's initial admission loop,
which historically submitted all `C` initial requests via a tight, zero-
delay `for` loop (see "Deterministic initial-admission ramp" below), and
(b) `output-heavy`'s long, uniform decode length (`L_out = 384`): requests
admitted within the same narrow burst, with identical prompt/decode
lengths, can progress through decode together and terminate close together.
Their closed-loop replacements can then re-enter together too, preserving a
"completion wave" structure indefinitely. A single such wave at `C=160`
represents `160 * 512 = 81,920` logical tokens; gaining or losing exactly
one wave inside a 180s window shifts the estimated rate by roughly `81,920
/ 180 ≈ 455 token/s` — large enough, relative to the ~1.2–1.8k token/s
capacities already measured for the other buckets, to plausibly explain the
observed non-monotonic curve as a boundary-aliasing artifact rather than a
real capacity effect.

**Measurement validity is not the same thing as estimator quality.** A
`[T0, T1)` window that only ever samples 1–2 discrete completion waves is
still a *valid* measurement by every existing check (D15): the requests
that did complete were server-validated, the drain was bounded, nothing
failed. But a rate estimated from 1–2 wave-sized samples is a much noisier
estimate of *steady-state* throughput than the same request count spread
smoothly across the window would be. This tool distinguishes the two
concerns explicitly: `run_valid` (and `invalidation_reasons`) answer "is
this data trustworthy at all", while the new `completion_clustering` block
(below) answers "how much should a human trust this point as a *smooth*
steady-state estimate" — and only the former ever gates anything
automatically.

**Completion-burst diagnostic (fixed-width sliding window, non-chaining).**
Every point summary includes a `completion_clustering` block, computed
purely from that point's own `request_results.jsonl` terminal timestamps
(reusable directly against a real artifact; no GPU required):

* `completion_count` — exactly the population used for the capacity
  numerator (`in_measurement_window and passed`); always equal to that same
  point's `completed_requests_in_window`.
* `inter_completion_gap_seconds` — `{available, count, min, max, avg}` over
  consecutive sorted terminal timestamps, or `{"available": false}` if
  fewer than two completions occurred. Diagnostic evidence only; it never
  decides anything by itself.
* `burst_window_seconds` / `max_completions_in_burst_window` /
  `max_burst_fraction_of_concurrency` — the maximum number of completions
  found in **any** fixed-width window of `burst_window_seconds` (default
  `0.5s`, configurable via `--burst-window-seconds`), computed by an exact
  O(n) two-pointer sliding-window scan over every possible window position,
  and that count as a fraction of target concurrency `C`. **Boundary
  semantics are exact and non-chaining**: for a window anchored at
  `window_start`, a timestamp `t` is inside the window iff
  `window_start <= t <= window_start + burst_window_seconds`. Critically,
  membership is always relative to the window's own fixed anchor, never to
  a chain of neighbors — a timestamp can never pull in another timestamp
  more than `burst_window_seconds` away from that anchor.
* `phase_synchronization_suspected` — `true` iff
  `max_burst_fraction_of_concurrency >= near_concurrency_burst_threshold_fraction`
  (default `0.8`, configurable via
  `--near-concurrency-burst-threshold-fraction`).
* `repeated_burst_episodes` (secondary, simpler evidence) —
  `{episode_count, episode_sizes, near_concurrency_episode_count}`: sorted
  timestamps are greedily partitioned into **non-overlapping**
  `burst_window_seconds`-wide episodes (each anchored at the next
  unassigned timestamp), giving a human a quick read on whether *multiple,
  separated* near-`C` bursts occurred (matching the real repeated
  `completed = k*C` observation) without reintroducing chaining.

**Corrected from an earlier defect: do not use single-linkage/chained
clustering here.** This diagnostic originally grouped timestamps by
nearest-neighbor-chain ("single-linkage") clustering: a timestamp joined a
cluster iff it was within a tolerance of the immediately *previous*
timestamp already in that cluster. That chains transitively — a perfectly
healthy, continuous, high-throughput completion stream with small adjacent
gaps (e.g. a steady ~3–4 req/s output-heavy stream, inter-completion gaps
~0.25–0.33s, all comfortably under a 0.5s tolerance) would be merged into
**one arbitrarily large "cluster" spanning the entire measurement window**,
producing a **false-positive** `phase_synchronization_suspected` verdict on
exactly the real regime this diagnostic exists to check. The fixed-width
sliding-window algorithm above cannot do this: a continuous stream with
gaps smaller than the window still only ever contributes a small, bounded
count to any one window (e.g. ~2–3 completions per 0.5s window at a
0.25–0.33s cadence), never the whole stream.

The now-removed fields `cluster_tolerance_seconds`, `cluster_count`,
`cluster_sizes`, `largest_cluster_size`,
`largest_cluster_fraction_of_concurrency`, and
`near_concurrency_cluster_count` (and the corresponding
`--cluster-tolerance-seconds` /
`--near-concurrency-cluster-threshold-fraction` flags) had misleading
semantics under the corrected algorithm and have been renamed rather than
kept for compatibility; no real-hardware artifact depended on the old
names, since `output-heavy` had not yet been validated at the time of the
rename (it has since been HUMAN-REVIEWED and ACCEPTED — see
["Supported buckets and validation status"](#supported-buckets-and-validation-status)).

**This diagnostic is evidence for HUMAN REVIEW only.** It is computed and
recorded for every point (including a trivial all-zero record for a
precondition-skipped point), but it **never** changes `run_valid`, and this
tool never uses it to automatically accept, reject, or select a plateau.

**Resolution (ramped re-profiling WAS performed).** The startup ramp
below and this diagnostic were both exercised on real output-heavy Tesla T4
hardware. With ramping enabled, the diagnostic found NO near-concurrency
completion-wave synchronization — the originally observed near-exact
`completed = k*C` pattern did not recur. This rules out initial-admission
phase synchronization as the primary cause. However, `completed_total_tokens_per_second`
still differed materially from `engine_token_throughput.total_tokens_per_second`
at several concurrency points (roughly -11% to +9%), confirming the
terminal-completion estimator is genuinely boundary-sensitive for this
bucket independent of any synchronization artifact. That evidence is why
engine-side counter throughput is now the PRIMARY estimator (see above) for
every bucket, not just `output-heavy`.

**Chronology to the current accepted `output-heavy` value.** In order: (1)
the initial completion-based estimator produced a suspicious non-monotonic
curve, described above; (2) the startup ramp plus the completion-wave
diagnostic ruled out initial-admission phase synchronization as the cause —
no near-concurrency completion bursts were found once ramping was enabled;
(3) engine counters replaced completion throughput as the PRIMARY estimator
for every bucket, since the remaining discrepancy was explained by
terminal-completion boundary aliasing, not phase synchronization; (4) the
`C=160` local dip was independently repeated on real hardware (~1529.9, then
~1443.3) and treated as reproducible real-runtime scheduler/batching
behavior rather than a measurement failure; (5) `C=256`/`C=288` saturation
evidence was subsequently obtained (~1878.2/~1884.0 and ~1839.2 tok/s
respectively). `output-heavy` was then HUMAN-REVIEWED and ACCEPTED at
`V_M^(output-heavy) ~= 1.88k` logical token/s — see
["Supported buckets and validation status"](#supported-buckets-and-validation-status)
for the complete accepted evidence trail. The lesson from this history
(terminal-completion accounting is boundary-sensitive for long-output
buckets; engine counters are not) is why engine-side counter throughput is
now mandatory for every bucket, not only `output-heavy`.

### Deterministic initial-admission ramp

The closed-loop scheduler's initial fill — the very first `C` admissions of
a load point, before settling begins — historically submitted all `C`
requests via a tight, zero-delay `for` loop. This is architecture-level
evidence supporting the phase-synchronization hypothesis above, independent
of the diagnostic: submitting `C` requests within a sub-millisecond window
is exactly the condition under which equal-length requests can decode in
lockstep.

`TimingConfig.ramp_admission_interval_seconds`
(`--ramp-admission-interval-seconds`, default `0.05s`) paces those specific
`C` initial admissions one at a time, this many seconds apart, instead of
submitting them in one burst:

* **Only the initial `C` admissions are paced.** Every later replacement
  admission — issued the instant a request completes, during settling,
  measurement, or drain — remains completely immediate, exactly as before.
  There is no think-time inserted anywhere during measurement (D7
  requirement 5).
* **Settling begins only once the full target concurrency has actually been
  reached.** `t_start` (settling start) is captured *after* the ramp
  finishes admitting all `C` requests, not before. `T0`/`T1` are derived
  from `t_start` exactly as before (`T0 = t_start + settling_seconds`,
  `T1 = T0 + measurement_seconds`), so ramp time is never counted as
  settling or measurement time and the measurement denominator
  (`measurement_seconds`) is completely unaffected.
* **Concurrency is never exceeded.** The ramp does not change target
  concurrency, `[T0, T1)` semantics, the request contract, or bucket
  records; it only changes the wall-clock pacing of the first `C`
  admissions.
* **`0` reproduces the original immediate-burst admission exactly** — for
  exact A/B comparison against pre-ramp artifacts, or if an operator wants
  to deliberately reproduce the burst to confirm the diagnostic detects it.
* **Deterministic and non-adaptive.** The interval is a fixed, explicit,
  operator-configured constant. It is never derived from observed model
  latency or any other feedback signal, so the experimental procedure stays
  reproducible.
* **Chosen to stay small relative to settling/measurement.** For the
  default concurrency ladder (max `C=32`), the default `0.05s` interval
  adds at most `(32-1)*0.05 ≈ 1.6s` before settling — negligible next to the
  default 30s settling window. For a `C=192` output-heavy point, it adds
  `(192-1)*0.05 ≈ 9.6s` — still well under the default 30s settling and
  60–180s measurement windows, so the ramp itself should not materially
  change the server's steady-state operating condition.

Every point summary's `startup_ramp` block records
`ramp_admission_interval_seconds`, `ramp_enabled`, `ramp_start_s`,
`target_concurrency_reached_s`, and `ramp_duration_s` for auditability;
completions that finish before `target_concurrency_reached_s` are labeled
with `phase: "ramp"` in `request_results.jsonl`. The experiment manifest
records the configured policy under `startup_ramp` (and the clustering
diagnostic's configuration under `completion_clustering_diagnostic`) so a
different ramp/diagnostic configuration is always visible in the artifact,
never silently reinterpreted.

### Known limitations / risks not yet resolved by local tests

* **Resolved defect (fail-closed preconditions):** a real fresh-runtime
  input-heavy repeatability run once started a `C=48` point while the vLLM
  server was briefly unreachable. The precondition probe correctly failed
  with `Connection refused`, but `run_load_point()` still entered the
  closed-loop executor and generated approximately 25,069 immediately-failing
  HTTP requests against the down server before the bounded window elapsed;
  the point was still (correctly) discarded as invalid, but it should never
  have generated that traffic. This is fixed: a failed precondition now
  prevents telemetry from starting and prevents any request worker or HTTP
  request from being created for that point at all (see above). This does
  **not** change `[T0, T1)` semantics, concurrency semantics, the request
  contract, capacity arithmetic, or telemetry interpretation for points that
  do execute.
* The closed-loop scheduler uses a bounded Python thread pool (not
  `asyncio`) as the concurrency-limiting mechanism, since this directory's
  existing components deliberately use only the Python standard library.
  This has since been exercised on real vLLM/Tesla T4 hardware up through
  `C=288` (see ["Supported buckets and validation status"](#supported-buckets-and-validation-status));
  local unit tests still only exercise mocked HTTP transports.
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
* vLLM 0.28.0's `/metrics` surface has since been observed on real
  hardware, and `vllm:prompt_tokens_total`/`vllm:generation_tokens_total`
  are the basis for every accepted `V_M` value above. Parsing remains
  generic Prometheus-text parsing plus a best-effort list of commonly
  expected `vllm:*` metric names (with a
  `kv_cache_usage_perc`/`gpu_cache_usage_perc` name fallback), each
  explicitly marked present/absent, since not every `vllm:*` metric this
  harness looks for has necessarily been confirmed present on every real
  run.
* `--gpu-memory-utilization` and `--prefix-caching` (and `--dtype`,
  `--tensor-parallel-size`, `--max-model-len`, `--generation-config`) are
  **operator-declared** values recorded in the manifest as-is; the profiler
  cannot independently inspect the running server's actual launch flags
  beyond `/v1/models` identity and a best-effort `/version` probe, and it
  never modifies the running server.
* **Behavior change:** since engine-counter throughput is now mandatory,
  `--no-telemetry` makes every point invalid
  (`engine_token_throughput_unavailable:telemetry_not_collected`). Use
  `--no-telemetry` only for harness debugging, never to produce capacity
  evidence.
* **Resolved defect (non-chaining completion-burst diagnostic):** the
  completion-clustering diagnostic originally used single-linkage/chained
  clustering, which could merge a perfectly healthy, continuous completion
  stream (small adjacent gaps, e.g. the real ~3–4 req/s output-heavy
  regime) into one arbitrarily large false-positive "cluster" spanning the
  whole window — see "Output-heavy completion-wave risk" above. This is
  fixed: the diagnostic now uses a non-chaining, fixed-width sliding-window
  computation (`max_completions_in_fixed_window`) where membership is
  always bounded by distance to a fixed window anchor, never by transitive
  adjacency. This is a diagnostic-only change: it does not affect the
  startup ramp, `[T0, T1)` semantics, capacity arithmetic, or `run_valid`.
* **Adopted fix (engine-counter throughput is now the primary estimator):**
  ramped real-runtime re-profiling of `output-heavy` WAS performed on real
  Tesla T4 hardware. The non-chaining completion-burst diagnostic found NO
  near-concurrency completion-wave synchronization after ramping, ruling
  out startup phase-synchronization as the main explanation. However, the
  secondary terminal-completion estimator still differed materially from
  vLLM's own engine-side counters (roughly -11% to +9% across concurrency
  points) — expected for a long-output bucket, since terminal-completion
  accounting assigns a request's full total-token work to its single
  terminal timestamp, while a request whose generation spans a `[T0,T1)`
  boundary should contribute smoothly throughout the window. Engine-side
  counter throughput (`engine_token_throughput.total_tokens_per_second`) is
  therefore now the PRIMARY capacity estimator for every bucket, and is
  MANDATORY for a valid point (see "Primary capacity estimator" above); a
  point without it is invalidated rather than silently falling back to
  completion throughput.
* **Resolved (all three buckets now VALIDATED and ACCEPTED):**
  `output-heavy`'s C=160 local dip (~1529.9, independently repeated at
  ~1443.3) and `balanced`'s C=160 local dip (~1623.2, repeat ~1624.7) were
  both independently reproduced on two runs each, and are treated as
  reproducible real-runtime scheduler/batching behavior -- not a
  measurement failure -- since they do not affect the observed maximum/
  saturated region. `V_M^(input-heavy) ~= 2.18k`, `V_M^(balanced) ~= 1.97k`,
  and `V_M^(output-heavy) ~= 1.88k` logical token/s are all HUMAN-REVIEWED,
  ACCEPTED engine-counter values; see `BUCKET_VALIDATION_STATUS` in
  `profile_bucket.py` for the complete per-point evidence trail. This
  module still never infers or auto-accepts any capacity value itself.
* The `0.05s` default ramp interval, and the `0.5s` default burst window /
  `0.8` default near-concurrency threshold, remain reasoned defaults
  (documented above); they were not altered by adopting the engine-counter
  estimator.
* The completion-burst diagnostic characterizes *client-observed*
  completion timing only; it does not independently confirm *why* requests
  clustered (e.g. it cannot distinguish "vLLM's continuous batching genuinely
  advances equal-length sequences in lockstep" from any other server-side
  cause of correlated completions). It is deliberately scoped as
  descriptive evidence, not a root-cause proof.
* The **automated unit test suite** in this directory never touches a real
  network: all local tests use fake/mocked HTTP transports and fake
  GPU/telemetry samplers. Real-network, real-GPU evidence for the accepted
  `V_M` values above comes only from the Colab/Kaggle runs described in
  ["Supported buckets and validation status"](#supported-buckets-and-validation-status),
  not from this test suite.

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
`test_validate_mixed_workload.py` covers the open-loop mixed-workload
harness described below (composition math, deterministic scheduling,
fail-closed client/scheduling-lag/drain-chaining behavior, and artifact
writing).

## Mixed-workload composition validation

Pure single-bucket profiling above establishes that each bucket's
monolithic `V_M^(b)` is independently *repeatable*. It does **not** prove
those numbers are *useful*: it says nothing about whether they predict
anything once buckets are mixed together. That is the next, decisive
research gate, implemented in `validate_mixed_workload.py`.

### The hypothesis under test

```
rho_pred = sum_b lambda'_b / V_M^(b)
lambda'_b = request_rate_b * W_b        (W_b = L_in + L_out for bucket b)
```

Term by term:

* `lambda'_b` — the offered *logical token rate* for bucket `b`: the rate at
  which that bucket's requests are arriving, multiplied by how many logical
  tokens (`W_b = L_in + L_out`) each one costs;
* `V_M^(b)` — the independently profiled, human-accepted sustainable
  capacity for that same bucket (see
  ["Supported buckets and validation status"](#supported-buckets-and-validation-status));
* `lambda'_b / V_M^(b)` — the normalized capacity *demand* that bucket `b`
  alone would contribute if it had the whole GPU to itself;
* `rho_pred` — the sum of those per-bucket demands: the total predicted
  normalized pressure across all buckets sharing the one GPU at once.

Interpretation:

* `rho_pred < 1` — the mix is predicted to sit below capacity;
* `rho_pred ~= 1` — the predicted saturation boundary;
* `rho_pred > 1` — the mix is predicted to produce sustained
  overload/backlog.

**This is an experimental hypothesis to be validated by mixed-load runs, not
a mathematical guarantee.** If `rho_pred` genuinely predicts a mixed
workload's saturation boundary, then offering a mix of buckets at a
combined `rho_pred` materially below 1 should look stable (no persistent
backlog growth), `rho_pred` near 1 should show onset/boundary behavior, and
`rho_pred` above 1 should show persistent accumulation of waiting/
outstanding work. **This experiment is the test of that hypothesis, not a
confirmation of it.** Pure-bucket profiling being valid does not itself
prove this equation holds.

### This is OPEN LOOP, not closed loop

Unlike `profile_bucket.py`'s fixed-concurrency closed loop, the independent
variable here is **arrival rate**, not a concurrency cap:

* each participating bucket gets its own deterministic, absolute-time
  arrival schedule: `scheduled_time(k) = point_start + phase_b + k /
  request_rate_b`, computed directly from `(origin, phase, rate, k)` --
  **never** as `previous_actual_time + interval`, which would let runtime
  delay silently shrink the achieved rate;
* a bucket's next arrival is scheduled purely from its own plan and never
  waits for its previous arrival to complete, so the generator keeps
  offering load while requests are still outstanding -- exactly what is
  needed to observe genuine server-side backlog;
* the HTTP client's concurrency budget (thread-pool size) is an explicit,
  generous, recorded value (`--client-concurrency-budget`, default 4096)
  deliberately set far above any expected server-side backlog, so client
  thread-pool queueing can never masquerade as vLLM's own queueing; if the
  budget is ever actually reached, the point is invalidated
  (`client_concurrency_budget_exceeded`) rather than silently reinterpreted
  as server saturation.

Target arrival rate and achieved (actually observed) arrival rate are both
recorded per bucket per point (`rho_pred_target` and `rho_pred_achieved`),
using the SAME `rho_pred` formula, so they are directly comparable.

### Composition weights

Weights (`alpha_b`, normalized to sum to 1) describe each bucket's
*intended contribution to `rho_pred`* -- **not** equal request counts and
**not** equal raw token rates:

```
rho_b       = target_rho * alpha_b
lambda'_b   = rho_b * V_M(b)
request_rate_b = lambda'_b / W_b
```

The first experiment used **equal normalized capacity contribution**
(`input-heavy : balanced : output-heavy = 1 : 1 : 1`) at target `rho` =
**0.70, 1.00, 1.15** (fully overridable via `--target-rho`; these are not
the only supported values). Two skewed compositions were subsequently run
using the same three `V_M(b)` values: an **input-heavy-dominant** mix
(`alpha = 0.70 / 0.15 / 0.15`) and an **output-heavy-dominant** mix
(`alpha = 0.15 / 0.15 / 0.70`) — see
["Composition validation evidence"](#composition-validation-evidence)
below. `V_M(b)` capacities are always an explicit `--capacity BUCKET:VALUE`
input, recorded verbatim in the manifest -- never a hard-coded constant.

### Saturation/backlog evidence is human-reviewed, not auto-decided

Every point records, in addition to the mandatory `engine_token_throughput`
estimator (the identical, unmodified contract from `profile_bucket.py`):
waiting-population mean/min/max/first-and-last-window medians/linear
trend, running-population mean/max/observed-ceiling fraction (never a
hard-coded 256), client outstanding-request counts at `T0`/`T1`, and
request/failure/drain evidence. The harness does **not** contain logic like
`if rho < 1 and waiting == 0: model_valid = true`; it produces a compact
human review table (`target_rho`, `achieved_rho`, request/engine rates,
waiting/running/outstanding evidence, `run_valid`) and stops there.
Expected overload/backlog under `rho >= 1` is NOT itself a run-invalid
condition -- only a harness/precondition/client-generator failure
(unreachable server, unsustainable client schedule, a prior point that
failed to drain) is.

### Point lifecycle

Precondition check → (fail closed if the *previous* point did not fully
drain) → start all bucket arrival streams → settling → `T0` → same arrival
streams continue → `T1` → stop new arrivals → bounded drain → atomic
artifact write. The server is never reset between points; every point
simply requires the previous one to have drained successfully first.

### First experiment: Kaggle command

```bash
python hack/benchmark/colab_profiling/validate_mixed_workload.py \
  --profiling-jsonl /kaggle/working/wva-1546-prompts/profiling.jsonl \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen2.5-3B \
  --tokenizer-revision 3aab1f1954e9cc14eb9509a215f9e5ca08227a9b \
  --vllm-version 0.28.0 \
  --dtype float16 \
  --tensor-parallel-size 1 \
  --max-model-len 1024 \
  --generation-config vllm \
  --gpu-memory-utilization 0.90 \
  --capacity input-heavy:2180 \
  --capacity balanced:1965 \
  --capacity output-heavy:1880 \
  --weight input-heavy:1 \
  --weight balanced:1 \
  --weight output-heavy:1 \
  --target-rho 0.70 --target-rho 1.00 --target-rho 1.15 \
  --settling-seconds 60 \
  --measurement-seconds 180 \
  --metrics-interval-seconds 1 \
  --drain-timeout-seconds 300 \
  --request-timeout-seconds 600 \
  --output-dir /kaggle/working/wva-1546-mixed-equal-weight
```

### Composition validation evidence

Three compositions have been run so far, all using the same three
independently profiled `V_M(b)` values (`V_M^(input-heavy) ~= 2.18k`,
`V_M^(balanced) ~= 1.97k`, `V_M^(output-heavy) ~= 1.88k` logical token/s):

| Composition (`alpha` input/balanced/output) | Predicted `rho=1` capacity | Observed boundary/saturated capacity | Approx. error |
| --- | ---: | ---: | ---: |
| Equal (`1/3, 1/3, 1/3`) | ~= 2008 tok/s | ~= 1996–2075 tok/s | within a few percent |
| Input-heavy dominant (`0.70, 0.15, 0.15`) | ~= 2103 tok/s | ~= 2093–2107 tok/s | nearly exact |
| Output-heavy dominant (`0.15, 0.15, 0.70`) | ~= 1938 tok/s | ~= 1868 tok/s | over-predicted by ~= 3.6% |

Across these three tested compositions, observed error is roughly within
±4%. **Do not overstate this beyond the exact tested configuration**: this
is evidence from three specific composition ratios, on one fixed Qwen2.5-3B
/ Tesla T4 / vLLM 0.28 monolithic serving configuration, using the three
fixed `L_in`/`L_out` buckets defined above. Stated precisely:

> Within this fixed Qwen2.5-3B / Tesla T4 / vLLM 0.28 monolithic
> configuration and the three tested work-shape buckets, the independently
> measured `V_M` values produced mixed-workload saturation predictions
> within roughly a few percent, across the three composition ratios tested
> so far.

### "Held-out" means held-out composition, not (yet) a held-out prompt split

This is an important distinction for anyone reading the evidence above.
Every mixed-workload run so far, including all three compositions in the
table above, was invoked with `--profiling-jsonl .../profiling.jsonl` --
the **same prompt pool** already used by the pure-bucket profiling runs
that produced the `V_M(b)` values being tested. So:

* "held-out" in the results above refers to **held-out workload
  composition** -- the equal, input-heavy-dominant, and output-heavy-
  dominant *mixture ratios* were not used to derive `V_M(b)` and are a
  genuine out-of-sample test of the composition equation;
* it does **not** mean a held-out **prompt split**. `heldout.jsonl` (see
  ["Artifacts and contract"](#artifacts-and-contract)) exists specifically
  as a separate prompt pool for exactly this purpose, but it has **not**
  been used to produce any of the mixed-GPU evidence reported here;
* a natural robustness follow-up -- not yet performed, and not claimed
  above -- is to repeat one or more of these mixed compositions using
  `--profiling-jsonl .../heldout.jsonl` instead, to rule out reuse of the
  same prompt pool as a confound. This is a suggested next step, not a
  result.

### Experiment scope: what this does and does not establish

This experiment, as run so far, **does** support:

* bucket-specific work-shape capacity is measurable and repeatable under
  this fixed serving configuration;
* independently measured bucket capacities approximately compose (within
  roughly a few percent) for the three tested workload mixtures;
* input/output token shape matters even when total `W` is held fixed at
  512 -- `V_M` differs meaningfully across the three buckets despite equal
  `W`.

This experiment, as run so far, **does not** establish:

* natural-language / semantic workload generalization;
* arbitrary prompt distributions;
* arbitrary `L_in`/`L_out` bucket definitions beyond the three tested;
* unequal-`W_b` mixed composition validity on real hardware -- the harness
  supports different `W_b` per bucket mathematically (see the `rho_pred`
  formula above), but **all three currently validated buckets have
  `W = 512`**, and mixed composition with genuinely different `W_b` values
  has not yet been validated on real hardware;
* another model, another GPU, or another vLLM configuration;
* P/D disaggregated decoder `V_D`;
* SLO-safe serving capacity;
* production autoscaling correctness.

### Synthetic work-shape isolation vs. real workload prediction

This phase intentionally isolates **length shape** from semantic content.
The model producing arbitrary, semantically meaningless text from synthetic
token-ID prompts is acceptable *because output semantics are not the
variable under study here* -- see
["Exactly what is sent to the model"](#exactly-what-is-sent-to-the-model).

However, before any of this can support a claim that a production
autoscaler can classify or predict *real* incoming workloads, later
research must connect real prompts and/or real runtime signals to these
work-shape regimes. That connection -- predictor design, predictor-less
approaches, hidden-state-based approaches, or any other classification
mechanism -- is explicitly out of scope here; this document only states the
boundary, it does not cross it.

### Next step

The input-heavy-dominant and output-heavy-dominant compositions above were
run specifically to test whether the equal-weight result was accidental,
and the composition equation held reasonably well (within roughly ±4%)
across all three. The most useful next step is the `heldout.jsonl` repeat
described above, to rule out prompt-pool reuse as a confound, before
treating this composition evidence as more broadly reliable. Predictor
design and autoscaling controller logic both remain explicitly deferred
beyond this validation gate.

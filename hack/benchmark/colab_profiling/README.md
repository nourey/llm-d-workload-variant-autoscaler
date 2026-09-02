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
| `input-heavy` | 384 | 128 | 512 | **VALIDATED** on independent real Tesla T4 hardware |
| `output-heavy` | 128 | 384 | 512 | **NOT YET VALIDATED** (see below) |

**Balanced-bucket result (already validated, do not re-derive from local
tests):** two independent 180s confirmation runs, on independent Tesla T4
instances, under the identical serving configuration documented below,
reproduced:

```
C=48: 1228.8 logical token/s
C=64: 1274.3 logical token/s   (adjacent gain ~3.7%)
```

giving a provisional empirical monolithic capacity
`V_M^(balanced) ≈ 1274 logical token/s`.

**Input-heavy result (already validated, do not re-derive from local
tests):** validated on independent real Tesla T4 hardware runs under the
same monolithic non-P/D serving configuration, giving a provisional
empirical monolithic capacity `V_M^(input-heavy) ≈ 1820 logical token/s`.

Both conclusions were reached by HUMAN REVIEW of real Colab evidence, not by
this repository's code.

**Output-heavy is NOT YET VALIDATED.** Real-runtime profiling exposed a
suspected initial-admission phase-synchronization / completion-wave
aliasing artifact — see
["Output-heavy completion-wave risk"](#output-heavy-completion-wave-risk-estimator-quality-vs-measurement-validity)
below before running or interpreting output-heavy results. Do not treat any
output-heavy number as validated until this has been resolved on real
hardware and reviewed the same way as balanced/input-heavy.

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
names, since output-heavy has not yet been validated.

**This diagnostic is evidence for HUMAN REVIEW only.** It is computed and
recorded for every point (including a trivial all-zero record for a
precondition-skipped point), but it **never** changes `run_valid`, and this
tool never uses it to automatically accept, reject, or select a plateau.

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
* `output-heavy` has a suspected phase-synchronization/completion-burst
  artifact that has NOT been re-profiled on real hardware with the ramp
  enabled; see "Output-heavy completion-wave risk" above. `balanced` and
  `input-heavy` are both validated; `output-heavy` is not.
* The `0.05s` default ramp interval, and the `0.5s` default burst window /
  `0.8` default near-concurrency threshold, are reasoned defaults
  (documented above), not values confirmed against real output-heavy
  hardware evidence yet — re-profiling output-heavy with ramping enabled
  has not been performed as part of this change.
* The completion-burst diagnostic characterizes *client-observed*
  completion timing only; it does not independently confirm *why* requests
  clustered (e.g. it cannot distinguish "vLLM's continuous batching genuinely
  advances equal-length sequences in lockstep" from any other server-side
  cause of correlated completions). It is deliberately scoped as
  descriptive evidence, not a root-cause proof.
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

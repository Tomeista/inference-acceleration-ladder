# ladder

A bits-per-weight speed curve for Qwen3-8B, measured on llama.cpp across 32
quantization widths from 16.01 down to 2.22 bits.

A companion to the `bench` study, which measures seven vLLM servers across
three axes (quantization, sparsity, speculation). This one measures a single
axis as densely as the published checkpoints allow.

This repository is **self-contained**: clone it and run it, nothing else
required. The measurement code it shares with that study -- load client,
percentile math, gauge sampling, MLflow structure -- is vendored under
`src/ladder/harness/`, and the prompt sets are copied in with their digests.

## Why a second study rather than more configs in the first

vLLM's low-precision floor is not 4 bits, it is *4 bits where a Marlin- or
Machete-class kernel exists*. GPTQ, AWQ and compressed-tensors INT4, and the
FP4 microscaling formats on new enough silicon, all have one. INT3 and INT2 can
be written into those checkpoint formats and have no kernel to run on. So the
vLLM ladder has three rungs — 16, 8, 4 — and there is no fourth to add.

llama.cpp has two dozen, all with working CUDA kernels, and for Qwen3-8B every
one is already published. That is the entire reason for the engine change: not
that llama.cpp serves better, but that it is the only place the question "how
does speed move with bit width" can be asked at more than three points.

The cost is that the engine is now a variable. Which is why:

**The two studies are never merged into one table.** They live in separate
MLflow experiments (`inference-acceleration` and `inference-acceleration-ladder`).
Each repository defaults to its own `mlflow.db`; to read the anchor across both
in one query, point them at a shared file with `--tracking-uri`. The experiment
names keep them from pooling by accident. Every run here is tagged
`engine=llama.cpp`.

**BF16 is run on both engines as an anchor.** It is the only configuration the
two studies share. Without it, every llama.cpp number differs from every vLLM
number by both quantization *and* engine with no way to separate them. The
anchor is not a point on the curve; it is the tie-off, and it is grouped
separately for that reason. See the RUNBOOK on why the vLLM half of this is in
doubt on the current hardware.

**Both engines see byte-identical prompts.** This package builds no prompt sets.
`prompts/` is a copy of the frozen files the vLLM study measured, carried with
its `manifest.json`, and every run records the same manifest digest. Because
they are now a copy rather than a shared directory, a test hashes each file and
fails if it is not the bytes the manifest recorded -- a regenerated or mangled
prompt set would otherwise load fine and quietly invalidate the anchor.

## Hardware

The measurements run on `dell-gb10-1`: an NVIDIA **GB10** (Grace-Blackwell,
compute capability 12.1, aarch64, 20 cores) with **119 GB of unified LPDDR5X**
shared between CPU and GPU. CUDA 13.0, driver 580.95.05.

Two properties of that machine shape everything below.

**Memory bandwidth is the binding constraint, and it is low.** Measured
effective bandwidth is **~240 GB/s** — roughly a quarter of a 4090's and a
fourteenth of an H100's. Decode reads every weight once per token, so this sets
a hard ceiling of `240 / model_GB` tokens per second, and the whole curve lives
under it.

**Memory capacity is not a constraint at all.** 119 GB unified means the BF16
anchor (15.3 GiB) and a 65536-token KV cache coexist with room to spare. None of
the usual "will it fit" reasoning applies, so `parallel` is held at 8 for
methodological reasons rather than memory ones.

### Already measured, before the sweep

`llama-bench`, single stream, on the two rungs that finished downloading first:

| Rung | Size | Prefill (pp512) | Decode (tg128) | Implied bandwidth |
|---|---|---|---|---|
| BF16 | 15.26 GiB | 3177 tok/s | **14.92 tok/s** | 228 GiB/s |
| Q8_0 | 8.11 GiB | 2840 tok/s | **27.27 tok/s** | 221 GiB/s |

Two things are already visible in those four numbers.

Decode scales at 1.83× for a 1.88× size reduction — **within 3% of exactly
proportional to bytes**. The memory-bandwidth hypothesis is not a prediction
about this machine, it is a description of it, and the implied bandwidth agrees
to within 3% across both rungs.

Prefill goes the *other way*: BF16 is **12% faster than Q8_0** at prompt
processing, because prefill is compute-bound and the quantized rung has to
dequantize before it can multiply. The TTFT-versus-throughput divergence that
`bench`'s reporting rule exists to expose is already present at the very top of
the ladder.

## The ladder

`bpw` is measured — `file_bytes * 8 / 8,190,427,136` — never the published
table. Those tables describe a quant type in the abstract, and Qwen3-8B carries
a 151936 × 4096 matrix at each end (15% of its parameters) that every publisher
keeps at higher precision than the name implies. **Q2_K is 3.21 bpw in this
model, not the 2.96 the table says; IQ2_M is 2.98, not 2.70.** The error is
largest exactly where the curve gets interesting.

**Ladder** — 23 rungs, `bartowski/Qwen_Qwen3-8B-GGUF`, one publisher:

| Rung | bpw | | Rung | bpw | | Rung | bpw |
|---|---|---|---|---|---|---|---|
| `q8_0` | 8.51 | | `q4_k_s` | 4.69 | | `q2_k_l` | 3.80 |
| `q6_k_l` | 6.86 | | `iq4_nl` *(I)* | 4.68 | | `q3_k_s` | 3.68 |
| `q6_k` | 6.57 | | `q4_0` | 4.68 | | `iq3_xs` *(I)* | 3.54 |
| `q5_k_l` | 6.09 | | `iq4_xs` *(I)* | 4.46 | | `iq3_xxs` *(I)* | 3.29 |
| `q5_k_m` | 5.72 | | `q3_k_l` | 4.33 | | `q2_k` | 3.21 |
| `q5_k_s` | 5.59 | | `q3_k_m` | 4.03 | | `iq2_m` *(I)* | 2.98 |
| `q4_k_l` | 5.36 | | `iq3_m` *(I)* | 3.81 | | | |
| `q4_1` | 5.13 | | | | | | |
| `q4_k_m` | 4.91 | | | | | | |
| `q3_k_xl` | 4.86 | | | | | | |

Plus `bf16` (16.01) as the anchor, five `unsloth_dynamic` UD rungs paired
against the curve at matched width, and three `sub3_extension` rungs
(2.54 / 2.34 / 2.22) — the only checkpoints that exist below 2.98, all of them
Unsloth dynamic, because at 8B nobody publishes uniform quants that narrow.

Quantization quality depends on the importance-matrix calibration set, so a
`Q4_K_M` from one repo and an `IQ3_M` from another are not two points on one
curve — they are two curves sampled once each. A test enforces single-publisher
for the ladder group.

## What to look for

**The curve itself will be near-linear in bytes, and that is nearly
tautological.** At concurrency 1 on a 240 GB/s machine, tokens/sec is bandwidth
divided by model size; the two rungs already measured agree with that to 3%. Say
so in the write-up rather than letting a reader say it first. The findings are
the departures.

**The matched-size trio at ~4.68 bpw is the cleanest experiment in the set.**
`q4_0` (legacy round-to-nearest, no imatrix), `q4_k_s` (k-quant) and `iq4_nl`
(codebook I-quant) land within 0.01 bpw of each other. Bytes read per token are
identical, so *any* speed difference between them is kernel, not information
content. A test fails if a re-upload moves them apart. `iq3_m` (3.81) and
`q2_k_l` (3.80) are a second such pair.

**The I-quant inversion — and an honest caveat about this machine.** I-quants
are codebook lookups: more compute per byte than a k-quant of the same size. On
a bandwidth-rich GPU they routinely lose to a physically *larger* k-quant, which
inverts the curve. GB10 is the opposite regime — bandwidth-starved, compute
comparatively abundant — so the inversion may be **weaker or absent here**. That
is a real result about hardware regime rather than a failed prediction, and it
is the reason the matched pairs are in the set regardless of which way they fall.

**Prefill should stay flat or invert.** Already true between BF16 and Q8_0.
`c3_rag` (4000 in / 128 out) is where this is measured properly: TTFT roughly
level across the whole ladder while throughput climbs 7×. Reporting a single
"speedup" across both would conceal exactly this.

**The Unsloth dynamic group answers the original question.** Each UD rung sits
within 0.25 bpw of a uniform rung; the comparison is whether spending the same
bytes non-uniformly changes *speed*. Expected to be "barely" — the kernel is
chosen per tensor, so the mixture mostly shifts which kernel runs where. Worth
showing rather than assuming. A test enforces that every UD rung has a partner.

## What is reused, and what is new

Everything that turns requests into numbers came from the `bench` study and is
vendored verbatim into `src/ladder/harness/`, so the two report the same
quantity under the same name:

| Vendored module | What it does here |
|---|---|
| `client.py` | The load generator. Already engine-agnostic; speaks OpenAI streaming, which `llama-server` provides. |
| `scenarios.py` | `Scenario` / `Turn`, and the frozen JSONL format. |
| `classes.py` | The four prompt classes, read from `config/classes.yaml`. |
| `build_prompts.py`, `corpus.py` | *Not* vendored. They need `transformers` and a tokenizer checkout; the files they produced are read directly from `prompts/`. |
| `metrics.py` | Percentiles, aggregation, counter differencing, peak gauge sampling. |
| `tracking.py` | MLflow parent/child run structure. |

New here, and only this: `server.py` (llama.cpp's context arithmetic and CLI),
`dialect.py` (a metric-name table and the prefill-reuse check), `models.py`
(GGUF inventory and measured bits per weight), `run.py` (the driver, with
engine-specific probing and guards).

One change was made upstream to enable this, and it is backwards compatible:
`metrics.py` gained a `MetricsDialect` describing what one engine calls things,
with `VLLM` as the default everywhere. All 45 existing bench tests pass
unchanged. That is what lets `GaugeSampler` serve both engines instead of being
reimplemented, so the "absent rather than zero" convention has one
implementation rather than two.

The vendored copies are byte-identical to their originals apart from the import
prefix, so checking them against an upstream checkout is a `diff` rather than a
merge. `src/ladder/harness/__init__.py` gives the exact command and names the
one deliberate exception (`classes.py`, whose hop to the repo root is one
directory longer here).

## Two things llama.cpp does differently that change the design

**Context is divided, not per-sequence.** `--ctx-size` is the size of the
*entire* KV cache, split evenly across `--parallel` slots. vLLM pins
`max_model_len` as a per-sequence budget; here the per-sequence budget is
derived, so `n_ctx_per_slot` is the pinned field and `--ctx-size` follows from
it. A serve script started with the wrong one does not error — `c3_rag`'s
4000-token prompts just stop fitting.

**The concurrency axis is shorter.** The vLLM study sweeps {1, 32}; here
the default is {1, 8}. On this box that is *not* a memory limit — 119 GB unified
would hold far more — it is that the bandwidth-bound single-stream regime is
what this study is about, and `ladder.run` refuses a cell asking for more
concurrency than there are slots rather than reporting queued latency as
concurrent latency.

## What llama.cpp does not report, and why that is fine

Of the four validity signals the vLLM study records: **preemptions** do not
exist (llama.cpp defers rather than evicting, so queue depth is the pressure
signal instead); **speculative acceptance** does not apply (no draft model
here); **KV cache peak and queue depth** map directly and are reported under
the same metric names, so one query reads both studies. `preemption_delta`
returns nothing rather than `0.0`, because `0.0` would read as "checked, none
happened" — a claim this engine cannot make. The "absent rather than zero"
convention already covered all of it; it needed a name table, not new logic.

## One validity check the vLLM study does not have

`llamacpp:prompt_tokens_total` counts tokens the server actually *processed*,
and a slot skips whatever prefix it still holds. The client reports what each
prompt *contained*. Differencing them measures prefill reuse directly.

This matters because `--cache-reuse 0` does not govern slot-level reuse of a
repeated prompt. If it happens, TTFT is faster than the rung can deliver cold,
unevenly across rungs, and whichever rung reused more looks quantization-faster.
Every cell logs `prefill_processed_ratio` and warns below 0.98.

## Layout

```
config/ladder.yaml      the 32 rungs
config/classes.yaml     the four prompt classes
prompts/                frozen prompt sets + manifest.json (digests checked by the suite)
scripts/                generated serve + download scripts, committed
src/ladder/
  server.py             config loader, serve scripts, /props probe, context guards
  dialect.py            llama.cpp metric names + the prefill-reuse check
  models.py             GGUF inventory, measured bits per weight
  run.py                sweep driver and preflight
  harness/              vendored from the bench study; see its __init__ docstring
tests/mock_llama_server.py   fake llama-server, so the harness is testable without a GPU
```

## Development

No sibling checkout, no path dependencies:

```bash
git clone <ladder-url> ladder
cd ladder && uv sync --extra mock --extra dev && uv run pytest -q   # 46 tests
```

The tradeoff that buys is worth stating plainly: `src/ladder/harness/` is a
copy, so a fix made upstream in `bench` does not arrive here on its own. The
files are kept byte-identical apart from the import prefix precisely so that
picking such a fix up stays a `diff` and a `cp`.

The prompt sets are copied too, and that is the riskier half -- two copies of a
frozen dataset can drift without anything failing. `manifest.json` travels with
them and the suite hashes every file against it, so drift is a test failure
rather than a silently unusable anchor.

See [RUNBOOK.md](RUNBOOK.md) for the GPU-box procedure.

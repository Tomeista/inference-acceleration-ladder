# RUNBOOK — the GGUF ladder

Steps to run on `dell-gb10-1`, in order. Each block is copy-pasteable and says
what to paste back. It mirrors the RUNBOOK of the vLLM `bench` study; where a
step differs, it says why.

This repository is self-contained -- no sibling checkout is needed. And unlike
that study, this one builds no checkpoints: every rung is a published file, so
no quantization step is involved at all.

**Machine.** NVIDIA GB10, compute capability 12.1, aarch64, 20 cores, 119 GB
unified LPDDR5X, CUDA 13.0, driver 580.95.05. No sudo, not in the `docker`
group, so everything is a userland build in `$HOME`.

---

## Step 0: workstation, no GPU

The harness is exercised against a mock llama-server first, which is how the
dialect, the guards and the MLflow structure get checked before deployment.

```bash
uv sync --extra mock --extra dev
uv run pytest -q          # 46 tests
```

Then a smoke sweep against the mock:

```bash
MOCK_TTFT_MS=15 MOCK_ITL_MS=1 MOCK_PROMPT_TOKENS=64 \
    uv run uvicorn --app-dir tests mock_llama_server:app --port 8111 &
uv run python -m ladder.run --config-id q4_k_m --server-url http://127.0.0.1:8111 \
    --classes c1_chat --requests-per-cell 24 --concurrency 1,4 --no-mlflow
```

Nothing in that output is a performance result — the mock's latencies are
configured, not modeled. What it proves is that the cell loop, the llama.cpp
metric names, the prefill check and the MLflow nesting all work.

---

# GPU server

## Step 1: build llama.cpp

Nothing is preinstalled. `apt` needs sudo, so llama.cpp is built from source
into `$HOME`. GB10 is compute capability 12.1, so the CUDA arch is **121** —
getting this wrong produces a binary that runs on CPU and silently measures
nothing but memcpy.

```bash
export PATH=/usr/local/cuda/bin:$PATH
git clone --depth 1 https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 \
      -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 20
mkdir -p ~/.local/bin
ln -sf ~/llama.cpp/build/bin/llama-server ~/.local/bin/llama-server
ln -sf ~/llama.cpp/build/bin/llama-bench  ~/.local/bin/llama-bench
```

`LLAMA_CURL=OFF` because libcurl-dev would need sudo; models come from the HF
CLI instead. The web UI download fails during cmake and that is harmless — every
serve script passes `--no-webui`.

**Confirm the CUDA backend actually built**, because a CPU-only fallback is the
failure that would quietly ruin the whole study:

```bash
grep -E "GGML_CUDA:|CMAKE_CUDA_ARCHITECTURES:" build/CMakeCache.txt
llama-bench --list-devices
```

**Paste back both.** The second must name `CUDA0: NVIDIA GB10`. Expect
~122506 MiB reported as VRAM — that is the unified pool, not a discrete card.

## Step 2: fetch the ladder

```bash
cd ~/ladder          # wherever this repository was cloned
uv sync --extra mock --extra dev
uv run python -m ladder.models --emit-download
uv run bash scripts/download_models.sh
uv run python -m ladder.models --verify
```

**~147 GiB**, dominated by the BF16 anchor at 15.3 GiB. Takes roughly half an
hour at 80 MB/s. Disk has 1.6 TB free, so this is not tight.

**Paste back the `--verify` table.** It writes `models.lock.json`, which is where
every run gets its measured bits-per-weight — the actual x-axis. Anything marked
`<- check` disagrees with the size HuggingFace advertised by more than 0.15 bpw,
which means a truncated or renamed file, not a surprising checkpoint.

> Repo naming has bitten this once already: bartowski publishes under
> `bartowski/Qwen_Qwen3-8B-GGUF`, not `bartowski/Qwen3-8B-GGUF`, and the files
> are `Qwen_Qwen3-8B-<QUANT>.gguf`. If a download 404s, fix `config/ladder.yaml`
> and re-emit — never rename the file on disk, or the config stops describing
> what was measured.

## Step 3: preflight, once per build

**Not optional.** Two assumptions hold up the entire sweep, both of which
llama.cpp can violate *silently* — every request still succeeds — and both of
which invalidate every number if wrong.

```bash
uv run python -m ladder.run --emit-scripts
bash scripts/serve_q4_k_m.sh &
uv run python -m ladder.run --config-id q4_k_m --preflight
```

**Paste back the output.** It checks:

1. **`ignore_eos` reaches the sampler through `/v1/chat/completions`.** It is a
   llama.cpp-native parameter riding in an OpenAI request body. If this build
   drops it, output length stops being pinned and each rung is measured over a
   different number of decode steps — so every tokens/sec figure in the study is
   computed over a different amount of work.
2. **The chat template agrees with the one the prompts were built against.** The
   prompt sets were tokenized against Qwen3's HF template; the GGUF carries its
   own copy and the two have diverged before. If they differ, `c3_rag` is no
   longer a 4000-token prefill class.
3. **`/metrics` answers with the names in `ladder/dialect.py`.** These have moved
   between llama.cpp releases exactly as vLLM's have. The preflight prints what
   it found; if the list is empty but the server has `--metrics`, update the
   dialect.

Do not start the sweep until this passes. Finding out afterwards costs the run.

## Step 4: validate the client, once

The client is ours, so its numbers need checking against a second implementation.

```bash
llama-bench -m models/gguf/Qwen_Qwen3-8B-Q4_K_M.gguf -p 512 -n 128 -r 2
uv run python -m ladder.run --config-id q4_k_m \
    --classes c1_chat --concurrency 1 --requests-per-cell 32 --no-mlflow
```

`1000 / tpot_p50` should land within a few percent of llama-bench's `tg128`. If
it does not, the client is the suspect, not llama.cpp. Note that llama-bench
runs a bare model with no server, template or HTTP, so only the decode rate is
comparable — its numbers say nothing about TTFT.

**Reference, already measured on this box:**

| Rung | Size | pp512 | tg128 |
|---|---|---|---|
| BF16 | 15.26 GiB | 3177 tok/s | 14.92 tok/s |
| Q8_0 | 8.11 GiB | 2840 tok/s | 27.27 tok/s |

Effective bandwidth is ~240 GB/s, so a rung's single-stream decode rate should
come out near `240 / size_GB`. A rung more than ~15% off that line is worth
investigating before it becomes a finding.

## Step 5: the anchor

```bash
bash scripts/serve_bf16.sh &
uv run python -m ladder.run --config-id bf16
```

Roughly **70 minutes**, dominated by `c2_longform` at 14.9 tok/s.

> **The vLLM half of this anchor is in doubt.** The `bench` study assumes vLLM,
> and this box is aarch64 with no sudo and no docker group. vLLM ships no
> prebuilt aarch64 + CUDA 13 wheels, and NGC containers need the docker group.
> Building from source is a multi-hour, failure-prone job. Until that is
> resolved the ladder stands alone: it is internally valid and complete on its
> own, but no numeric statement can be made relating it to the vLLM study. Do
> not quietly compare a llama.cpp number to a vLLM number from an earlier
> machine — the engine and the hardware would both differ.

## Step 6: sweep the ladder

One server at a time, restarted between rungs.

```bash
for r in q8_0 q6_k_l q6_k q5_k_l q5_k_m q5_k_s q4_k_l q4_1 q4_k_m q3_k_xl \
         q4_k_s iq4_nl q4_0 iq4_xs q3_k_l q3_k_m iq3_m q2_k_l q3_k_s \
         iq3_xs iq3_xxs q2_k iq2_m ud_q4_k_xl ud_q3_k_xl ud_q2_k_xl \
         ud_iq3_xxs ud_iq2_m ud_iq2_xxs ud_iq1_m ud_iq1_s; do
    bash scripts/serve_${r}.sh & SRV=$!
    uv run python -m ladder.run --config-id ${r}
    kill $SRV; wait $SRV 2>/dev/null || true
done
```

**Budget ~11.5 hours** for all 32 rungs. Per-rung time is dominated by
`c2_longform` (32 requests × 1024 output tokens at concurrency 1) and therefore
scales inversely with bpw: ~70 min at BF16, ~25 min at Q4_K_M, ~15 min at the
bottom. Run it overnight.

**A cheaper first pass** — `--classes c1_chat,c2_longform --concurrency 1` — is
about a third of that and enough to see the curve's shape before committing.

### Warnings that mean stop

- **`prefill=0.xx`** below ~0.98 on the summary line. A slot reused a cached
  prefix, so TTFT in that cell is faster than the rung can deliver cold — and
  rungs that reuse more look quantization-faster.

  The cause is *not* the prompt set wrapping. It is that different prompts in
  the same class share a prefix — the tool schema in `c5_toolcall`, the chat
  template in `c1_chat`, a shared source document in `c3_rag` — and llama.cpp
  assigns an incoming request to whichever slot already holds the longest
  matching prefix. It is therefore worse at `c=8` than at `c=1`, because eight
  occupied slots offer eight chances to match. Lowering `--requests-per-cell`
  does nothing.

  The fix is **`--no-cache-prompt`** in `extra_args`, then `--emit-scripts`.
  Two neighbouring flags look like they should do it and do not, both measured
  on this box:

  | Flag | Effect on `c1_chat` c=8 |
  |---|---|
  | `--cache-reuse 0` (already set) | 0.02 — governs KV shifting inside a slot |
  | `--slot-prompt-similarity 0` | 0.02 — only changes *which* slot is chosen |
  | `--no-cache-prompt` | **1.00** |

  Prefix reuse is a property of the slot's KV cache, not of slot assignment, so
  only disabling prompt caching outright removes it.

  `c2_longform` sits at 0.97 legitimately — a few tokens of shared template and
  nothing more — so 0.97 is the clean reading for that class, not a failure.
  This is why `ladder.report` cuts at 0.95 rather than 0.98.

  **Turning this off is not free, and the cost is the point.** Uncached, every
  request prefills in full, which competes with decode for the same GPU. At
  c=1 the difference is under 6% on every cell; at c=8 it is large — `c3_rag`
  throughput roughly halves and its p95 TTFT goes to ~10 s. Those are the
  honest numbers for a cold cache; the cached ones were measuring a workload
  that skipped most of its own prefill.
- **`did not stop on length`.** `ignore_eos` stopped being honoured mid-sweep,
  usually a different server binary. Everything since the last clean cell is
  suspect.
- **peak queue depth > 0.** Requests waited for a slot, so the cell measured
  queueing as well as decoding. Should be impossible given the concurrency
  guard; if it happens, the running server has fewer slots than its script asked
  for.

## Step 7: read it

```bash
uv run python -m ladder.report                 # the merged curve
uv run python -m ladder.report --all           # including cells that failed a check
uv run python -m ladder.report --csv curve.csv # the table, for plotting
```

**Use this rather than reading cells out of the UI by hand**, for one specific
reason: MLflow appends, so a class that was measured twice has two cells, and
nothing in the UI stops a group-by from averaging a discarded measurement with
the one that replaced it. `ladder.report` keeps the most recent cell per
`(config_id, class_id, concurrency)`, says how many it superseded, and drops
cells that failed a validity check unless asked for them.

The UI is still the right tool for looking at one run:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

That is this repository's own store, written by default. To read this study and
the vLLM one in a single table -- the only way the BF16 anchor can be compared
across engines -- run both against one explicit path instead:

```bash
uv run python -m ladder.run --config-id bf16 --tracking-uri sqlite:////abs/path/shared.db
```

The experiment names differ (`inference-acceleration-ladder` here,
`inference-acceleration` for the vLLM study), so they never pool into one query
by accident. Filter on `tags.engine`.

**The curve.** `params.bpw_measured` against `metrics.output_tps`, one class at a
time, `params.ladder_group = "ladder"` only. Expect near-linear at concurrency 1;
overlay `240 / size_GB` and expect the points to sit on it.

**The matched-size trio — the cleanest experiment in the set.** `q4_0`,
`q4_k_s` and `iq4_nl` are all 4.68–4.69 bpw. Identical bytes per token, three
different formats: legacy RTN, k-quant, codebook I-quant. Any speed difference
is kernel, not information content. `iq3_m` (3.81) vs `q2_k_l` (3.80) is the
second such pair.

**The I-quant question.** Colour the curve by `params.quant_family` and ask
whether any `iquant` point sits below a `kquant` point of *larger* bpw. On a
bandwidth-rich GPU it usually does. **GB10 is the opposite regime** —
bandwidth-starved, compute abundant — so the inversion may be weak or absent
here. Either outcome is a result about hardware regime; report it as such rather
than as a property of the formats.

**TTFT against throughput.** `metrics.ttft_s_p50` on the same x-axis, for
`c3_rag`. Expect it flat or rising slightly as bpw falls, while throughput
climbs ~7×. This is already visible in step 4's reference numbers: BF16 prefills
12% *faster* than Q8_0. Never collapse the two into one "speedup".

**The dynamic group, separately.** `params.ladder_group = "unsloth_dynamic"`,
each rung against the ladder rung nearest it in *measured* bpw. Then
`sub3_extension` as the tail below 2.98 — three points, all Unsloth, not part of
the curve.

---

## What this study cannot tell you

**At concurrency 1 this is largely a memory-bandwidth measurement.** Decode
reads every weight once per token, so tokens/sec tracks file size nearly by
definition — the two reference rungs agree with that to 3%. The curve being
clean is not evidence that quantization is clever; it is evidence that the GPU
reads fewer bytes when there are fewer bytes. The informative parts are the
departures: the matched-size trio, the I-quant behaviour, the flat TTFT line.

**Nothing here measures quality.** A 2.22-bit Qwen3-8B may be fast and useless.
The prompt sets run with `ignore_eos` and fixed lengths precisely so output
*content* does not affect timing, which also means the outputs are not scoreable.
Quality is a separate pass with natural stopping, on the same frozen prompts.

**Nothing here is a serving recommendation.** llama.cpp with eight slots on a
bandwidth-limited unified-memory box is not how the models in the vLLM study
would be deployed. This study answers "how does speed move with bit width", and the
engine was chosen because it is the only one that can be asked at 32 points.

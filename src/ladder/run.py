"""Sweep driver: run every (prompt class x concurrency) cell against one rung.

One invocation measures one GGUF file, because llama-server is started by hand
on the GPU box. Sweeping the ladder means running this once per rung and letting
MLflow hold the comparison -- the same shape as `bench.run`, and deliberately
the same CLI, so the two studies are driven identically.

What this module does NOT own is worth listing, because it is most of the work:
the prompt sets, the load generator, the percentile math, the gauge sampling and
the MLflow structure all come from `bench` unchanged. In particular the prompt
sets are *bench's frozen files*, read from `bench.classes.PROMPTS_DIR`. Building
a second set here would have been the single most expensive mistake available:
the F16 anchor only ties the two studies together if both engines saw byte-
identical prompts.

Usage:
    # emit the serve scripts, then start one of them on the GPU box
    python -m ladder.run --emit-scripts

    # measure the running server
    python -m ladder.run --config-id q4_k_m --server-url http://127.0.0.1:8080
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from bench import metrics as metrics_mod
from bench import tracking
from bench.classes import PROMPTS_DIR, PromptClass, select_classes
from bench.client import run_load, warmup
from bench.scenarios import read_scenarios

from ladder.dialect import LLAMACPP, prefill_reuse_check, throughput_cross_check
from ladder.models import LOCK_PATH
from ladder.server import (
    PACKAGE_ROOT,
    LadderConfig,
    check_context_budget,
    load_configs,
    server_info,
    wait_for_ready,
    write_serve_scripts,
)

ARTIFACT_ROOT = PACKAGE_ROOT / "results"

# Separate experiment, same store. The two studies must not be pooled into one
# table -- an engine difference would masquerade as a quantization difference --
# but they do have to be queryable together for the F16 anchor to be readable.
DEFAULT_EXPERIMENT = "inference-acceleration-ladder"

# Default tracking store is bench's own sqlite database, so `mlflow ui` from
# either directory shows both experiments and the F16 anchor can be read across
# them in one query. Not bench/mlruns -- that directory is the artifact tree,
# and current MLflow refuses the filesystem backend outright.
DEFAULT_TRACKING_URI = "sqlite:///" + (PACKAGE_ROOT.parent / "bench" / "mlflow.db").as_posix()


def auto_requests(concurrency: int) -> int:
    """Requests per cell when not given explicitly. Matches bench exactly."""
    return max(32, 4 * concurrency)


def format_cell(cls: PromptClass, concurrency: int, values: dict[str, float]) -> str:
    def g(key: str, scale: float = 1.0, digits: int = 1) -> str:
        v = values.get(key)
        return f"{v * scale:.{digits}f}" if v is not None else "-"

    line = (
        f"  {cls.id:16s} c={concurrency:<4d} "
        f"ttft_p50={g('ttft_s_p50', 1000):>8s}ms "
        f"ttft_p95={g('ttft_s_p95', 1000):>8s}ms "
        f"tpot_p50={g('tpot_s_p50', 1000):>7s}ms "
        f"out_tps={g('output_tps', 1, 1):>8s} "
        f"ok={int(values.get('requests_ok', 0))}/{int(values.get('requests_total', 0))}"
    )
    # The ladder's own validity signal, printed next to the number it qualifies
    # rather than only in MLflow: a TTFT achieved on reused prefill is not a
    # TTFT this rung can deliver.
    ratio = values.get("prefill_processed_ratio")
    if ratio is not None:
        line += f" prefill={ratio:.2f}"
    return line


async def run_cell(
    cls: PromptClass,
    concurrency: int,
    cfg: LadderConfig,
    base_url: str,
    args: argparse.Namespace,
) -> tuple[dict, dict, list[dict]]:
    scenarios = read_scenarios(cls.prompt_file)
    n_requests = args.requests_per_cell or auto_requests(concurrency)

    if n_requests > len(scenarios):
        print(
            f"  note: {cls.id} has {len(scenarios)} prompts for {n_requests} requests; "
            f"the set will wrap. Repeating a prompt is worse here than in bench: a "
            f"slot that still holds it skips prefill entirely.",
            file=sys.stderr,
        )

    before = metrics_mod.scrape(base_url, dialect=LLAMACPP)

    async with metrics_mod.GaugeSampler(base_url, dialect=LLAMACPP) as sampler:
        result = await run_load(
            scenarios,
            base_url=base_url,
            model=cfg.served_model_name,
            concurrency=concurrency,
            num_requests=n_requests,
            api_key=args.api_key,
            request_timeout=args.request_timeout,
            collect_output=args.collect_output,
        )

    after = metrics_mod.scrape(base_url, dialect=LLAMACPP)

    run_metrics = metrics_mod.aggregate(result)
    values = run_metrics.merged()
    # preemption_delta and spec_decode_metrics both return {} under this
    # dialect, by design: llama.cpp defers rather than preempting, and there is
    # no draft model in this study. Calling them anyway keeps the two drivers
    # line-for-line comparable and means a future engine that does publish them
    # needs no change here.
    values.update(metrics_mod.preemption_delta(before, after, dialect=LLAMACPP))
    values.update(throughput_cross_check(before, after))
    values.update(sampler.metrics())

    client_prompt_tokens = sum(r.prompt_tokens or 0 for r in result.successful)
    prefill = prefill_reuse_check(before, after, client_prompt_tokens)
    values.update({k: v for k, v in prefill.items() if not isinstance(v, bool)})

    notes = dict(run_metrics.notes)
    notes.update(metrics_mod.check_prompt_lengths(result.successful, cls.input_tokens))
    if "prefill_reuse_free" in prefill:
        notes["prefill_reuse_free"] = prefill["prefill_reuse_free"]

    if not notes.get("length_capped", True):
        print(
            f"  warning: {cls.id} did not stop on length in every request "
            f"({notes.get('finish_reasons')}). llama-server has to accept "
            f"ignore_eos through the OpenAI endpoint for the output-length pin "
            f"to hold; if it does not, every rung is being measured on a "
            f"different number of decode steps. Run --preflight.",
            file=sys.stderr,
        )

    if notes.get("prefill_reuse_free") is False:
        print(
            f"  warning: server prefilled only "
            f"{values.get('prefill_processed_ratio', float('nan')):.2f} of the prompt "
            f"tokens sent during {cls.id} at c={concurrency}. A slot reused a "
            f"cached prefix, so TTFT here is faster than this rung can deliver "
            f"cold. Rungs that reuse more will look quantization-faster.",
            file=sys.stderr,
        )

    if values.get("requests_waiting_peak", 0) > 0:
        print(
            f"  warning: peak queue depth {values['requests_waiting_peak']:.0f} during "
            f"{cls.id} at c={concurrency}. Requests waited for a slot, so this "
            f"cell measures queueing as well as decoding.",
            file=sys.stderr,
        )

    records = [r.to_dict() for r in result.records]
    return values, notes, records


def _lock_entry(config_id: str) -> dict:
    """Measured bpw for this rung, from models.lock.json.

    The x-axis of the study. Recorded as a run parameter so a plot never has to
    join back to a file that may have been regenerated since.
    """
    if not LOCK_PATH.exists():
        return {}
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    for row in lock.get("rungs", []):
        if row.get("config_id") == config_id:
            return {
                "bpw_measured": row.get("bpw_measured"),
                "gguf_size_bytes": row.get("size_bytes"),
            }
    return {}


async def preflight(cfg: LadderConfig, base_url: str) -> int:
    """Prove the two assumptions the whole sweep rests on, before spending a day.

    Both are things llama-server does silently rather than by erroring, and both
    invalidate every number in the sweep:

      1. `ignore_eos` reaches the sampler through /v1/chat/completions. It is a
         llama.cpp-native parameter riding in the OpenAI request body, and if a
         build ignores it, output length stops being pinned and each rung is
         measured over a different number of decode steps.
      2. The chat template renders to the length the prompt set was built for.
         The prompts were tokenized against Qwen3's HF template; the GGUF file
         carries its own copy, and the two have diverged before.
    """
    cls = next(c for c in select_classes(["c1_chat"]))
    scenarios = read_scenarios(cls.prompt_file)[:4]
    result = await run_load(
        scenarios,
        base_url=base_url,
        model=cfg.served_model_name,
        concurrency=1,
        num_requests=4,
    )

    ok = result.successful
    if not ok:
        first = next((r.error for r in result.records if r.error), "no detail")
        print(f"preflight FAILED: no request succeeded ({first})", file=sys.stderr)
        return 1

    problems: list[str] = []

    reasons = {r.finish_reason for r in ok}
    lengths = {r.completion_tokens for r in ok}
    print(f"  finish reasons      : {sorted(reasons)}")
    print(f"  completion tokens   : {sorted(lengths)} (expected {{{cls.output_tokens}}})")
    if reasons != {"length"} or lengths != {cls.output_tokens}:
        problems.append(
            "ignore_eos is not in force: generation stopped somewhere other than "
            "the token cap. Output length is not pinned, so rungs are not "
            "comparable. Check that this llama.cpp build reads ignore_eos from "
            "the /v1/chat/completions body."
        )

    observed = {r.prompt_tokens for r in ok if r.prompt_tokens}
    print(f"  prompt tokens       : {sorted(observed)} (expected {{{cls.input_tokens}}})")
    if observed and abs(max(observed) - cls.input_tokens) > 2:
        problems.append(
            f"prompt length is {sorted(observed)}, not {cls.input_tokens}. The "
            f"GGUF chat template differs from the HF template the prompt set was "
            f"built against, so prefill cost is not what the class specifies."
        )

    scraped = metrics_mod.scrape(base_url, dialect=LLAMACPP)
    print(f"  metrics found       : {sorted(scraped) or 'NONE'}")
    if not scraped:
        problems.append(
            "no llama.cpp counters at /metrics. Start the server with --metrics, "
            "or the sweep loses every validity check. If it is running with "
            "--metrics, the metric names have moved in this build and "
            "ladder/dialect.py needs updating."
        )

    if problems:
        print("\npreflight FAILED:", file=sys.stderr)
        for i, problem in enumerate(problems, 1):
            print(f"  {i}. {problem}", file=sys.stderr)
        return 1

    print("\npreflight OK")
    return 0


async def main_async(args: argparse.Namespace) -> int:
    configs = load_configs()

    if args.emit_scripts:
        for path in write_serve_scripts(configs):
            print(f"wrote {path}")
        return 0

    if not args.config_id:
        print("--config-id is required (or use --emit-scripts)", file=sys.stderr)
        return 2
    if args.config_id not in configs:
        print(f"unknown config {args.config_id!r}; known: {sorted(configs)}", file=sys.stderr)
        return 2

    cfg = configs[args.config_id]
    base_url = args.server_url or cfg.base_url
    classes = select_classes([c.strip() for c in args.classes.split(",") if c.strip()])
    concurrencies = [int(c) for c in args.concurrency.split(",") if c.strip()]

    # Refuse the cell rather than reporting it. Both of these produce plausible
    # numbers that mean something other than what the run claims.
    for cls in classes:
        needed = cls.input_tokens + cls.output_tokens
        for concurrency in concurrencies:
            if problem := check_context_budget(cfg, concurrency, needed):
                print(f"cannot run {cls.id} at c={concurrency}: {problem}", file=sys.stderr)
                return 2

    print(f"rung        : {cfg.id} ({cfg.name})")
    print(f"quant       : {cfg.quant_type}  ~{cfg.bpw_nominal} bpw nominal  [{cfg.group}]")
    print(f"file        : {cfg.model_file}")
    print(f"server      : {base_url}")
    print(f"context     : {cfg.n_ctx_per_slot}/slot x {cfg.parallel} slots = {cfg.n_ctx}")
    print(f"classes     : {', '.join(c.id for c in classes)}")
    print(f"concurrency : {concurrencies}")
    print(f"cells       : {len(classes) * len(concurrencies)}")

    if args.dry_run:
        for cls in classes:
            for concurrency in concurrencies:
                n = args.requests_per_cell or auto_requests(concurrency)
                print(f"  would run {cls.id} at c={concurrency} for {n} requests")
        return 0

    wait_for_ready(base_url, timeout=args.ready_timeout)
    info = server_info(base_url)
    print(f"llama.cpp   : {info.get('build_info') or 'build unreported'}")
    print(f"served      : {info.get('served_models')}  slots={info.get('total_slots')}")

    reported_ctx = info.get("n_ctx_reported")
    # /props reports the PER-SLOT context, not the total: with --ctx-size 65536
    # and --parallel 8 it says 8192. Older builds reported the total instead, so
    # either value is accepted and only a third value means the running server
    # was not started from this rung's serve script -- the most likely way this
    # sweep goes wrong, and worth a loud line rather than a silent parameter.
    if reported_ctx and int(reported_ctx) not in (cfg.n_ctx_per_slot, cfg.n_ctx):
        print(
            f"warning: server reports n_ctx {reported_ctx}, which is neither this "
            f"rung's per-slot budget ({cfg.n_ctx_per_slot}) nor its total "
            f"({cfg.n_ctx}). The running server was not started from "
            f"scripts/serve_{cfg.id}.sh.",
            file=sys.stderr,
        )

    if cfg.served_model_name not in (info.get("served_models") or []):
        print(
            f"warning: served model {info.get('served_models')} does not include "
            f"{cfg.served_model_name!r} from config {cfg.id!r}. The running server "
            f"may not be the one this config describes.",
            file=sys.stderr,
        )

    if args.preflight:
        print("\npreflight ...")
        return await preflight(cfg, base_url)

    if args.warmup:
        print(f"warmup      : {args.warmup} requests on {classes[0].id}")
        await warmup(
            read_scenarios(classes[0].prompt_file),
            base_url=base_url,
            model=cfg.served_model_name,
            num_requests=args.warmup,
        )

    stamp = time.strftime("%Y%m%d-%H%M%S")
    artifact_dir = ARTIFACT_ROOT / f"{cfg.id}-{stamp}"

    if args.mlflow:
        tracking.setup(args.experiment, tracking_uri=args.tracking_uri)

    parent_params = {
        **cfg.as_params(),
        **_lock_entry(cfg.id),
        "llamacpp_build": info.get("build_info"),
        "n_ctx_reported": reported_ctx,
        "server_url": base_url,
        "prompt_manifest": _manifest_digest(),
    }

    summary: list[str] = []

    def emit(line: str) -> None:
        print(line)
        summary.append(line)

    ctx = (
        tracking.config_run(
            run_name=cfg.id, params=parent_params, tags={"engine": "llama.cpp"}
        )
        if args.mlflow
        else _null_context()
    )

    with ctx:
        for cls in classes:
            for concurrency in concurrencies:
                cell_id = f"{cls.id}__c{concurrency}"
                print(f"\nrunning {cell_id} ...")
                values, notes, records = await run_cell(cls, concurrency, cfg, base_url, args)
                emit(format_cell(cls, concurrency, values))

                if args.mlflow:
                    cell_params = {
                        **cfg.as_params(),
                        **_lock_entry(cfg.id),
                        **cls.as_params(),
                        "concurrency": concurrency,
                        "num_requests": len(records),
                        "ignore_eos": True,
                    }
                    with tracking.cell_run(run_name=cell_id, params=cell_params):
                        tracking.log_cell_results(values, notes, records, artifact_dir, cell_id)

    print("\nsummary")
    for line in summary:
        print(line)
    if args.mlflow:
        print(f"\nartifacts: {artifact_dir}")
    return 0


def _manifest_digest() -> str:
    """Tie a run to the exact prompt sets it used -- bench's, not a copy."""
    path = PROMPTS_DIR / "manifest.json"
    if not path.exists():
        return "missing"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return ",".join(
        f"{k}:{v.get('sha256_16', '?')}"
        for k, v in sorted(manifest.get("classes", {}).items())
    )


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config-id", help="Which rung of config/ladder.yaml is running")
    parser.add_argument("--server-url", default=None, help="Overrides the config's host/port")
    parser.add_argument("--classes", default="", help="Comma separated ids; default all enabled")
    # Defaults to the slot count in ladder.yaml, not bench's 1,32: llama.cpp
    # would queue anything past --parallel and report the wait as latency.
    parser.add_argument("--concurrency", default="1,8")
    parser.add_argument(
        "--requests-per-cell", type=int, default=0, help="0 selects max(32, 4 * concurrency)"
    )
    parser.add_argument("--warmup", type=int, default=8, help="Discarded requests before measuring")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--ready-timeout", type=float, default=900.0)
    parser.add_argument(
        "--collect-output",
        action="store_true",
        help="Keep generated text in memory; needed later for the quality pass",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check ignore_eos, the chat template and /metrics, then exit",
    )
    parser.add_argument("--no-mlflow", dest="mlflow", action="store_false", default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--emit-scripts", action="store_true", help="Write scripts/serve_*.sh and exit"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()

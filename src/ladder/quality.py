"""Quality pass: what a rung gets *wrong* as bits per weight fall.

The other half of `ladder.run`. That module measures how fast a rung answers and
goes to some trouble to make sure the content of the answer cannot affect the
timing -- `ignore_eos` on, output length pinned, every rung decoding the same
number of steps. The consequence, stated in the RUNBOOK, is that its outputs are
not scoreable. This module scores instead, and inverts nearly every one of those
choices:

    ladder.run                          ladder.quality
    ignore_eos on, length pinned        natural stopping, truncation counted
    prompt shapes, no right answer      public benchmark items with a key
    all 32 rungs                        the 9 rungs flagged `quality: true`
    output discarded                    output scored and kept per item

What it does NOT invert is the server. The same `scripts/serve_<rung>.sh` starts
the same llama-server with the same context, slots and KV dtype; prompt caching
cannot change *what* a rung answers, only how fast, so there is no second script
set and no second context regime to keep in sync.

Run the reference rung first. Everything else is compared against it per item,
and `agreement_with_reference` -- not accuracy -- is the metric with the
resolution to see where the damage starts.

Usage:
    python -m ladder.quality --config-id bf16          # the reference, first
    python -m ladder.quality --config-id q4_k_m
    python -m ladder.quality --config-id bf16 --concurrency 1 --suffix c1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from ladder.harness import tracking
from ladder.harness.client import run_load

from ladder.models import lock_entry
from ladder.scoring import ItemResult, aggregate, agreement, score_records
from ladder.server import (
    PACKAGE_ROOT,
    LadderConfig,
    check_context_budget,
    load_configs,
    server_info,
    wait_for_ready,
)
from ladder.suites import MANIFEST_PATH, Suite, load_suite, select_suites
from ladder.run import DEFAULT_EXPERIMENT, DEFAULT_TRACKING_URI

ARTIFACT_ROOT = PACKAGE_ROOT / "results"

# Per-item answers, kept outside the MLflow store because they are read back by
# the *next* rung's run rather than by a human. This is the join that makes
# agreement-with-reference possible, so it needs a stable path and a lifetime
# longer than one run's artifacts.
QUALITY_ROOT = ARTIFACT_ROOT / "quality"

# The rung every other rung is scored against. Not "the best" rung -- the
# full-precision one, which is the only sense in which any of these has a
# correct answer to diverge from.
DEFAULT_REFERENCE = "bf16"

# Warning thresholds. Neither is fatal: both describe how to read the cell
# rather than whether it happened.
TRUNCATION_WARN = 0.05
UNPARSEABLE_WARN = 0.20


def items_path(suite_id: str, config_id: str, suffix: str = "") -> Path:
    name = f"{config_id}__{suffix}" if suffix else config_id
    return QUALITY_ROOT / suite_id / f"{name}.jsonl"


def meta_path(items: Path) -> Path:
    """Sidecar describing what produced an items file.

    Exists because the reference join is the one place this pass can be
    confidently, silently wrong. `results/` is machine state -- gitignored,
    regenerated, easy to leave lying around -- so a `bf16.jsonl` from a smoke
    test against the mock, or from before an eval set was rebuilt, joins by
    scenario_id perfectly well and produces an agreement column about nothing.
    Item ids alone cannot detect that; what produced them can.
    """
    return items.with_suffix(".meta.json")


def write_items(path: Path, results: list[ItemResult], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for result in results:
            fh.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
    with meta_path(path).open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(meta, indent=2) + "\n")


def read_meta(path: Path) -> dict:
    sidecar = meta_path(path)
    if not sidecar.exists():
        return {}
    return json.loads(sidecar.read_text(encoding="utf-8"))


def read_reference(path: Path) -> dict[str, str | None]:
    """The reference rung's extracted answer per item, or {} if it has not run.

    {} rather than an error: a sweep that starts somewhere other than the
    reference still produces valid accuracy numbers, it just cannot produce the
    paired metric, and `scoring.agreement` then omits it rather than logging a
    zero that would read as total disagreement.
    """
    if not path.exists():
        return {}
    reference: dict[str, str | None] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            reference[row["scenario_id"]] = row["extracted"]
    return reference


def _manifest_digest() -> str:
    """Tie a run to the exact eval sets it scored, as run.py does for prompts."""
    if not MANIFEST_PATH.exists():
        return "missing"
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return ",".join(
        f"{k}:{v.get('sha256_16', '?')}"
        for k, v in sorted((manifest.get("suites") or {}).items())
    )


def format_suite(suite: Suite, values: dict[str, float]) -> str:
    def g(key: str, digits: int = 3) -> str:
        v = values.get(key)
        return f"{v:.{digits}f}" if v is not None else "-"

    line = (
        f"  {suite.id:10s} n={int(values.get('n_items', 0)):<4d} "
        f"acc={g('accuracy')} "
        f"[{g('accuracy_ci_lo')}, {g('accuracy_ci_hi')}] "
        f"unparse={g('unparseable_rate')} "
        f"trunc={g('truncated_rate')} "
        f"rep={g('repetition_ratio')}"
    )
    if "agreement_with_reference" in values:
        line += f" agree={g('agreement_with_reference')}"
    return line


async def run_suite(
    suite: Suite,
    cfg: LadderConfig,
    base_url: str,
    args: argparse.Namespace,
) -> tuple[dict, dict, list[ItemResult]]:
    """Score one suite against one rung."""
    scenarios, answers = load_suite(suite, verify=not args.no_verify)
    if args.n_items:
        # Smoke-test escape hatch. Never for a real measurement: two rungs
        # scored over different item counts are not comparable, which is why
        # the run parameters record what was actually used.
        scenarios = scenarios[: args.n_items]

    started = time.perf_counter()
    result = await run_load(
        scenarios,
        base_url=base_url,
        model=cfg.served_model_name,
        concurrency=args.concurrency,
        # Every item exactly once. `run_load` cycles the set to fill the count,
        # so asking for precisely len(scenarios) is what stops an item being
        # scored twice and another not at all.
        num_requests=len(scenarios),
        api_key=args.api_key,
        request_timeout=args.request_timeout,
        collect_output=True,
    )
    elapsed = time.perf_counter() - started

    results = score_records(result.records, answers, suite.scorer)
    values = aggregate(results)
    values.update(
        {
            "requests_ok": float(len(result.successful)),
            "requests_total": float(len(result.records)),
            "duration_s": elapsed,
        }
    )

    # Read the reference before writing our own file, so that a re-run of the
    # reference rung under a --suffix compares against its earlier pass instead
    # of against the file it is in the middle of replacing. That comparison is
    # the determinism check: same weights, same prompts, different batch
    # composition.
    out_path = items_path(suite.id, cfg.id, args.suffix)
    reference_path = items_path(suite.id, args.reference)
    eval_manifest = _manifest_digest()
    stale_reference = False

    if reference_path != out_path:
        reference = read_reference(reference_path)
        reference_meta = read_meta(reference_path)
        recorded = reference_meta.get("eval_manifest")
        # A reference built against different eval bytes is comparing answers
        # to different questions. Warn rather than refuse: the accuracy in this
        # cell is still sound, and only the agreement column is affected.
        stale_reference = bool(reference) and recorded is not None and recorded != eval_manifest
        if stale_reference:
            print(
                f"  warning: the reference answers in {reference_path.name} were "
                f"produced against eval sets {recorded}, not the {eval_manifest} "
                f"being scored now. agreement_with_reference compares answers to "
                f"different questions. Re-run --config-id {args.reference} first.",
                file=sys.stderr,
            )
        values.update(agreement(results, reference))

        overlap = values.get("agreement_n")
        if overlap is not None and overlap < len(results):
            print(
                f"  warning: the reference covers only {int(overlap)} of "
                f"{len(results)} items scored here. Agreement is over the "
                f"overlap; the two runs did not see the same set.",
                file=sys.stderr,
            )

    write_items(
        out_path,
        results,
        {
            "config_id": cfg.id,
            "suite_id": suite.id,
            "concurrency": args.concurrency,
            "suffix": args.suffix,
            "n_items": len(results),
            "eval_manifest": eval_manifest,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )

    notes = {
        "finish_reasons": sorted({r.finish_reason or "none" for r in results}),
        "reference": args.reference,
        "reference_available": "agreement_with_reference" in values,
        "reference_stale": stale_reference,
        # Repo-relative when it sits under the repo, absolute otherwise: the
        # items store is redirectable, and a note that records where the
        # answers went must not be the thing that ends the run.
        "items_file": str(
            out_path.relative_to(PACKAGE_ROOT)
            if out_path.is_relative_to(PACKAGE_ROOT)
            else out_path
        ),
    }

    if values.get("requests_ok", 0) < values.get("requests_total", 0):
        failed = int(values["requests_total"] - values["requests_ok"])
        first = next((r.error for r in result.records if r.error), "no detail")
        print(
            f"  warning: {failed} request(s) failed and are not scored ({first}). "
            f"Accuracy here is over fewer items than the suite defines.",
            file=sys.stderr,
        )

    if values.get("truncated_rate", 0.0) > TRUNCATION_WARN:
        print(
            f"  warning: {values['truncated_rate']:.0%} of replies hit max_tokens "
            f"({suite.max_tokens}). A truncated reply is an unmeasured item, not a "
            f"wrong one, so this cell understates accuracy by up to that much. "
            f"Raise max_tokens for {suite.id} and re-run every rung, or read the "
            f"accuracy as a lower bound.",
            file=sys.stderr,
        )

    if values.get("unparseable_rate", 0.0) > UNPARSEABLE_WARN:
        print(
            f"  warning: no answer could be extracted from "
            f"{values['unparseable_rate']:.0%} of replies. If this rung is high on "
            f"the ladder, suspect the scorer and read {out_path.name}; if it is "
            f"near the bottom, this is the result -- instruction-following fails "
            f"before accuracy does.",
            file=sys.stderr,
        )

    return values, notes, results


async def preflight(cfg: LadderConfig, suite: Suite, base_url: str, args) -> int:
    """Prove the server can be scored at all, before spending the pass.

    One assumption, and it is the mirror image of the one `ladder.run` checks.
    That module needs `ignore_eos` to be *honoured*; this one needs it to be
    absent, so that generation stops where the model would stop. A server
    configured to force length -- `--ignore-eos` on the command line, say --
    produces max_tokens of text for every item, and every reply is then either
    truncated or padded past its answer. Nothing errors; the accuracy is just
    wrong, uniformly, in a way that looks like a bad checkpoint.
    """
    scenarios, answers = load_suite(suite, verify=not args.no_verify)
    result = await run_load(
        scenarios[:4],
        base_url=base_url,
        model=cfg.served_model_name,
        concurrency=1,
        num_requests=4,
        collect_output=True,
    )

    ok = result.successful
    if not ok:
        first = next((r.error for r in result.records if r.error), "no detail")
        print(f"preflight FAILED: no request succeeded ({first})", file=sys.stderr)
        return 1

    results = score_records(result.records, answers, suite.scorer)
    reasons = sorted({r.finish_reason or "none" for r in ok})
    extracted = [r.extracted for r in results]
    print(f"  finish reasons      : {reasons}")
    print(f"  extracted answers   : {extracted}")
    print(f"  expected answers    : {[r.expected for r in results]}")

    problems: list[str] = []
    if reasons == ["length"]:
        problems.append(
            "every reply stopped at max_tokens. The server is forcing length "
            "(ignore_eos), so nothing here stops where the model would stop and "
            "no reply can be scored honestly. Start it from scripts/serve_"
            f"{cfg.id}.sh, which does not pass --ignore-eos."
        )
    if all(e is None for e in extracted):
        problems.append(
            "no answer could be extracted from any of the four replies. Either "
            "the chat template is not rendering the prompt as built, or this "
            "rung cannot follow the format at all. Read the replies before "
            "trusting a full pass."
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
    if not args.config_id:
        print("--config-id is required", file=sys.stderr)
        return 2
    if args.config_id not in configs:
        print(f"unknown config {args.config_id!r}; known: {sorted(configs)}", file=sys.stderr)
        return 2

    cfg = configs[args.config_id]
    if not cfg.quality and not args.force:
        subset = sorted(c.id for c in configs.values() if c.quality)
        print(
            f"{cfg.id} is not in the quality subset ({', '.join(subset)}). The "
            f"subset is pinned in config/ladder.yaml so that the scored rungs "
            f"are a property of the study rather than of whoever typed the "
            f"loop. Use --force to score it anyway.",
            file=sys.stderr,
        )
        return 2

    base_url = args.server_url or cfg.base_url
    suites = select_suites([s.strip() for s in args.suites.split(",") if s.strip()])

    for suite in suites:
        # The context guard, reused from the speed sweep. The bound is
        # deliberately loose -- these prompts are a few hundred tokens and are
        # not length-pinned the way a prompt class is -- so what it really
        # catches is a concurrency above the slot count, which would queue and
        # is the one way this pass can silently measure the wrong thing.
        if problem := check_context_budget(cfg, args.concurrency, 4096 + suite.max_tokens):
            print(f"cannot run {suite.id}: {problem}", file=sys.stderr)
            return 2

    print(f"rung        : {cfg.id} ({cfg.name})")
    print(f"quant       : {cfg.quant_type}  ~{cfg.bpw_nominal} bpw nominal  [{cfg.group}]")
    print(f"server      : {base_url}")
    print(f"suites      : {', '.join(s.id for s in suites)}")
    print(f"concurrency : {args.concurrency}")
    print(f"reference   : {args.reference}" + ("  (this rung)" if args.reference == cfg.id else ""))

    if args.dry_run:
        for suite in suites:
            n = args.n_items or suite.n_items
            print(f"  would score {suite.id} over {n} items at c={args.concurrency}")
        return 0

    wait_for_ready(base_url, timeout=args.ready_timeout)
    info = server_info(base_url)
    print(f"llama.cpp   : {info.get('build_info') or 'build unreported'}")

    if cfg.served_model_name not in (info.get("served_models") or []):
        print(
            f"warning: served model {info.get('served_models')} does not include "
            f"{cfg.served_model_name!r} from config {cfg.id!r}. The running server "
            f"may not be the one this config describes.",
            file=sys.stderr,
        )

    if args.preflight:
        print("\npreflight ...")
        return await preflight(cfg, suites[0], base_url, args)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    artifact_dir = ARTIFACT_ROOT / f"{cfg.id}-quality-{stamp}"

    if args.mlflow:
        tracking.setup(args.experiment, tracking_uri=args.tracking_uri)

    parent_params = {
        **cfg.as_params(),
        **lock_entry(cfg.id),
        "llamacpp_build": info.get("build_info"),
        "server_url": base_url,
        "eval_manifest": _manifest_digest(),
        "reference": args.reference,
    }

    summary: list[str] = []
    ctx = (
        tracking.config_run(
            run_name=f"{cfg.id}-quality",
            params=parent_params,
            # `pass` separates this parent from the speed sweep's parent for the
            # same rung. The cells are already separated by carrying suite_id
            # instead of class_id, which is what keeps ladder.report's curve
            # from ever seeing them.
            tags={"engine": "llama.cpp", "pass": "quality"},
        )
        if args.mlflow
        else _null_context()
    )

    with ctx:
        for suite in suites:
            cell_id = f"{suite.id}__c{args.concurrency}"
            print(f"\nscoring {cell_id} ...")
            values, notes, results = await run_suite(suite, cfg, base_url, args)
            line = format_suite(suite, values)
            print(line)
            summary.append(line)

            if args.mlflow:
                cell_params = {
                    **cfg.as_params(),
                    **lock_entry(cfg.id),
                    **suite.as_params(),
                    "concurrency": args.concurrency,
                    "n_items_run": len(results),
                    "reference": args.reference,
                    "suffix": args.suffix,
                    "ignore_eos": False,
                }
                with tracking.cell_run(run_name=cell_id, params=cell_params):
                    tracking.log_cell_results(
                        values,
                        notes,
                        [r.to_dict() for r in results],
                        artifact_dir,
                        cell_id,
                    )

    print("\nsummary")
    for line in summary:
        print(line)
    if args.mlflow:
        print(f"\nartifacts: {artifact_dir}")
    return 0


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
    parser.add_argument("--suites", default="", help="Comma separated ids; default all enabled")
    # One value, not a list. Scoring at two concurrencies would produce two
    # accuracies for one rung that differ only by batch numerics, and nothing
    # downstream would know which to plot.
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Pinned across rungs; 8 is the slot count, chosen for wall clock",
    )
    parser.add_argument(
        "--reference",
        default=DEFAULT_REFERENCE,
        help="Rung whose per-item answers every other rung is compared against",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Tag this run's items file, e.g. --suffix c1 for the determinism check",
    )
    parser.add_argument(
        "--n-items", type=int, default=0, help="Truncate each suite; smoke tests only"
    )
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--ready-timeout", type=float, default=900.0)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check that generation stops naturally and parses, then exit",
    )
    parser.add_argument(
        "--force", action="store_true", help="Score a rung not in the quality subset"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the eval-set digest check (for a locally modified set)",
    )
    parser.add_argument("--no-mlflow", dest="mlflow", action="store_false", default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()

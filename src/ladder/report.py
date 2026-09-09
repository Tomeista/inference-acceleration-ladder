"""Aggregate the swept cells into the curve, across re-measurement passes.

Until now "read the results" meant filtering the MLflow UI by hand, which was
fine while every cell had been measured exactly once. It stops being fine the
moment a class is re-measured: MLflow appends rather than overwrites, so a
rerun of three classes leaves two parent runs with the same name and two cells
for every `(config_id, class_id, concurrency)` it touched. Averaging those --
which is what any straightforward group-by does -- silently blends a discarded
measurement with the one that replaced it and reports a number that was never
observed.

So the merge rule is applied here, in code, once:

    the most recent cell wins for each (config_id, class_id, concurrency)

That rule needs no list of which classes were rerun and no bookkeeping tag. A
class measured only in the first pass keeps its original cell; a class measured
again takes the newer one; a rerun that died halfway leaves the rungs it never
reached untouched. Superseded cells are counted and reportable rather than
quietly dropped, because "this number replaced an earlier one" is exactly the
kind of thing that should be visible when reading a result.

    python -m ladder.report                 # the curve, valid cells only
    python -m ladder.report --all           # every cell, including superseded
    python -m ladder.report --csv out.csv   # the merged table

The same merge rule, and the same module, serve the quality pass:

    python -m ladder.report --quality       # accuracy against bits per weight

Two passes rather than two reports, because the hazard is identical -- a
re-scored suite appends a second cell exactly as a re-measured class does -- and
because the one thing that must never happen is the two being averaged together.
They are kept apart by what a cell carries: a speed cell has `class_id`, a
quality cell has `suite_id`, and neither query can see the other's rows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ladder.run import DEFAULT_EXPERIMENT, DEFAULT_TRACKING_URI

# The prefill-reuse floor, below which a cell's TTFT describes work the server
# skipped rather than work the rung can do.
#
# The value has to sit in a gap that the data itself defines. `c2_longform`
# legitimately reads 0.97 -- consecutive prompts share a few tokens of chat
# template and nothing more -- while the cells damaged by slot prefix reuse
# came in at 0.76 and below. Anything in (0.76, 0.97] separates them; 0.95 is
# chosen to sit clear of both edges. Note this is deliberately looser than the
# 0.98 that `ladder.run` warns at: a warning while measuring should fire early,
# but discarding a cell afterwards should be certain.
PREFILL_FLOOR = 0.95

# The identity of a measurement. Two rows sharing these three describe the same
# experiment run twice, which is the whole reason this module exists.
CELL_KEY = ["params.config_id", "params.class_id", "params.concurrency"]

# The same idea for the quality pass. `suite_id` rather than `class_id` is what
# keeps the two apart everywhere: a quality cell never carries `class_id`, so
# `load_cells` below cannot see it and the speed curve cannot accidentally
# average an accuracy into a throughput.
#
# `suffix` is part of the identity rather than something to merge away. The
# determinism check re-runs the reference rung at a different concurrency and
# tags it; that row is not a superseding re-measurement of the main one, it is a
# second measurement meant to be read beside it.
QUALITY_KEY = [
    "params.config_id",
    "params.suite_id",
    "params.concurrency",
    "params.suffix",
]

# Above this share of replies cut off at max_tokens, the cell understates
# accuracy by an unknown amount: a truncated reply is an unmeasured item rather
# than a wrong one. Matches `quality.TRUNCATION_WARN`, which fires while
# measuring; this is the same rule applied when deciding what to plot.
TRUNCATION_CEILING = 0.05

CURVE_COLUMNS = [
    "params.config_id",
    "params.ladder_group",
    "params.quant_family",
    "params.class_id",
    "params.concurrency",
    "params.bpw_measured",
    "metrics.output_tps",
    "metrics.ttft_s_p50",
    "metrics.ttft_s_p95",
    "metrics.tpot_s_p50",
    "metrics.prefill_processed_ratio",
    "metrics.requests_ok",
    "metrics.requests_total",
    "metrics.requests_waiting_peak",
]

QUALITY_COLUMNS = [
    "params.config_id",
    "params.ladder_group",
    "params.quant_family",
    "params.suite_id",
    "params.concurrency",
    "params.suffix",
    "params.bpw_measured",
    "params.dataset",
    "params.dataset_revision",
    "metrics.n_items",
    "metrics.accuracy",
    "metrics.accuracy_ci_lo",
    "metrics.accuracy_ci_hi",
    "metrics.accuracy_parsed",
    "metrics.agreement_with_reference",
    "metrics.unparseable_rate",
    "metrics.truncated_rate",
    "metrics.repetition_ratio",
    "metrics.requests_ok",
    "metrics.requests_total",
]


def load_cells(tracking_uri: str, experiment: str, *, marker: str = "params.class_id") -> "Any":
    """Every child (cell) run in the experiment, newest last.

    Parent runs carry no `class_id`, which is what distinguishes them; they are
    dropped rather than filtered on `tags.mlflow.parentRunId`, because that tag
    has moved between MLflow versions and `class_id` is ours.

    `marker` is the column that identifies a cell of the kind wanted, and it is
    also what separates the two passes: speed cells carry `class_id` and quality
    cells carry `suite_id`, so each query sees only its own and neither has to
    know the other exists.
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    df = mlflow.search_runs(experiment_names=[experiment], output_format="pandas")
    if df.empty:
        return df

    if marker not in df.columns:
        return df.iloc[0:0]

    cells = df[df[marker].notna()].copy()
    return cells.sort_values("start_time")


def select_cells(cells: "Any", key: list[str] | None = None) -> tuple["Any", "Any"]:
    """Split cells into (winners, superseded) by the most-recent-wins rule.

    Pure, and separate from `load_cells`, so the merge rule can be tested
    against a hand-built frame instead of a live tracking store. Parameterized
    over the key so the quality pass reuses the rule rather than growing a
    second, subtly different copy of it.
    """
    key = key or CELL_KEY
    if cells.empty:
        return cells, cells

    missing = [c for c in key if c not in cells.columns]
    if missing:
        raise KeyError(f"cells are missing identifying columns: {missing}")

    ordered = cells.sort_values("start_time")
    winners = ordered.drop_duplicates(key, keep="last")
    superseded = ordered.drop(index=winners.index)
    return winners, superseded


def cell_problems(row: "Any") -> list[str]:
    """Validity checks, as a list of reasons this cell should not be plotted.

    An empty list means the cell is usable. These mirror the guards
    `ladder.run` prints while sweeping; repeating them here is deliberate,
    because the sweep's warnings scroll past in a log and the decision about
    what enters a plot is made at this end.
    """
    problems: list[str] = []

    ratio = row.get("metrics.prefill_processed_ratio")
    if ratio is not None and ratio == ratio and ratio < PREFILL_FLOOR:
        problems.append(f"prefill={ratio:.2f}")

    ok = row.get("metrics.requests_ok")
    total = row.get("metrics.requests_total")
    if ok is not None and total is not None and ok == ok and total == total and ok < total:
        problems.append(f"failed={int(total - ok)}")

    # Logged as a JSON string tag, so "false" is the literal to look for. Its
    # absence is not a pass -- older cells predate the tag -- so only an
    # explicit false counts.
    capped = row.get("tags.note.length_capped")
    if isinstance(capped, str) and capped.strip().lower() == "false":
        problems.append("not length-capped")

    waiting = row.get("metrics.requests_waiting_peak")
    if waiting is not None and waiting == waiting and waiting > 0:
        problems.append(f"queued={int(waiting)}")

    return problems


def quality_problems(row: "Any") -> list[str]:
    """Reasons an accuracy should not be read as this rung's accuracy.

    Shorter than `cell_problems` because most of the speed pass's hazards do not
    apply: a reused prefix or a queued request changes how fast an answer
    arrived, not what it said. What does apply is anything that means the score
    was computed over fewer items than the suite defines.
    """
    problems: list[str] = []

    truncated = row.get("metrics.truncated_rate")
    if truncated is not None and truncated == truncated and truncated > TRUNCATION_CEILING:
        # Understates accuracy by an unknown amount, so the number is a lower
        # bound rather than a measurement.
        problems.append(f"truncated={truncated:.0%}")

    ok = row.get("metrics.requests_ok")
    total = row.get("metrics.requests_total")
    if ok is not None and total is not None and ok == ok and total == total and ok < total:
        problems.append(f"failed={int(total - ok)}")

    return problems


def render_quality(winners: "Any", *, only_valid: bool = True) -> list[str]:
    """The damage curve, one block per suite, ordered by bits per weight.

    Accuracy is never printed without its interval. At 250 items the interval is
    about +/-6 points, which is wider than the difference between most adjacent
    rungs -- so a reader who sees only the point estimate will find a knee
    wherever the noise happens to dip. `agree` is the column with the resolution
    to locate one.
    """
    lines: list[str] = []
    if winners.empty:
        return ["no quality cells found"]

    df = winners.copy()
    df["_bpw"] = df["params.bpw_measured"].astype(float, errors="ignore")
    df["_problems"] = [quality_problems(row) for _, row in df.iterrows()]
    if only_valid:
        df = df[df["_problems"].map(len) == 0]
        if df.empty:
            return ["every quality cell was excluded by a validity check; rerun with --all"]

    # Only worth a column when a determinism re-run is actually present.
    has_variant = any(
        isinstance(v, str) and v.strip() for v in df.get("params.suffix", [])
    )

    for suite_id, block in df.groupby("params.suite_id", sort=True):
        lines.append("")
        lines.append(f"{suite_id}")
        header = (
            f"  {'rung':14s} {'group':16s} {'fam':7s} {'bpw':>6s} {'n':>5s} "
            f"{'acc':>7s} {'95% CI':>16s} {'agree':>7s} "
            f"{'unparse':>8s} {'trunc':>7s} {'rep':>6s}"
        )
        if has_variant:
            header += f" {'variant':>8s}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        block = block.sort_values("_bpw", ascending=False)
        for _, row in block.iterrows():
            problems = row["_problems"]
            lo = row.get("metrics.accuracy_ci_lo")
            hi = row.get("metrics.accuracy_ci_hi")
            interval = (
                f"[{lo:.3f}, {hi:.3f}]"
                if lo is not None and hi is not None and lo == lo and hi == hi
                else "-"
            )
            line = (
                f"  {str(row['params.config_id']):14s} "
                f"{str(row.get('params.ladder_group', '')):16s} "
                f"{str(row.get('params.quant_family', '')):7s} "
                f"{_fmt(row['_bpw'], 1, 2, 6)} "
                f"{_fmt(row.get('metrics.n_items'), 1, 0, 5)} "
                f"{_fmt(row.get('metrics.accuracy'), 1, 3, 7)} "
                f"{interval:>16s} "
                f"{_fmt(row.get('metrics.agreement_with_reference'), 1, 3, 7)} "
                f"{_fmt(row.get('metrics.unparseable_rate'), 1, 3, 8)} "
                f"{_fmt(row.get('metrics.truncated_rate'), 1, 3, 7)} "
                f"{_fmt(row.get('metrics.repetition_ratio'), 1, 3, 6)}"
            )
            if has_variant:
                variant = row.get("params.suffix")
                line += f" {(variant if isinstance(variant, str) and variant else ''):>8s}"
            if problems:
                line += "  <- " + ", ".join(problems)
            lines.append(line)

    # The reference rung has no agreement figure by construction -- it would be
    # comparing a file with itself -- so say why rather than leaving a dash a
    # reader has to guess at.
    lines.append("")
    lines.append(
        "  agree = fraction of items answered identically to the reference rung; "
        "blank on the reference itself."
    )
    return lines


def _fmt(value: Any, scale: float = 1.0, digits: int = 1, width: int = 8) -> str:
    if value is None or value != value:
        return "-".rjust(width)
    return f"{float(value) * scale:.{digits}f}".rjust(width)


def render(winners: "Any", *, only_valid: bool = True) -> list[str]:
    """The curve, one block per (class, concurrency), ordered by bits per weight."""
    lines: list[str] = []
    if winners.empty:
        return ["no cells found"]

    df = winners.copy()
    df["_bpw"] = df["params.bpw_measured"].astype(float, errors="ignore")
    df["_problems"] = [cell_problems(row) for _, row in df.iterrows()]
    if only_valid:
        df = df[df["_problems"].map(len) == 0]
        if df.empty:
            return ["every cell was excluded by a validity check; rerun with --all"]

    for (class_id, conc), block in df.groupby(
        ["params.class_id", "params.concurrency"], sort=True
    ):
        lines.append("")
        lines.append(f"{class_id}  c={conc}")
        header = (
            f"  {'rung':14s} {'group':16s} {'fam':7s} {'bpw':>6s} "
            f"{'out_tps':>8s} {'ttft_p50':>9s} {'tpot_p50':>9s} {'prefill':>8s}"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        block = block.sort_values("_bpw", ascending=False)
        for _, row in block.iterrows():
            problems = row["_problems"]
            flag = ("  <- " + ", ".join(problems)) if problems else ""
            lines.append(
                f"  {str(row['params.config_id']):14s} "
                f"{str(row.get('params.ladder_group', '')):16s} "
                f"{str(row.get('params.quant_family', '')):7s} "
                f"{_fmt(row['_bpw'], 1, 2, 6)} "
                f"{_fmt(row.get('metrics.output_tps'), 1, 1, 8)} "
                f"{_fmt(row.get('metrics.ttft_s_p50'), 1000, 1, 9)} "
                f"{_fmt(row.get('metrics.tpot_s_p50'), 1000, 1, 9)} "
                f"{_fmt(row.get('metrics.prefill_processed_ratio'), 1, 2, 8)}"
                f"{flag}"
            )
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show cells that failed a validity check instead of hiding them",
    )
    parser.add_argument(
        "--quality",
        action="store_true",
        help="Read the accuracy pass (ladder.quality) instead of the speed curve",
    )
    parser.add_argument("--csv", type=Path, help="Write the merged cell table here")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # The two passes differ in four places and are otherwise the same report:
    # which column marks a cell, what identifies a repeat of it, which checks
    # disqualify it, and how a row is drawn.
    marker = "params.suite_id" if args.quality else "params.class_id"
    key = QUALITY_KEY if args.quality else CELL_KEY
    columns = QUALITY_COLUMNS if args.quality else CURVE_COLUMNS
    problems_of = quality_problems if args.quality else cell_problems
    draw = render_quality if args.quality else render

    cells = load_cells(args.tracking_uri, args.experiment, marker=marker)
    if cells.empty:
        kind = "quality" if args.quality else "speed"
        print(f"no {kind} cells in experiment {args.experiment!r} at {args.tracking_uri}")
        return 1

    winners, superseded = select_cells(cells, key)

    print(f"{len(winners)} cells, from {len(cells)} measurements")
    if len(superseded):
        # Named rather than counted: which classes got re-measured is the first
        # thing a reader of a merged table needs to know.
        redone = sorted(
            {
                f"{r[marker]}/c{r['params.concurrency']}"
                for _, r in superseded.iterrows()
            }
        )
        print(
            f"{len(superseded)} superseded by a later pass "
            f"({', '.join(redone)}); the newest measurement of each is used"
        )

    problems = {
        f"{r['params.config_id']}/{r[marker]}/c{r['params.concurrency']}": p
        for _, r in winners.iterrows()
        if (p := problems_of(r))
    }
    if problems:
        print(f"{len(problems)} cell(s) fail a validity check:", file=sys.stderr)
        for cell_id, reasons in sorted(problems.items()):
            print(f"  {cell_id}: {', '.join(reasons)}", file=sys.stderr)
        if not args.all:
            print("  (excluded from the tables below; --all to show them)", file=sys.stderr)

    for line in draw(winners, only_valid=not args.all):
        print(line)

    if args.csv:
        columns = [c for c in columns if c in winners.columns]
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        out = winners[columns].sort_values(
            [marker, "params.concurrency", "params.bpw_measured"]
        )
        out.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}  ({len(out)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

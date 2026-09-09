"""Build the frozen eval sets. Run once; the output is committed.

Deliberately outside the run path, the same way `bench.build_prompts` is not
vendored into `harness/`: it needs the network and pyarrow, and nothing that
*measures* should need either. A clone can run the whole quality pass from the
committed `evals/` directory with no dataset access at all.

    python -m ladder.build_evals             # fetch, sample, write evals/
    python -m ladder.build_evals --check     # digests only, no network

What is pinned, and why each one matters
----------------------------------------
*The commit.* `suites.yaml` names a dataset revision, not `main`. MMLU and GSM8K
have both been re-uploaded before; a set that changes underneath the study makes
rungs measured weeks apart incomparable while every run still succeeds.

*The sample.* Fixed seed, and MMLU is stratified across all 57 subjects rather
than sampled flat -- a flat 250-of-14042 draw would over-weight the large
subjects and give a score that moves when the sample moves.

*The bytes.* Written with an explicit LF newline rather than the platform
default. `harness.scenarios.write_scenarios` opens in text mode, so on Windows
it would emit CRLF and a rebuild on the Linux GPU box would produce different
digests for identical content -- which is exactly the drift `manifest.json`
exists to catch, arriving from the tool that writes the manifest.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable

import httpx

from ladder.harness.scenarios import Scenario, Turn
from ladder.scoring import normalize_number
from ladder.suites import MANIFEST_PATH, Suite, digest, load_suites

SEED = 0

# Passed through to llama-server on every quality request. Three pins, and the
# absence of a fourth:
#
#   top_k 1 + temperature 0  greedy. Two rungs must differ because their weights
#                            differ, not because their samplers rolled apart.
#   seed                     belt and braces; greedy should not consult it.
#   enable_thinking false    matches prompts/*.jsonl. Left on, Qwen3 reasons for
#                            as long as it likes and GSM8K's output length -- and
#                            therefore its truncation rate -- becomes a property
#                            of the rung's verbosity rather than of the suite.
#
# No `ignore_eos`. The speed sweep pins output length with it so content cannot
# affect timing; scoring needs the opposite, and a test asserts it stays absent.
EXTRA_BODY: dict[str, Any] = {
    "top_k": 1,
    "seed": SEED,
    "chat_template_kwargs": {"enable_thinking": False},
}

MMLU_INSTRUCTION = (
    'Reply with exactly "Answer: X", where X is A, B, C or D. Do not explain.'
)
GSM8K_INSTRUCTION = (
    "Solve the problem step by step. Then give the final numeric answer on its "
    'own last line, in the form "#### <number>".'
)

LETTERS = "ABCD"


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------


def fetch_parquet(suite: Suite) -> "Any":
    """The pinned parquet, as a pyarrow table.

    Resolved at the suite's commit rather than at a branch, so re-running this
    a year from now rebuilds the same set or fails loudly.
    """
    import pyarrow.parquet as pq

    url = (
        f"https://huggingface.co/datasets/{suite.dataset}"
        f"/resolve/{suite.revision}/{suite.parquet}"
    )
    print(f"  fetching {url}")
    response = httpx.get(url, follow_redirects=True, timeout=120.0)
    response.raise_for_status()
    return pq.read_table(io.BytesIO(response.content))


# --------------------------------------------------------------------------
# builders -- one per `source` in config/suites.yaml
# --------------------------------------------------------------------------


def build_mmlu(suite: Suite, table: "Any") -> tuple[list[Scenario], list[dict], dict]:
    """250 items, stratified round-robin across every subject.

    Stratified rather than flat: subject sizes range from 100 to 1534, so a flat
    draw would be most of a score about professional law and high-school
    psychology. Round-robin over subjects shuffled at a fixed seed gives an even
    spread and a reproducible one.
    """
    rows = table.to_pylist()
    by_subject: dict[str, list[dict]] = {}
    for row in rows:
        by_subject.setdefault(row["subject"], []).append(row)

    rng = random.Random(SEED)
    for subject in sorted(by_subject):
        rng.shuffle(by_subject[subject])

    picked: list[dict] = []
    subjects = sorted(by_subject)
    depth = 0
    while len(picked) < suite.n_items:
        added = False
        for subject in subjects:
            if depth < len(by_subject[subject]):
                picked.append(by_subject[subject][depth])
                added = True
                if len(picked) == suite.n_items:
                    break
        if not added:
            break
        depth += 1

    scenarios, key = [], []
    for i, row in enumerate(picked):
        options = "\n".join(f"{LETTERS[j]}. {c}" for j, c in enumerate(row["choices"]))
        content = f"{row['question'].strip()}\n\n{options}\n\n{MMLU_INSTRUCTION}"
        scenario_id = f"{suite.id}-{i:04d}"
        scenarios.append(
            Scenario(
                scenario_id=scenario_id,
                class_id=suite.id,
                turns=[
                    Turn(
                        messages=[{"role": "user", "content": content}],
                        max_tokens=suite.max_tokens,
                        temperature=suite.temperature,
                        extra_body=dict(EXTRA_BODY),
                    )
                ],
            )
        )
        key.append(
            {
                "scenario_id": scenario_id,
                "answer": LETTERS[int(row["answer"])],
                "meta": {"subject": row["subject"]},
            }
        )

    return scenarios, key, {"subjects": len(subjects), "instruction": MMLU_INSTRUCTION}


def build_gsm8k(suite: Suite, table: "Any") -> tuple[list[Scenario], list[dict], dict]:
    """250 items sampled flat -- GSM8K has no subject axis to stratify over."""
    rows = table.to_pylist()
    rng = random.Random(SEED)
    picked = rng.sample(rows, min(suite.n_items, len(rows)))

    scenarios, key = [], []
    for i, row in enumerate(picked):
        content = f"{row['question'].strip()}\n\n{GSM8K_INSTRUCTION}"
        scenario_id = f"{suite.id}-{i:04d}"
        scenarios.append(
            Scenario(
                scenario_id=scenario_id,
                class_id=suite.id,
                turns=[
                    Turn(
                        messages=[{"role": "user", "content": content}],
                        max_tokens=suite.max_tokens,
                        temperature=suite.temperature,
                        extra_body=dict(EXTRA_BODY),
                    )
                ],
            )
        )
        # The gold answer sits after the "#### " marker in the reference
        # solution. Normalized at build time with the same function the scorer
        # uses, so "1,000" in the dataset and "1000" from the model are one
        # answer rather than a scoring bug discovered on the GPU box.
        gold = row["answer"].rsplit("####", 1)[-1]
        key.append(
            {
                "scenario_id": scenario_id,
                "answer": normalize_number(gold),
                "meta": {},
            }
        )

    return scenarios, key, {"instruction": GSM8K_INSTRUCTION}


BUILDERS: dict[str, Callable[[Suite, Any], tuple[list[Scenario], list[dict], dict]]] = {
    "mmlu": build_mmlu,
    "gsm8k": build_gsm8k,
}


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------


def _write_lines(path: Path, lines: list[str]) -> None:
    """Explicit LF, so the digest does not depend on which OS built the set."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for line in lines:
            fh.write(line + "\n")


def build(suite: Suite) -> dict:
    if suite.source not in BUILDERS:
        raise KeyError(
            f"suite {suite.id!r} names source {suite.source!r} with no builder; "
            f"known: {sorted(BUILDERS)}"
        )
    table = fetch_parquet(suite)
    scenarios, key, extra = BUILDERS[suite.source](suite, table)

    if len(scenarios) != suite.n_items:
        print(
            f"  warning: {suite.id} produced {len(scenarios)} items, not "
            f"{suite.n_items}; the source split may be smaller than requested",
            file=sys.stderr,
        )

    _write_lines(suite.prompt_file, [s.to_json() for s in scenarios])
    _write_lines(suite.key_file, [json.dumps(row, ensure_ascii=False) for row in key])

    return {
        "suite_id": suite.id,
        "source": suite.source,
        "dataset": suite.dataset,
        "revision": suite.revision,
        "parquet": suite.parquet,
        "license": suite.license,
        "n_items": len(scenarios),
        "max_tokens": suite.max_tokens,
        "temperature": suite.temperature,
        "scorer": suite.scorer,
        "shots": 0,
        **extra,
        "sha256_16": digest(suite.prompt_file),
        "key_sha256_16": digest(suite.key_file),
    }


def check() -> int:
    """Re-hash the committed sets against the manifest. No network."""
    if not MANIFEST_PATH.exists():
        print(f"no {MANIFEST_PATH}", file=sys.stderr)
        return 1
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    suites = load_suites()
    bad = 0
    for suite_id, recorded in sorted((manifest.get("suites") or {}).items()):
        suite = suites[suite_id]
        for path, field in ((suite.prompt_file, "sha256_16"), (suite.key_file, "key_sha256_16")):
            actual = digest(path) if path.exists() else "MISSING"
            ok = actual == recorded[field]
            bad += 0 if ok else 1
            print(f"  {path.name:24s} {actual}  {'ok' if ok else '<- ' + recorded[field]}")
    return 1 if bad else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--suites", default="", help="Comma separated ids; default all enabled")
    parser.add_argument(
        "--check", action="store_true", help="Verify committed digests and exit; no network"
    )
    args = parser.parse_args()

    if args.check:
        return check()

    requested = [s.strip() for s in args.suites.split(",") if s.strip()]
    suites = load_suites()
    chosen = [suites[s] for s in requested] if requested else [
        s for s in suites.values() if s.enabled
    ]

    # Merge rather than replace: building one suite must not drop the other's
    # entry and leave a manifest that no longer describes what is on disk.
    manifest: dict[str, Any] = {"seed": SEED, "built_by": "ladder.build_evals", "suites": {}}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest.setdefault("suites", {})

    for suite in chosen:
        print(f"{suite.id}: {suite.name}")
        manifest["suites"][suite.id] = build(suite)
        entry = manifest["suites"][suite.id]
        print(f"  wrote {entry['n_items']} items  {entry['sha256_16']} / {entry['key_sha256_16']}")

    manifest["seed"] = SEED
    manifest["built_by"] = "ladder.build_evals"
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

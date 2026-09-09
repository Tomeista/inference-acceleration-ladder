"""The quality pass: the frozen sets, the guards, and a whole cell end to end.

The mock has no model, so nothing here is a quality result. What it checks is
everything between the server and the number -- digest verification, the
answer-key join, extraction, the reference comparison, the merge rule -- which
would otherwise be exercised for the first time on the GPU box, at the end of a
two-hour pass, with no way to tell a scorer bug from a bad checkpoint.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from ladder import quality as quality_mod
from ladder.report import QUALITY_KEY, quality_problems, render_quality, select_cells
from ladder.server import load_configs
from ladder.suites import (
    EVALS_DIR,
    MANIFEST_PATH,
    check_digests,
    digest,
    load_suite,
    load_suites,
    select_suites,
)


@pytest.fixture(scope="module")
def suites():
    return load_suites()


@pytest.fixture(scope="module")
def configs():
    return load_configs()


@pytest.fixture
def args(tmp_path, monkeypatch):
    """A parsed CLI with the items store redirected out of the repo."""
    monkeypatch.setattr(quality_mod, "QUALITY_ROOT", tmp_path)
    parsed = quality_mod.build_parser().parse_args([])
    parsed.n_items = 8
    parsed.concurrency = 2
    return parsed


# --------------------------------------------------------------------------
# the frozen eval sets
# --------------------------------------------------------------------------


def test_the_eval_files_are_the_bytes_the_manifest_recorded(suites):
    """The same guarantee prompts/manifest.json gives the speed sets.

    A regenerated or line-ending-mangled eval set would still load, still score,
    and still look plausible -- while making two rungs measured either side of
    the change incomparable. Only the digest notices.
    """
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    recorded = manifest["suites"]
    assert recorded, "manifest lists no suites"

    for suite_id, stats in sorted(recorded.items()):
        suite = suites[suite_id]
        assert digest(suite.prompt_file) == stats["sha256_16"], f"{suite_id}.jsonl drifted"
        assert digest(suite.key_file) == stats["key_sha256_16"], f"{suite_id}.key.jsonl drifted"


def test_every_enabled_suite_has_its_files():
    for suite in select_suites():
        assert suite.prompt_file.exists(), f"{suite.id} has no prompt file"
        assert suite.key_file.exists(), f"{suite.id} has no key file"


def test_the_frozen_files_are_lf_so_a_rebuild_reproduces_the_digest(suites):
    """Written with an explicit newline rather than the platform default.

    `harness.scenarios.write_scenarios` opens in text mode, so building on
    Windows would emit CRLF and building on the Linux GPU box would emit LF --
    identical content, different digests, and the manifest would report drift
    that is really just a change of machine.
    """
    for suite in suites.values():
        raw = suite.prompt_file.read_bytes()
        assert b"\r\n" not in raw, f"{suite.prompt_file.name} has CRLF line endings"


def test_a_tampered_eval_set_is_refused(suites, tmp_path, monkeypatch):
    """The check has to fail closed, or it is decoration."""
    monkeypatch.setattr("ladder.suites.EVALS_DIR", tmp_path)
    suite = suites["mmlu"]
    suite.prompt_file.write_text('{"scenario_id": "x"}\n', encoding="utf-8")
    suite.key_file.write_text('{"scenario_id": "x", "answer": "A"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not the frozen set"):
        check_digests(suite)


def test_prompts_and_key_describe_the_same_items(suites):
    """Two files that drift apart still run, and score noise."""
    for suite in suites.values():
        scenarios, answers = load_suite(suite)
        assert {s.scenario_id for s in scenarios} == set(answers)
        assert len(answers) == suite.n_items


def test_a_key_that_disagrees_with_the_prompts_is_refused(suites, tmp_path, monkeypatch):
    # Read the real set first: `prompt_file` resolves against EVALS_DIR at
    # access time, so everything after the patch points at tmp_path.
    scenarios, _ = load_suite(suites["mmlu"])

    monkeypatch.setattr("ladder.suites.EVALS_DIR", tmp_path)
    suite = suites["mmlu"]
    suite.prompt_file.write_text(
        "\n".join(s.to_json() for s in scenarios[:4]) + "\n", encoding="utf-8"
    )
    suite.key_file.write_text(
        json.dumps({"scenario_id": "mmlu-9999", "answer": "A"}) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="different items"):
        load_suite(suite, verify=False)


# --------------------------------------------------------------------------
# the pins that make two rungs comparable
# --------------------------------------------------------------------------


def test_no_quality_prompt_pins_output_length(suites):
    """The exact inversion of the speed pass, and the reason for a separate set.

    `ladder.run` sets ignore_eos so that content cannot affect timing. Scoring
    needs generation to stop where the model would stop; with ignore_eos on,
    every reply runs to max_tokens and is either truncated or padded past its
    answer, uniformly, in a way that looks like a bad checkpoint.
    """
    for suite in suites.values():
        scenarios, _ = load_suite(suite)
        for scenario in scenarios:
            for turn in scenario.turns:
                assert "ignore_eos" not in turn.extra_body, scenario.scenario_id


def test_every_quality_prompt_decodes_greedily(suites):
    """Two rungs must differ because their weights differ, not their samplers."""
    for suite in suites.values():
        scenarios, _ = load_suite(suite)
        for scenario in scenarios:
            for turn in scenario.turns:
                assert turn.temperature == 0.0
                assert turn.extra_body["top_k"] == 1
                # Left on, Qwen3 reasons for as long as it likes and GSM8K's
                # truncation rate becomes a property of the rung's verbosity.
                assert turn.extra_body["chat_template_kwargs"]["enable_thinking"] is False


def test_the_quality_subset_still_spans_the_axis(configs):
    """Nine rungs is a sampling decision; it must not become a narrow one.

    The claim the pass makes is about degradation from full precision to the
    published floor, so the subset has to reach both ends and include the
    reference every other rung is scored against.
    """
    subset = [c for c in configs.values() if c.quality]
    assert len(subset) >= 6, f"only {len(subset)} rungs scored; the curve will be coarse"

    widths = sorted(c.bpw_nominal for c in subset)
    assert widths[-1] > 15.0, "no full-precision rung to score everything else against"
    assert widths[0] < 3.0, "the scored subset stops above the interesting end"

    anchors = [c for c in subset if c.group == "anchor"]
    assert len(anchors) == 1, "the reference rung is not in the scored subset"
    assert anchors[0].id == quality_mod.DEFAULT_REFERENCE

    # A gap wide enough to hide the knee would defeat the point of sampling.
    gaps = [hi - lo for lo, hi in zip(widths, widths[1:])]
    assert max(gaps) <= 8.0, f"widest gap is {max(gaps):.2f} bpw"


def test_the_scored_rungs_reach_the_bottom_of_the_ladder(configs):
    """The question is where quality stops being worth the speed. That answer
    lives at the bottom, so the floor has to be in the subset."""
    subset = [c for c in configs.values() if c.quality]
    floor = min(c.bpw_nominal for c in configs.values())
    assert min(c.bpw_nominal for c in subset) == floor


# --------------------------------------------------------------------------
# the suite config
# --------------------------------------------------------------------------


def test_unknown_key_in_a_suite_is_rejected(tmp_path):
    path = tmp_path / "suites.yaml"
    path.write_text(
        "defaults: {}\n"
        "suites:\n"
        "  - id: x\n"
        "    name: x\n"
        "    source: mmlu\n"
        "    dataset: d\n"
        "    revision: r\n"
        "    parquet: p\n"
        "    max_tokens: 16\n"
        "    scorer: mc\n"
        "    scorrer: mc\n",  # typo
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown keys"):
        load_suites(path)


def test_every_suite_names_a_scorer_that_exists(suites):
    from ladder.scoring import EXTRACTORS

    for suite in suites.values():
        assert suite.scorer in EXTRACTORS


def test_every_suite_pins_a_dataset_commit(suites):
    """`main` would let the benchmark change underneath the study."""
    for suite in suites.values():
        assert len(suite.revision) == 40, f"{suite.id} is not pinned to a commit"


def test_a_suite_records_itself_as_suite_id_not_class_id(suites):
    """Load-bearing. `report.load_cells` selects speed cells on class_id, so a
    quality cell carrying that name would be averaged into the speed curve."""
    params = suites["mmlu"].as_params()
    assert params["suite_id"] == "mmlu"
    assert "class_id" not in params


# --------------------------------------------------------------------------
# a whole cell, end to end
# --------------------------------------------------------------------------


def _expected_accuracy(letter: str, n: int) -> float:
    """What a server that always answers `letter` scores on the first n items."""
    rows = [json.loads(l) for l in (EVALS_DIR / "mmlu.key.jsonl").read_text().splitlines() if l]
    answers = [r["answer"] for r in rows[:n]]
    return answers.count(letter) / n


async def test_a_cell_scores_against_the_real_key(mock_llama, configs, args, monkeypatch):
    """A server with a known, fixed answer has a known, computable accuracy."""
    monkeypatch.setenv("MOCK_REPLY", "Answer: C")
    suite = select_suites(["mmlu"])[0]

    values, notes, results = await quality_mod.run_suite(
        suite, configs["bf16"], mock_llama, args
    )

    assert len(results) == 8
    assert values["accuracy"] == pytest.approx(_expected_accuracy("C", 8))
    assert values["unparseable_rate"] == 0.0
    assert values["truncated_rate"] == 0.0
    assert values["accuracy_ci_lo"] < values["accuracy"] < values["accuracy_ci_hi"]


async def test_the_reference_join_is_paired_per_item(mock_llama, configs, args, monkeypatch):
    """Score the reference, then a second rung, and check the comparison.

    Same answers means agreement 1.0 even though accuracy is far from it -- the
    point of the metric is that it measures divergence from the reference, not
    correctness.
    """
    monkeypatch.setenv("MOCK_REPLY", "Answer: C")
    suite = select_suites(["mmlu"])[0]

    reference_values, _, _ = await quality_mod.run_suite(
        suite, configs["bf16"], mock_llama, args
    )
    # The reference has nothing to compare itself against.
    assert "agreement_with_reference" not in reference_values

    values, notes, _ = await quality_mod.run_suite(suite, configs["q4_k_m"], mock_llama, args)
    assert values["agreement_with_reference"] == 1.0
    assert notes["reference_available"] is True

    monkeypatch.setenv("MOCK_REPLY", "Answer: A")
    diverged, _, _ = await quality_mod.run_suite(suite, configs["q3_k_m"], mock_llama, args)
    assert diverged["agreement_with_reference"] == 0.0
    # Accuracy moved too, but in the other direction: agreement and accuracy
    # are independent, which is exactly why both are reported.
    assert diverged["accuracy"] > values["accuracy"]


async def test_no_reference_yields_no_agreement_metric(mock_llama, configs, args, monkeypatch):
    """A sweep that starts somewhere other than the reference still measures
    accuracy; it just cannot measure divergence, and must not claim zero."""
    monkeypatch.setenv("MOCK_REPLY", "Answer: C")
    suite = select_suites(["mmlu"])[0]
    values, notes, _ = await quality_mod.run_suite(suite, configs["q4_k_m"], mock_llama, args)
    assert "agreement_with_reference" not in values
    assert notes["reference_available"] is False


async def test_a_reference_from_different_eval_bytes_is_flagged(
    mock_llama, configs, args, monkeypatch, capsys
):
    """`results/` is machine state: gitignored, regenerated, easily stale.

    An items file left over from a smoke test or from before the eval sets were
    rebuilt joins by scenario_id perfectly well and yields an agreement figure
    about nothing. Item ids cannot detect that; the sidecar recording what
    produced them can.
    """
    monkeypatch.setenv("MOCK_REPLY", "Answer: C")
    suite = select_suites(["mmlu"])[0]

    await quality_mod.run_suite(suite, configs["bf16"], mock_llama, args)

    # Rewrite the reference's sidecar as though it came from another eval set.
    reference = quality_mod.items_path(suite.id, "bf16")
    sidecar = quality_mod.meta_path(reference)
    meta = json.loads(sidecar.read_text())
    meta["eval_manifest"] = "mmlu:deadbeefdeadbeef"
    sidecar.write_text(json.dumps(meta), encoding="utf-8")

    _, notes, _ = await quality_mod.run_suite(suite, configs["q4_k_m"], mock_llama, args)
    assert notes["reference_stale"] is True
    assert "were produced against eval sets" in capsys.readouterr().err


async def test_a_matching_reference_is_not_flagged(mock_llama, configs, args, monkeypatch):
    monkeypatch.setenv("MOCK_REPLY", "Answer: C")
    suite = select_suites(["mmlu"])[0]
    await quality_mod.run_suite(suite, configs["bf16"], mock_llama, args)
    _, notes, _ = await quality_mod.run_suite(suite, configs["q4_k_m"], mock_llama, args)
    assert notes["reference_stale"] is False


async def test_a_partial_reference_warns_about_the_overlap(
    mock_llama, configs, args, monkeypatch, capsys
):
    """Agreement over 4 of 8 items is not agreement over the suite."""
    monkeypatch.setenv("MOCK_REPLY", "Answer: C")
    suite = select_suites(["mmlu"])[0]

    args.n_items = 4
    await quality_mod.run_suite(suite, configs["bf16"], mock_llama, args)
    capsys.readouterr()

    args.n_items = 8
    values, _, _ = await quality_mod.run_suite(suite, configs["q4_k_m"], mock_llama, args)
    assert values["agreement_n"] == 4
    assert "reference covers only 4 of 8" in capsys.readouterr().err


async def test_truncation_is_counted_rather_than_scored_wrong(
    mock_llama, configs, args, monkeypatch
):
    """A reply cut off at max_tokens is an unmeasured item, not a wrong one."""
    monkeypatch.setenv("MOCK_REPLY", " ".join(str(i) for i in range(40)))
    suite = select_suites(["mmlu"])[0]
    values, _, _ = await quality_mod.run_suite(suite, configs["bf16"], mock_llama, args)
    assert values["truncated_rate"] == 1.0


async def test_a_cell_carries_no_boolean_into_the_metric_set(
    mock_llama, configs, args, monkeypatch
):
    """MLflow metrics are floats; the booleans belong in tags."""
    monkeypatch.setenv("MOCK_REPLY", "Answer: C")
    suite = select_suites(["mmlu"])[0]
    values, _, _ = await quality_mod.run_suite(suite, configs["bf16"], mock_llama, args)
    assert not any(isinstance(v, bool) for v in values.values())


async def test_every_item_is_scored_exactly_once(mock_llama, configs, args, monkeypatch):
    """`run_load` cycles its scenario set to fill the request count. Asking for
    anything but len(scenarios) would score one item twice and another never."""
    monkeypatch.setenv("MOCK_REPLY", "Answer: C")
    suite = select_suites(["mmlu"])[0]
    _, _, results = await quality_mod.run_suite(suite, configs["bf16"], mock_llama, args)
    assert len({r.scenario_id for r in results}) == len(results)


# --------------------------------------------------------------------------
# the guards
# --------------------------------------------------------------------------


async def test_a_rung_outside_the_subset_is_refused(args):
    """The scored set is a property of the study, not of whoever typed the loop."""
    args.config_id = "iq4_nl"  # in the ladder, not in the quality subset
    assert await quality_mod.main_async(args) == 2

    args.force = True
    args.dry_run = True
    assert await quality_mod.main_async(args) == 0


async def test_preflight_catches_a_server_forcing_length(
    mock_llama, configs, args, monkeypatch
):
    """The mirror image of run.py's preflight.

    That one fails when ignore_eos is dropped; this one fails when generation
    never stops on its own, because then nothing stops where the model would and
    no reply can be scored honestly.
    """
    monkeypatch.setenv("MOCK_REPLY", " ".join(str(i) for i in range(40)))
    suite = select_suites(["mmlu"])[0]
    assert await quality_mod.preflight(configs["bf16"], suite, mock_llama, args) == 1


async def test_preflight_passes_against_a_scoreable_server(
    mock_llama, configs, args, monkeypatch
):
    monkeypatch.setenv("MOCK_REPLY", "Answer: C")
    suite = select_suites(["mmlu"])[0]
    assert await quality_mod.preflight(configs["bf16"], suite, mock_llama, args) == 0


async def test_preflight_fails_when_nothing_parses(mock_llama, configs, args, monkeypatch):
    monkeypatch.setenv("MOCK_REPLY", "I could not possibly say")
    suite = select_suites(["mmlu"])[0]
    assert await quality_mod.preflight(configs["bf16"], suite, mock_llama, args) == 1


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def _qcell(config_id, suite_id, start_time, *, acc=0.7, trunc=0.0, ok=250, total=250, suffix=""):
    return {
        "start_time": pd.Timestamp(start_time, unit="s"),
        "params.config_id": config_id,
        "params.suite_id": suite_id,
        "params.concurrency": "8",
        "params.suffix": suffix,
        "params.bpw_measured": "4.91",
        "params.ladder_group": "ladder",
        "params.quant_family": "kquant",
        "metrics.n_items": 250.0,
        "metrics.accuracy": acc,
        "metrics.accuracy_ci_lo": acc - 0.06,
        "metrics.accuracy_ci_hi": acc + 0.06,
        "metrics.agreement_with_reference": 0.9,
        "metrics.unparseable_rate": 0.0,
        "metrics.truncated_rate": trunc,
        "metrics.repetition_ratio": 0.02,
        "metrics.requests_ok": float(ok),
        "metrics.requests_total": float(total),
    }


def test_a_rescored_suite_is_merged_by_the_same_rule_as_a_remeasured_class():
    """The hazard is identical -- MLflow appends -- so the rule is reused."""
    df = pd.DataFrame(
        [_qcell("q4_k_m", "mmlu", 100, acc=0.60), _qcell("q4_k_m", "mmlu", 200, acc=0.71)]
    )
    winners, superseded = select_cells(df, QUALITY_KEY)
    assert len(winners) == 1
    assert winners.iloc[0]["metrics.accuracy"] == 0.71
    assert len(superseded) == 1


def test_a_determinism_rerun_does_not_supersede_the_measurement():
    """`--suffix c1` is a second reading meant to sit beside the first, not a
    correction of it, so the suffix is part of a cell's identity."""
    df = pd.DataFrame(
        [_qcell("bf16", "mmlu", 100), _qcell("bf16", "mmlu", 200, suffix="c1")]
    )
    winners, superseded = select_cells(df, QUALITY_KEY)
    assert len(winners) == 2
    assert len(superseded) == 0


def test_a_heavily_truncated_cell_is_flagged():
    """Its accuracy is a lower bound, not a measurement."""
    assert quality_problems(pd.Series(_qcell("q2_k", "gsm8k", 100, trunc=0.4)))
    assert not quality_problems(pd.Series(_qcell("q2_k", "gsm8k", 100, trunc=0.0)))


def test_failed_requests_are_flagged():
    assert quality_problems(pd.Series(_qcell("q2_k", "gsm8k", 100, ok=240)))


def test_the_table_never_prints_accuracy_without_its_interval():
    """At 250 items the interval is wider than most adjacent rungs differ by. A
    point estimate on its own invites a reader to find a knee in the noise."""
    df = pd.DataFrame([_qcell("q4_k_m", "mmlu", 100, acc=0.7)])
    lines = render_quality(df)
    body = [l for l in lines if "q4_k_m" in l]
    assert body and "0.700" in body[0]
    assert "[0.640, 0.760]" in body[0]

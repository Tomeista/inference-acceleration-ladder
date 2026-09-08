"""The merge rule, which is the only thing standing between a rerun and a
silently wrong table.

Every test here builds its frame by hand rather than through MLflow. The
question being asked is "given two measurements of the same cell, which one
survives", and that question has nothing to do with a tracking store.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ladder.report import PREFILL_FLOOR, cell_problems, render, select_cells


def cell(
    config_id: str,
    class_id: str,
    concurrency: int,
    start_time: int,
    *,
    output_tps: float = 100.0,
    prefill: float = 1.0,
    ok: int = 16,
    total: int = 16,
    waiting: float = 0.0,
    length_capped: str = "true",
    bpw: float = 4.68,
) -> dict:
    return {
        "start_time": pd.Timestamp(start_time, unit="s"),
        "params.config_id": config_id,
        "params.class_id": class_id,
        "params.concurrency": str(concurrency),
        "params.bpw_measured": str(bpw),
        "params.ladder_group": "ladder",
        "params.quant_family": "kquant",
        "metrics.output_tps": output_tps,
        "metrics.ttft_s_p50": 0.05,
        "metrics.tpot_s_p50": 0.02,
        "metrics.prefill_processed_ratio": prefill,
        "metrics.requests_ok": float(ok),
        "metrics.requests_total": float(total),
        "metrics.requests_waiting_peak": waiting,
        "tags.note.length_capped": length_capped,
    }


def frame(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# --------------------------------------------------------------------------
# the merge rule
# --------------------------------------------------------------------------


def test_the_later_measurement_of_a_cell_wins():
    df = frame(
        cell("q4_k_m", "c1_chat", 8, 1000, output_tps=111.0, prefill=0.02),
        cell("q4_k_m", "c1_chat", 8, 2000, output_tps=222.0, prefill=0.99),
    )
    winners, superseded = select_cells(df)

    assert len(winners) == 1
    assert len(superseded) == 1
    assert winners.iloc[0]["metrics.output_tps"] == 222.0
    assert superseded.iloc[0]["metrics.output_tps"] == 111.0


def test_row_order_does_not_decide_the_winner():
    """The frame arrives in whatever order the store returns it.

    If this ever regressed to "keep the last row" rather than "keep the latest
    start_time", the merge would depend on MLflow's result ordering, which is
    not something this code should be trusting.
    """
    newest = cell("q4_k_m", "c1_chat", 8, 2000, output_tps=222.0)
    oldest = cell("q4_k_m", "c1_chat", 8, 1000, output_tps=111.0)

    winners, _ = select_cells(frame(newest, oldest))
    assert winners.iloc[0]["metrics.output_tps"] == 222.0

    winners, _ = select_cells(frame(oldest, newest))
    assert winners.iloc[0]["metrics.output_tps"] == 222.0


def test_a_class_measured_once_is_untouched_by_a_rerun_of_others():
    """c2_longform is not re-measured in the corrective pass.

    Its original cell has to survive a merge whose other classes all gained a
    newer measurement -- otherwise the cheap rerun would cost the one class it
    deliberately skipped.
    """
    df = frame(
        cell("q4_k_m", "c2_longform", 1, 1000, output_tps=50.0),
        cell("q4_k_m", "c1_chat", 1, 1000, output_tps=10.0),
        cell("q4_k_m", "c1_chat", 1, 2000, output_tps=99.0),
    )
    winners, superseded = select_cells(df)

    by_class = {r["params.class_id"]: r["metrics.output_tps"] for _, r in winners.iterrows()}
    assert by_class == {"c2_longform": 50.0, "c1_chat": 99.0}
    assert len(superseded) == 1


def test_cells_differing_only_in_concurrency_are_not_merged():
    """c=1 and c=8 are separate measurements of separate questions."""
    df = frame(
        cell("q4_k_m", "c1_chat", 1, 1000, output_tps=50.0),
        cell("q4_k_m", "c1_chat", 8, 1000, output_tps=300.0),
    )
    winners, superseded = select_cells(df)
    assert len(winners) == 2
    assert superseded.empty


def test_cells_differing_only_in_rung_are_not_merged():
    df = frame(
        cell("q4_k_m", "c1_chat", 1, 1000),
        cell("q4_k_s", "c1_chat", 1, 1000),
    )
    winners, superseded = select_cells(df)
    assert len(winners) == 2
    assert superseded.empty


def test_a_half_finished_rerun_leaves_untouched_rungs_alone():
    """The rerun can die partway; that must not lose the rungs it never reached."""
    df = frame(
        cell("q8_0", "c1_chat", 8, 1000, output_tps=10.0, prefill=0.02),
        cell("q8_0", "c1_chat", 8, 2000, output_tps=88.0, prefill=0.99),
        cell("iq2_m", "c1_chat", 8, 1000, output_tps=20.0, prefill=0.02),
    )
    winners, _ = select_cells(df)

    got = {r["params.config_id"]: r["metrics.output_tps"] for _, r in winners.iterrows()}
    assert got == {"q8_0": 88.0, "iq2_m": 20.0}


def test_missing_identifying_columns_raise_rather_than_merge_wrongly():
    df = pd.DataFrame([{"start_time": pd.Timestamp(0), "params.config_id": "q4_k_m"}])
    with pytest.raises(KeyError, match="identifying columns"):
        select_cells(df)


def test_an_empty_frame_survives_the_merge():
    empty = pd.DataFrame()
    winners, superseded = select_cells(empty)
    assert winners.empty and superseded.empty


# --------------------------------------------------------------------------
# validity
# --------------------------------------------------------------------------


def test_reused_prefill_is_a_problem():
    row = pd.Series(cell("q4_k_m", "c3_rag", 8, 1000, prefill=0.12))
    assert any("prefill" in p for p in cell_problems(row))


def test_longform_at_0_97_is_not_a_problem():
    """The floor exists to separate 0.97 from 0.76, and it has to keep 0.97.

    c2_longform reads 0.97 on every rung because consecutive prompts share a
    few tokens of chat template. A floor set at run.py's 0.98 warning level
    would discard the one class that was never damaged.
    """
    row = pd.Series(cell("q4_k_m", "c2_longform", 8, 1000, prefill=0.97))
    assert cell_problems(row) == []
    assert PREFILL_FLOOR < 0.97


def test_failed_requests_are_a_problem():
    row = pd.Series(cell("q4_k_m", "c1_chat", 8, 1000, ok=14, total=16))
    assert any("failed=2" in p for p in cell_problems(row))


def test_generation_that_was_not_length_capped_is_a_problem():
    row = pd.Series(cell("q4_k_m", "c1_chat", 8, 1000, length_capped="false"))
    assert "not length-capped" in cell_problems(row)


def test_a_missing_length_capped_tag_is_not_treated_as_a_pass():
    """Cells predating the tag exist; absence must not read as 'checked, fine'."""
    row = pd.Series(cell("q4_k_m", "c1_chat", 8, 1000, length_capped="true"))
    del row["tags.note.length_capped"]
    assert "not length-capped" not in cell_problems(row)


def test_queued_requests_are_a_problem():
    row = pd.Series(cell("q4_k_m", "c1_chat", 8, 1000, waiting=3.0))
    assert any("queued=3" in p for p in cell_problems(row))


def test_a_clean_cell_has_no_problems():
    assert cell_problems(pd.Series(cell("q4_k_m", "c1_chat", 1, 1000))) == []


def test_nan_metrics_do_not_count_as_failures():
    """A cell that never logged a gauge is not a cell that logged a bad one."""
    row = pd.Series(cell("q4_k_m", "c1_chat", 1, 1000))
    row["metrics.prefill_processed_ratio"] = float("nan")
    row["metrics.requests_waiting_peak"] = float("nan")
    assert cell_problems(row) == []


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def test_the_curve_is_ordered_by_descending_bits_per_weight():
    df = frame(
        cell("iq2_m", "c1_chat", 1, 1000, bpw=3.04),
        cell("bf16", "c1_chat", 1, 1000, bpw=16.01),
        cell("q4_k_m", "c1_chat", 1, 1000, bpw=4.99),
    )
    winners, _ = select_cells(df)
    body = [ln for ln in render(winners) if ln.startswith("  ") and "rung" not in ln]
    order = [ln.split()[0] for ln in body if not ln.strip().startswith("-")]
    assert order == ["bf16", "q4_k_m", "iq2_m"]


def test_invalid_cells_are_hidden_by_default_and_shown_with_all():
    df = frame(
        cell("q4_k_m", "c3_rag", 8, 1000, prefill=0.12),
        cell("q8_0", "c3_rag", 8, 1000, prefill=0.99),
    )
    winners, _ = select_cells(df)

    default = "\n".join(render(winners))
    assert "q8_0" in default and "q4_k_m" not in default

    everything = "\n".join(render(winners, only_valid=False))
    assert "q4_k_m" in everything and "prefill=0.12" in everything

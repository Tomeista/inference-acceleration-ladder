"""The scorer, against the output a degrading model actually produces.

These matter more than they look. The study's claim is "accuracy falls as bits
fall", and a scorer that silently stops matching once the prose degrades would
produce exactly that curve out of nothing. Every case below that looks like an
odd thing to test is a way the extractor could have invented a finding: a
fallback that fires on the article "a", a number regex that reads the year out
of a sentence, a repetition detector that flags every correct short answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ladder.scoring import (
    ItemResult,
    aggregate,
    agreement,
    extract_mc,
    extract_mc10,
    extract_numeric,
    normalize_number,
    repetition_ratio,
    score_records,
    wilson_interval,
)


# --------------------------------------------------------------------------
# multiple choice
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Answer: C", "C"),
        ("Answer: C.", "C"),
        ("**Answer:** B", "B"),
        ("Answer: (D)", "D"),
        ("answer: a", "A"),
        ("The answer is C", "C"),
        # Drifted off the format but still committed. The first "answer" is
        # followed by "The" and fails the letter group; the second matches.
        ("Answer: The answer is C", "C"),
        # No "answer" anywhere: the bare-letter fallback carries it.
        ("The correct option is (B).", "B"),
        ("D", "D"),
    ],
)
def test_a_letter_is_found_however_the_model_phrases_it(text, expected):
    assert extract_mc(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "I am not sure about this one."])
def test_no_letter_means_none_rather_than_a_guess(text):
    """`unparseable_rate` is a measurement; a guess would erase it."""
    assert extract_mc(text) is None


def test_the_fallback_does_not_fire_on_the_english_article():
    """The likeliest way this scorer could invent answers.

    A lowercase `a` appears in nearly every sentence. If the bare-letter
    fallback were case-insensitive, a rung producing fluent refusals would score
    at chance instead of being counted as unparseable.
    """
    assert extract_mc("a lazy dog walked past a quiet house") is None


def test_the_last_commitment_wins():
    """Models reconsider mid-reply; the final answer is the answer."""
    assert extract_mc("Answer: A. Wait, that is wrong. Answer: D") == "D"


# --------------------------------------------------------------------------
# multiple choice, ten options
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Answer: F", "F"),
        ("Answer: I", "I"),
        ("Answer: I.", "I"),
        ("**Answer:** J", "J"),
        ("Answer: (H)", "H"),
        ("answer: g", "G"),
        ("The answer is E", "E"),
        ("Answer: The answer is I", "I"),
        ("The correct option is (J).", "J"),
        ("D", "D"),
        ("Answer: A. Wait, that is wrong. Answer: I", "I"),
    ],
)
def test_ten_option_letters_are_found_however_the_model_phrases_it(text, expected):
    assert extract_mc10(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "I am not sure about this one.",
        "I I I I I I I I I I",
        "I think I would need more information to answer.",
    ],
)
def test_the_pronoun_never_becomes_option_i(text):
    """The way this scorer would most easily invent a finding.

    9% of MMLU-Pro's gold answers are "I". If the bare-letter fallback accepted
    a standalone uppercase "I", a rung degrading into first-person filler would
    score those items correct, and the bump would be indistinguishable from a
    real result -- the exact failure this module exists to prevent, and the
    reason `mc10` is a separate extractor rather than `mc` widened to A-J.
    """
    assert extract_mc10(text) is None


def test_a_letter_that_merely_starts_the_next_word_is_not_an_answer():
    """Both of these yield a letter under a naive widening of the A-D pattern."""
    assert extract_mc10("the answer is a bit unclear") is None
    assert extract_mc10("the answer is I think so") is None


def test_narrowing_the_fallback_does_not_cost_the_letter_outright():
    """Only the unlabelled form loses "I"; the instructed format keeps it."""
    assert extract_mc10("Answer: I") == "I"
    assert extract_mc10("The answer is (I).") == "I"


# --------------------------------------------------------------------------
# numeric
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("She has 3 apples left.\n#### 3", "3"),
        ("#### 1,000", "1000"),
        ("#### $42", "42"),
        ("#### -7", "-7"),
        ("#### 18.00", "18"),
        # No hash marker: fall back to the last number in the reply.
        ("First 5, then 10, so the total is 15.", "15"),
        ("The answer is 72", "72"),
    ],
)
def test_a_number_is_found_however_the_model_phrases_it(text, expected):
    assert extract_numeric(text) == expected


@pytest.mark.parametrize("text", ["", "I cannot solve this.", "no digits here"])
def test_no_number_means_none(text):
    assert extract_numeric(text) is None


def test_the_hash_marker_beats_a_later_stray_number():
    """The instructed format is authoritative when the model honoured it."""
    assert extract_numeric("#### 12\nNote: see problem 99 for a variant.") == "12"


@pytest.mark.parametrize(
    "raw,expected",
    [("1,000", "1000"), ("$5.50", "5.5"), ("18.", "18"), ("-0", "0"), ("007", "007")],
)
def test_normalization_makes_formatting_stop_mattering(raw, expected):
    assert normalize_number(raw) == expected


# --------------------------------------------------------------------------
# degeneracy
# --------------------------------------------------------------------------


def test_a_short_correct_answer_is_not_a_loop():
    """Every MC reply is a few words long. Flagging those would make the
    repetition metric say the anchor is as degenerate as the 2-bit rung."""
    assert repetition_ratio("Answer: C") == 0.0


def test_a_loop_is_caught():
    """How the sub-3-bit rungs fail: the same clause forever."""
    looping = "the total is the total is " * 20
    assert repetition_ratio(looping) > 0.8


def test_ordinary_prose_is_not_flagged():
    prose = (
        "Janet sells the remainder at the farmers market daily. She has sixteen "
        "eggs and eats three for breakfast, then bakes muffins with four more, "
        "leaving nine to sell at two dollars each for a total of eighteen."
    )
    assert repetition_ratio(prose) < 0.1


# --------------------------------------------------------------------------
# uncertainty
# --------------------------------------------------------------------------


def test_the_interval_brackets_the_estimate():
    lo, hi = wilson_interval(180, 250)
    assert lo < 0.72 < hi


def test_the_interval_never_leaves_the_unit_range():
    """The reason this is Wilson and not the normal approximation.

    The bottom rungs are expected near zero, where a normal interval reports a
    negative lower bound and it gets printed next to real numbers.
    """
    for k in (0, 1, 249, 250):
        lo, hi = wilson_interval(k, 250)
        assert 0.0 <= lo <= hi <= 1.0


def test_the_interval_is_wide_enough_to_be_worth_printing():
    """The premise of the whole reporting design: at n=250 accuracy is coarse."""
    lo, hi = wilson_interval(175, 250)
    assert hi - lo > 0.09


# --------------------------------------------------------------------------
# scoring a whole suite
# --------------------------------------------------------------------------


@dataclass
class FakeRecord:
    scenario_id: str
    output_text: str
    finish_reason: str = "stop"
    success: bool = True


KEY = {"mmlu-0000": "A", "mmlu-0001": "B", "mmlu-0002": "C", "mmlu-0003": "D"}


def test_a_suite_is_scored_item_by_item():
    records = [
        FakeRecord("mmlu-0000", "Answer: A"),
        FakeRecord("mmlu-0001", "Answer: D"),
        FakeRecord("mmlu-0002", "no idea"),
        FakeRecord("mmlu-0003", "Answer: D", finish_reason="length"),
    ]
    results = score_records(records, KEY, "mc")
    values = aggregate(results)

    assert values["n_items"] == 4
    assert values["accuracy"] == 0.5  # 0000 and 0003 correct
    assert values["unparseable_rate"] == 0.25
    assert values["truncated_rate"] == 0.25


def test_a_failed_request_is_not_a_wrong_answer():
    """It is a missing measurement; scoring it would punish an HTTP error."""
    records = [
        FakeRecord("mmlu-0000", "Answer: A"),
        FakeRecord("mmlu-0001", "", success=False),
    ]
    results = score_records(records, KEY, "mc")
    assert len(results) == 1


def test_the_two_accuracies_separate_wrong_from_silent():
    """The gap between them is the finding, so it has to be computed right.

    Three items answered, one right; one item unparseable. Over everything that
    is 0.25; over what was actually scoreable it is 0.33. A rung that has
    stopped answering shows up as a widening gap rather than as a slow decline.
    """
    key = {f"s{i}": "A" for i in range(4)}
    records = [
        FakeRecord("s0", "Answer: A"),
        FakeRecord("s1", "Answer: B"),
        FakeRecord("s2", "Answer: C"),
        FakeRecord("s3", "hmm"),
    ]
    values = aggregate(score_records(records, key, "mc"))
    assert values["accuracy"] == 0.25
    assert values["accuracy_parsed"] == pytest.approx(1 / 3)


def test_an_item_measured_but_absent_from_the_key_is_an_error():
    """Silent misalignment between prompt file and key would score noise."""
    with pytest.raises(KeyError, match="disagree"):
        score_records([FakeRecord("mmlu-9999", "Answer: A")], KEY, "mc")


def test_an_unknown_scorer_is_refused():
    with pytest.raises(KeyError, match="unknown scorer"):
        score_records([], KEY, "vibes")


def test_an_empty_suite_reports_nothing_rather_than_zero():
    """A zero accuracy reads as "measured, and it was terrible"."""
    assert aggregate([]) == {}


# --------------------------------------------------------------------------
# agreement with the reference rung
# --------------------------------------------------------------------------


def _result(scenario_id: str, extracted: str | None) -> ItemResult:
    return ItemResult(
        scenario_id=scenario_id,
        expected="A",
        extracted=extracted,
        correct=extracted == "A",
        finish_reason="stop",
        truncated=False,
        repetition=0.0,
        text="",
    )


def test_agreement_is_paired_per_item():
    """The metric that catches a rung churning answers at flat accuracy."""
    results = [_result("s0", "A"), _result("s1", "B"), _result("s2", "C")]
    reference = {"s0": "A", "s1": "C", "s2": "C"}
    assert agreement(results, reference)["agreement_with_reference"] == pytest.approx(2 / 3)


def test_two_unparseable_replies_agree():
    """Same behaviour. Counting it as disagreement would make a rung's score
    depend on how the reference happened to fail."""
    assert agreement([_result("s0", None)], {"s0": None})["agreement_with_reference"] == 1.0


def test_no_reference_yields_no_metric_rather_than_zero():
    """The repo-wide "absent rather than zero" convention.

    0.0 here would read as "this rung agreed with BF16 on nothing", which is a
    claim, where the truth is that BF16 has not been run yet.
    """
    assert agreement([_result("s0", "A")], {}) == {}

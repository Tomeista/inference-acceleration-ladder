"""Turning generated text into a score, and saying how much to trust it.

Pure functions: strings in, numbers out. No server, no config, no MLflow. That
separation is deliberate rather than tidy-minded -- extraction is where this
feature can most easily manufacture its own finding. A regex that quietly stops
matching once a rung's prose degrades would produce a beautiful accuracy cliff
that is entirely an artifact of the scorer, and it would be indistinguishable
from the result the study is looking for. So the extractors are tested here,
against the output low-bit rungs actually emit, before anything reaches a GPU.

Three conventions carry that caution into the numbers:

  * An extractor returns None rather than guessing. "No answer found" is a
    measurement -- `unparseable_rate` -- and the whole point is that
    instruction-following usually breaks before accuracy does.
  * Truncation is counted, not scored as wrong. A reply cut off at max_tokens is
    unmeasured; folding it into the wrong pile would credit the scorer with
    knowing something it does not.
  * Accuracy is reported twice: over every item, and over only the items that
    were actually scoreable. When those two diverge, the rung is failing to
    answer rather than answering wrongly, and that is a different finding.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable

# How much generated text to keep per item in the artifact. Enough to see why an
# extraction failed -- the whole reason the raw text is kept at all -- without
# turning a 250-item run into a megabyte of JSONL.
TEXT_KEPT = 600


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

# The instructed format is "Answer: C". This also catches the ways a model
# drifts off it while still committing to a letter: "**Answer:** C",
# "Answer: (C)", "the answer is C", "Answer: The answer is C". findall plus
# last-match is what handles the last of those -- the first "answer" is followed
# by "The", which fails the letter group, and the engine moves on to the second.
_MC_LABELLED = re.compile(
    r"answer\s*(?:is\s*)?[:\-]?\s*\**\s*\(?\s*([A-Da-d])\s*[\)\.\:,]?",
    re.IGNORECASE,
)

# Fallback: a bare letter standing on its own. Deliberately uppercase-only,
# because a lowercase `a` is the English article and would match in almost any
# sentence -- the single most likely way this fallback could invent answers.
_MC_BARE = re.compile(r"(?<![A-Za-z])([A-D])(?![A-Za-z])")

# MMLU-Pro runs to ten options, and the A-D extractor above cannot simply be
# widened to A-J. Two of the added letters are ordinary English words in
# uppercase -- "I" is the first-person pronoun, and a bare "A" opens sentences --
# so the loose fallback that is safe over four letters starts inventing answers
# over ten. That matters here more than it looks: 9% of MMLU-Pro's gold answers
# ARE "I", so a rung whose prose degrades into "I I I ..." would score those
# items correct, and the resulting bump would be indistinguishable from a real
# result.
#
# Two changes, both narrowing:
#
#   the labelled form  a letter that is merely the first character of the next
#                      word no longer counts. "the answer is a bit unclear" and
#                      "the answer is I think so" match nothing rather than
#                      yielding A and I.
#   the bare fallback  drops "I" entirely. A reply of exactly "I" is scored
#                      unparseable rather than as a commitment to option I --
#                      the conservative direction, since `unparseable_rate` is a
#                      measurement and a wrong answer is not. "Answer: I" is
#                      unaffected; only the unlabelled form loses that letter.
_MC10_LABELLED = re.compile(
    r"answer\s*(?:is\s*)?[:\-]?\s*\**\s*\(?\s*([A-Ja-j])(?!\s*[A-Za-z])\s*[\)\.\:,]?",
    re.IGNORECASE,
)

_MC10_BARE = re.compile(r"(?<![A-Za-z])([A-HJ])(?![A-Za-z])")

# GSM8K's own convention, and what the prompt asks for.
_GSM_HASH = re.compile(r"####\s*(-?\$?[\d,]*\.?\d+)")

# Fallback: the last number anywhere in the reply. Reasonable for arithmetic --
# a worked solution ends on its result -- and wrong often enough that
# `unparseable_rate` and the kept raw text both matter.
_NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def extract_mc(text: str) -> str | None:
    """The letter this reply committed to, or None if it committed to none."""
    if not text:
        return None
    matches = _MC_LABELLED.findall(text)
    if matches:
        return matches[-1].upper()
    bare = _MC_BARE.findall(text)
    if bare:
        return bare[-1]
    return None


def extract_mc10(text: str) -> str | None:
    """The letter this reply committed to, over A-J, or None.

    Same shape as `extract_mc`, deliberately not the same function: the two
    suites have different letter ranges and therefore different false-positive
    surfaces, and sharing one extractor would mean widening MMLU's to A-J and
    changing what the four-option suite scores.
    """
    if not text:
        return None
    matches = _MC10_LABELLED.findall(text)
    if matches:
        return matches[-1].upper()
    bare = _MC10_BARE.findall(text)
    if bare:
        return bare[-1]
    return None


def normalize_number(raw: str) -> str:
    """Canonical form, so "$1,000.00" and "1000" are the same answer.

    String rather than float on purpose: the key is compared for equality, and
    round-tripping through a float would make 0.1 + 0.2 a scoring question.
    """
    cleaned = raw.strip().strip(".,").replace(",", "").replace("$", "").replace(" ", "")
    if not cleaned or cleaned in ("-", "."):
        return ""
    # Trailing zeros after a decimal point are formatting, not precision: the
    # gold answers are integers, and "18.00" is the same answer as "18".
    if "." in cleaned:
        cleaned = cleaned.rstrip("0").rstrip(".")
    if cleaned in ("", "-"):
        return "0"
    # "-0" and "0" are the same number and would otherwise score as a miss.
    return "0" if cleaned in ("-0", "0") else cleaned


def extract_numeric(text: str) -> str | None:
    """The number this reply committed to, normalized, or None."""
    if not text:
        return None
    hashed = _GSM_HASH.findall(text)
    if hashed:
        value = normalize_number(hashed[-1])
        return value or None
    numbers = _NUMBER.findall(text)
    if numbers:
        value = normalize_number(numbers[-1])
        return value or None
    return None


EXTRACTORS: dict[str, Callable[[str], str | None]] = {
    "mc": extract_mc,
    "mc10": extract_mc10,
    "numeric": extract_numeric,
}


# --------------------------------------------------------------------------
# degeneracy and uncertainty
# --------------------------------------------------------------------------


def repetition_ratio(text: str, n: int = 4) -> float:
    """How much of this reply is the same n-gram over again.

    0.0 is text that never repeats itself; values approaching 1.0 are a model
    stuck in a loop, which is how the sub-3-bit rungs fail. Reported as its own
    metric because "produced 400 tokens of the same clause" and "answered
    incorrectly" are different failures that a single accuracy number merges.
    """
    words = text.split()
    if len(words) < n * 2:
        # Too short for repetition to mean anything; a two-word reply is not a
        # loop, and scoring it as one would flag every correct MC answer.
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for k successes in n trials.

    Wilson rather than the normal approximation, because the low rungs are
    expected to sit near 0 and the anchor near its ceiling, and the normal
    interval misbehaves at both ends -- it happily reports a lower bound below
    zero, which would be printed next to a real number as though it were one.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# --------------------------------------------------------------------------
# scoring one suite
# --------------------------------------------------------------------------


@dataclass
class ItemResult:
    """One benchmark item, scored. Written per-item to the run's artifact."""

    scenario_id: str
    expected: str
    extracted: str | None
    correct: bool
    finish_reason: str | None
    truncated: bool
    repetition: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "expected": self.expected,
            "extracted": self.extracted,
            "correct": self.correct,
            "finish_reason": self.finish_reason,
            "truncated": self.truncated,
            "repetition": round(self.repetition, 4),
            "text": self.text[:TEXT_KEPT],
        }


def score_records(
    records: Iterable[Any], answers: dict[str, str], scorer: str
) -> list[ItemResult]:
    """Score the successful records of one suite against its answer key.

    `records` are `harness.client.RequestRecord`s, which is why this takes an
    iterable of anything with the right attributes rather than importing the
    class: it keeps this module free of the vendored harness and therefore
    testable with two lines of fake.
    """
    if scorer not in EXTRACTORS:
        raise KeyError(f"unknown scorer {scorer!r}; known: {sorted(EXTRACTORS)}")
    extract = EXTRACTORS[scorer]

    results: list[ItemResult] = []
    for record in records:
        if not record.success:
            continue
        expected = answers.get(record.scenario_id)
        if expected is None:
            raise KeyError(
                f"{record.scenario_id} was measured but is not in the answer key; "
                f"the prompt file and the key file disagree"
            )
        text = record.output_text or ""
        extracted = extract(text)
        results.append(
            ItemResult(
                scenario_id=record.scenario_id,
                expected=expected,
                extracted=extracted,
                correct=extracted is not None and extracted == expected,
                finish_reason=record.finish_reason,
                truncated=record.finish_reason == "length",
                repetition=repetition_ratio(text),
                text=text,
            )
        )
    return results


def aggregate(results: list[ItemResult]) -> dict[str, float]:
    """The metric set for one (rung, suite) cell.

    Two accuracies, on purpose. `accuracy` divides by every item and is the
    headline -- a rung that cannot produce a parseable answer has failed the
    task, and hiding that behind a filter would flatter the bottom of the
    ladder. `accuracy_parsed` divides by the items that were actually
    scoreable, so the gap between the two says whether a rung is answering
    wrongly or has stopped answering at all.
    """
    n = len(results)
    if not n:
        return {}

    correct = sum(1 for r in results if r.correct)
    truncated = sum(1 for r in results if r.truncated)
    unparseable = sum(1 for r in results if r.extracted is None)
    scoreable = [r for r in results if r.extracted is not None and not r.truncated]

    lo, hi = wilson_interval(correct, n)
    out = {
        "n_items": float(n),
        "accuracy": correct / n,
        "accuracy_ci_lo": lo,
        "accuracy_ci_hi": hi,
        "unparseable_rate": unparseable / n,
        "truncated_rate": truncated / n,
        "repetition_ratio": sum(r.repetition for r in results) / n,
    }
    if scoreable:
        out["accuracy_parsed"] = sum(1 for r in scoreable if r.correct) / len(scoreable)
        out["n_scoreable"] = float(len(scoreable))
    return out


def agreement(
    results: list[ItemResult], reference: dict[str, str | None]
) -> dict[str, float]:
    """How often this rung gave the same answer as the reference rung.

    The sensitive metric, and the reason the reference run is required to go
    first. Accuracy on 250 items has a +/-6pp interval, so it can sit flat while
    a rung quietly changes a third of its answers; this catches that, because it
    is paired per item rather than compared in aggregate.

    Two unparseable replies count as agreeing. They are the same *behaviour*,
    and calling that a disagreement would make a rung's agreement score depend
    on how the reference happened to fail.

    Returns {} when there is no reference, rather than 0.0. Same convention as
    `dialect.prefill_reuse_check` and `metrics.preemption_delta`: a metric that
    was never computed must not read as one that came out at zero.
    """
    shared = [r for r in results if r.scenario_id in reference]
    if not shared:
        return {}
    same = sum(1 for r in shared if r.extracted == reference[r.scenario_id])
    return {
        "agreement_with_reference": same / len(shared),
        "agreement_n": float(len(shared)),
    }

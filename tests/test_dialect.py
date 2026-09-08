"""The llama.cpp dialect, against the mock server.

Two things are being checked. First, that bench's measurement code works
unmodified once it is told what llama.cpp calls things -- if it does not, the
reuse this study is built on is not real. Second, and more important, that the
metrics llama.cpp *lacks* come back absent rather than as zeros: a preemption
count of 0.0 says "checked, none happened", and publishing that for an engine
that cannot preempt would be a fabricated validity check.
"""

from __future__ import annotations

import httpx
import pytest

from ladder.harness.client import run_load
from ladder.harness.metrics import GaugeSampler, parse_prometheus, preemption_delta, scrape, spec_decode_metrics
from ladder.harness.scenarios import Scenario, Turn

from ladder.dialect import LLAMACPP, prefill_reuse_check, throughput_cross_check


def _scenarios(n: int, max_tokens: int = 24) -> list[Scenario]:
    return [
        Scenario(
            scenario_id=f"l-{i}",
            class_id="c1_chat",
            turns=[
                Turn(
                    messages=[{"role": "user", "content": f"question number {i}"}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                    extra_body={"ignore_eos": True},
                )
            ],
        )
        for i in range(n)
    ]


# --------------------------------------------------------------------------
# bench's code, driven by the llama.cpp dialect
# --------------------------------------------------------------------------


def test_scrape_finds_llamacpp_counters(mock_llama):
    scraped = scrape(mock_llama, dialect=LLAMACPP)
    assert "llamacpp:prompt_tokens_total" in scraped
    assert "llamacpp:tokens_predicted_total" in scraped


async def test_gauge_sampler_captures_load_under_the_new_names(mock_llama):
    """Same sampler, same reported metric names, different engine underneath."""
    async with GaugeSampler(mock_llama, interval=0.02, dialect=LLAMACPP) as sampler:
        await run_load(
            _scenarios(16),
            base_url=mock_llama,
            model="qwen3-8b",
            concurrency=4,
            num_requests=16,
        )

    peaks = sampler.metrics()
    assert peaks["requests_running_peak"] > 1, "sampler never observed concurrent load"
    assert peaks["kv_cache_usage_peak"] > 0

    # Reported under bench's names, so one MLflow query reads both studies.
    assert set(peaks) <= {"kv_cache_usage_peak", "requests_running_peak", "requests_waiting_peak"}

    after = parse_prometheus(httpx.get(mock_llama + "/metrics").text, LLAMACPP.gauges)
    assert after["llamacpp:requests_processing"] == 0


async def test_sampler_still_reports_nothing_rather_than_a_zero_peak(mock_llama):
    """The convention has to survive the dialect swap, not just the vLLM path."""
    async with GaugeSampler(mock_llama, interval=30.0, dialect=LLAMACPP) as sampler:
        await run_load(
            _scenarios(1, max_tokens=2),
            base_url=mock_llama,
            model="qwen3-8b",
            concurrency=1,
            num_requests=1,
        )
    assert not sampler.observed_load
    assert sampler.metrics() == {}


# --------------------------------------------------------------------------
# what llama.cpp does not have
# --------------------------------------------------------------------------


def test_preemptions_are_absent_not_zero(mock_llama):
    """llama.cpp defers rather than evicting, so there is nothing to report.

    A 0.0 here would read months later as "we checked, the cache never
    overflowed", which is a claim this engine cannot make.
    """
    before = scrape(mock_llama, dialect=LLAMACPP)
    after = scrape(mock_llama, dialect=LLAMACPP)
    assert preemption_delta(before, after, dialect=LLAMACPP) == {}


def test_speculative_metrics_are_absent_not_zero(mock_llama):
    """No draft model in this study; a 0.0 acceptance rate would read as rejection."""
    before = scrape(mock_llama, dialect=LLAMACPP)
    after = scrape(mock_llama, dialect=LLAMACPP)
    assert spec_decode_metrics(before, after, dialect=LLAMACPP) == {}


def test_queue_depth_is_the_pressure_signal_instead(mock_llama):
    """The gauge that replaces preemptions: requests waiting for a slot."""
    assert "llamacpp:requests_deferred" in LLAMACPP.gauges
    assert LLAMACPP.gauge_names["llamacpp:requests_deferred"] == "requests_waiting_peak"


# --------------------------------------------------------------------------
# the prefill-reuse check
# --------------------------------------------------------------------------


async def test_prefill_check_passes_when_the_server_prefills_everything(mock_llama, monkeypatch):
    monkeypatch.setenv("MOCK_PREFIX_REUSE", "0")
    before = scrape(mock_llama, dialect=LLAMACPP)
    result = await run_load(
        _scenarios(8), base_url=mock_llama, model="qwen3-8b", concurrency=1, num_requests=8
    )
    after = scrape(mock_llama, dialect=LLAMACPP)

    sent = sum(r.prompt_tokens or 0 for r in result.successful)
    check = prefill_reuse_check(before, after, sent)
    assert check["prefill_reuse_free"] is True
    assert check["prefill_processed_ratio"] == pytest.approx(1.0, abs=0.05)


async def test_prefill_check_catches_a_server_skipping_half_of_every_prefill(
    mock_llama, monkeypatch
):
    """The confound the check exists for.

    A slot that still holds the previous prompt skips its prefill, so TTFT is
    faster than the rung can deliver cold. Whichever rung happens to reuse more
    then looks quantization-faster, which is a result about slot scheduling
    wearing the costume of a result about bit width.
    """
    monkeypatch.setenv("MOCK_PREFIX_REUSE", "0.5")
    before = scrape(mock_llama, dialect=LLAMACPP)
    result = await run_load(
        _scenarios(8), base_url=mock_llama, model="qwen3-8b", concurrency=1, num_requests=8
    )
    after = scrape(mock_llama, dialect=LLAMACPP)

    sent = sum(r.prompt_tokens or 0 for r in result.successful)
    check = prefill_reuse_check(before, after, sent)
    assert check["prefill_reuse_free"] is False
    assert check["prefill_processed_ratio"] == pytest.approx(0.5, abs=0.05)


def test_prefill_check_is_absent_without_the_counter():
    """No counter means no check, not a passing check."""
    assert prefill_reuse_check({}, {}, 1000) == {}
    assert prefill_reuse_check({}, {"llamacpp:prompt_tokens_total": 5.0}, 0) == {}


async def test_server_side_token_counts_are_logged_alongside_the_client(mock_llama, monkeypatch):
    monkeypatch.setenv("MOCK_PREFIX_REUSE", "0")
    before = scrape(mock_llama, dialect=LLAMACPP)
    await run_load(
        _scenarios(4, max_tokens=12),
        base_url=mock_llama,
        model="qwen3-8b",
        concurrency=1,
        num_requests=4,
    )
    after = scrape(mock_llama, dialect=LLAMACPP)

    cross = throughput_cross_check(before, after)
    assert cross["server_generation_tokens"] == pytest.approx(48.0)

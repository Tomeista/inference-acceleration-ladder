"""llama.cpp's Prometheus vocabulary, plus the one validity check bench lacks.

Everything here is a name table and twenty lines of arithmetic. The percentile
math, the counter differencing, the peak-sampling and the "absent rather than
zero" convention all come from `bench.metrics` unchanged; this module only says
what llama.cpp calls things.

Metric names have moved between llama.cpp releases exactly as they have between
vLLM releases, which is why `bench`'s RUNBOOK tells you to check them against a
live server before trusting a sweep. Same instruction here, and `ladder.run`
prints which of these it actually found so the check is hard to skip.
"""

from __future__ import annotations

from typing import Any

from ladder.harness.metrics import MetricsDialect

# llama-server exposes these only when started with `--metrics`.
#
# Deliberately absent, and correct to be absent:
#
#   preemptions -- llama.cpp does not evict a running sequence. When every slot
#     is busy it defers the request, so the pressure signal is queue depth
#     (`requests_deferred`) rather than a preemption counter. Leaving the field
#     empty makes `bench.metrics.preemption_delta` return {} instead of a 0.0
#     that would read as "checked, and there were none".
#
#   spec_* -- no speculative decoding in this study. The GGUF ladder is a
#     single-axis bits-per-weight sweep; speculation stays in `bench/`, where
#     the acceptance counters exist to explain it.
LLAMACPP = MetricsDialect(
    name="llama.cpp",
    counters=(
        "llamacpp:prompt_tokens_total",
        "llamacpp:tokens_predicted_total",
        "llamacpp:prompt_seconds_total",
        "llamacpp:tokens_predicted_seconds_total",
        "llamacpp:n_decode_total",
    ),
    gauges=(
        "llamacpp:kv_cache_usage_ratio",
        "llamacpp:requests_processing",
        "llamacpp:requests_deferred",
    ),
    # Reported under bench's names so one MLflow query reads both studies. The
    # underlying quantities match: a ratio of cache in use, a count of requests
    # being decoded, a count waiting for a slot.
    gauge_names={
        "llamacpp:kv_cache_usage_ratio": "kv_cache_usage_peak",
        "llamacpp:requests_processing": "requests_running_peak",
        "llamacpp:requests_deferred": "requests_waiting_peak",
    },
    running="llamacpp:requests_processing",
)

PROMPT_TOKENS = "llamacpp:prompt_tokens_total"
GENERATION_TOKENS = "llamacpp:tokens_predicted_total"


def prefill_reuse_check(
    before: dict[str, float],
    after: dict[str, float],
    client_prompt_tokens: int,
    tolerance: float = 0.02,
) -> dict[str, Any]:
    """Did the server actually prefill every prompt token the run sent it?

    This is the ladder's equivalent of bench's preemption check: a validity
    guard, not a result.

    `llamacpp:prompt_tokens_total` counts tokens the server *processed*, and a
    slot skips whatever prefix it still holds from its previous request. The
    client, meanwhile, reports what each prompt *contained*. When the server's
    number falls short of the client's, prefill was reused, and every TTFT in
    the cell is faster than the config can actually deliver on a cold prompt.

    That matters most for c3_rag, whose 4000-token prompts share a system
    preamble and are built from the same corpus, and it matters unevenly across
    rungs: whichever rung happens to reuse more looks quantization-faster. It is
    the one confound `--cache-reuse 0` does not fully close, because slot-level
    reuse of a repeated prompt is not what that flag governs.

    Returns {} when the counter is unavailable, rather than claiming a clean
    check that never ran.
    """
    if PROMPT_TOKENS not in after or not client_prompt_tokens:
        return {}

    processed = after.get(PROMPT_TOKENS, 0.0) - before.get(PROMPT_TOKENS, 0.0)
    ratio = processed / client_prompt_tokens
    return {
        "prefill_tokens_processed": processed,
        "prefill_tokens_sent": float(client_prompt_tokens),
        "prefill_processed_ratio": ratio,
        # Slightly over 1.0 is normal: the server counts a BOS token and any
        # template the client's usage block does not. Materially under 1.0 is
        # the problem.
        "prefill_reuse_free": ratio >= 1.0 - tolerance,
    }


def throughput_cross_check(
    before: dict[str, float], after: dict[str, float]
) -> dict[str, float]:
    """The server's own token counts for the window, next to the client's.

    bench treats the server as the authority on token counts and stops there.
    Here the two are logged side by side, because llama.cpp's OpenAI-compatible
    usage block is a thinner shim than vLLM's and a disagreement between them is
    worth seeing before it becomes a result.
    """
    out: dict[str, float] = {}
    for key, name in (
        (PROMPT_TOKENS, "server_prompt_tokens"),
        (GENERATION_TOKENS, "server_generation_tokens"),
    ):
        if key in after:
            out[name] = after.get(key, 0.0) - before.get(key, 0.0)
    return out

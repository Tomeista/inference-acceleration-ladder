"""Aggregation of per-request records, and the /metrics scrape.

Two things live here that are easy to get subtly wrong and are therefore
written down rather than inlined: the percentile definition, and the fact that
the speculative-decoding counters are cumulative, so a run has to difference
them across its own window instead of reading them once.

Everything above the "Prometheus scraping" divider is engine-agnostic: it works
off the per-request records the client returns and knows nothing about what
served them. Below the divider, engine differences are confined to a
`MetricsDialect`, which is how the sibling `ladder/` study measures llama.cpp
with this same code.
"""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

import httpx

from ladder.harness.client import LoadResult, RequestRecord

PERCENTILES = (50, 90, 95, 99)


def percentile(values: Iterable[float], p: float) -> float | None:
    """Linear-interpolation percentile, matching numpy's default method.

    Spelled out so that a reported p95 means the same thing here as in
    `vllm bench serve`, which uses numpy.
    """
    data = sorted(v for v in values if v is not None and not math.isnan(v))
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    rank = (p / 100.0) * (len(data) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return data[int(rank)]
    return data[low] + (data[high] - data[low]) * (rank - low)


def _spread(values: list[float], prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for p in PERCENTILES:
        v = percentile(values, p)
        if v is not None:
            out[f"{prefix}_p{p}"] = v
    if values:
        out[f"{prefix}_mean"] = sum(values) / len(values)
    return out


@dataclass
class RunMetrics:
    values: dict[str, float]
    notes: dict[str, Any]

    def merged(self) -> dict[str, float]:
        return dict(self.values)


def aggregate(result: LoadResult) -> RunMetrics:
    """Turn per-request records into the metric set logged to MLflow.

    Throughput is computed over the wall-clock window of the whole load phase,
    not as a sum of per-request rates, because the latter would overstate a run
    whose requests did not overlap perfectly.
    """
    ok = result.successful
    failed = len(result.records) - len(ok)

    ttfts = [r.ttft for r in ok if r.ttft is not None]
    tpots = [r.tpot for r in ok if r.tpot is not None]
    e2es = [r.latency for r in ok]
    itls: list[float] = [x for r in ok for x in r.itls]

    completion_tokens = sum(r.completion_tokens or 0 for r in ok)
    prompt_tokens = sum(r.prompt_tokens or 0 for r in ok)
    duration = result.duration or float("nan")

    values: dict[str, float] = {
        "requests_total": float(len(result.records)),
        "requests_ok": float(len(ok)),
        "requests_failed": float(failed),
        "duration_s": duration,
        "output_tokens_total": float(completion_tokens),
        "prompt_tokens_total": float(prompt_tokens),
    }
    values.update(_spread([t for t in ttfts], "ttft_s"))
    values.update(_spread([t for t in tpots], "tpot_s"))
    values.update(_spread(e2es, "e2e_s"))
    values.update(_spread(itls, "itl_s"))

    if duration and not math.isnan(duration) and duration > 0:
        values["output_tps"] = completion_tokens / duration
        values["total_tps"] = (completion_tokens + prompt_tokens) / duration
        values["requests_per_s"] = len(ok) / duration

    notes: dict[str, Any] = {}
    finish_reasons = {r.finish_reason for r in ok if r.finish_reason}
    notes["finish_reasons"] = sorted(finish_reasons)
    # With ignore_eos set, every request should terminate on length. Anything
    # else means generation stopped early and the output-length control that the
    # whole comparison rests on was not actually in force.
    notes["length_capped"] = finish_reasons == {"length"}

    if failed:
        first = next(r.error for r in result.records if not r.success)
        notes["first_error"] = first

    return RunMetrics(values=values, notes=notes)


def check_prompt_lengths(
    records: list[RequestRecord], expected: int, tolerance: int = 0
) -> dict[str, Any]:
    """Confirm the server saw the prompt length the class promised.

    A mismatch usually means the served chat template differs from the one used
    to build the prompt set, which would quietly invalidate cross-config
    comparisons.
    """
    observed = [r.prompt_tokens for r in records if r.prompt_tokens]
    if not observed:
        return {"prompt_length_checked": False}
    lo, hi = min(observed), max(observed)
    return {
        "prompt_length_checked": True,
        "prompt_tokens_expected": expected,
        "prompt_tokens_observed_min": lo,
        "prompt_tokens_observed_max": hi,
        "prompt_length_ok": abs(lo - expected) <= tolerance
        and abs(hi - expected) <= tolerance,
    }


# --------------------------------------------------------------------------
# Prometheus scraping
# --------------------------------------------------------------------------

_SAMPLE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>[^\s]+)$")


@dataclass(frozen=True)
class MetricsDialect:
    """The Prometheus names one serving engine happens to use.

    vLLM and llama.cpp publish the same handful of facts under different names,
    and each omits some the other has: llama.cpp does not preempt, vLLM has no
    notion of a deferred slot. Naming the mapping once keeps everything below
    engine-agnostic and, more importantly, keeps the "absent rather than zero"
    convention in a single place. A metric this engine does not publish yields
    no key at all, never a 0.0, because for acceptance rate and for preemptions
    the two readings mean opposite things.

    An empty string in an optional field means "this engine does not have it".
    """

    name: str
    counters: tuple[str, ...]
    gauges: tuple[str, ...]
    # raw gauge name -> the name it is reported under, so a query written
    # against one engine keeps working against the other.
    gauge_names: dict[str, str]
    # The gauge that is non-zero exactly while requests are in flight. The
    # sampler uses it to tell "the cache was empty" from "nobody looked".
    running: str
    preemptions: str = ""
    spec_drafts: str = ""
    spec_draft_tokens: str = ""
    spec_accepted: str = ""


VLLM = MetricsDialect(
    name="vllm",
    counters=(
        "vllm:spec_decode_num_drafts_total",
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total",
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:num_preemptions_total",
    ),
    gauges=(
        "vllm:gpu_cache_usage_perc",
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
    ),
    gauge_names={
        "vllm:gpu_cache_usage_perc": "kv_cache_usage_peak",
        "vllm:num_requests_running": "requests_running_peak",
        "vllm:num_requests_waiting": "requests_waiting_peak",
    },
    running="vllm:num_requests_running",
    preemptions="vllm:num_preemptions_total",
    spec_drafts="vllm:spec_decode_num_drafts_total",
    spec_draft_tokens="vllm:spec_decode_num_draft_tokens_total",
    spec_accepted="vllm:spec_decode_num_accepted_tokens_total",
)

# Kept as module constants: they read better at call sites, and every existing
# caller predates the dialect.
COUNTERS = VLLM.counters
GAUGES = VLLM.gauges


def parse_prometheus(text: str, names: Iterable[str] = COUNTERS) -> dict[str, float]:
    """Sum each named counter across all of its label sets."""
    wanted = set(names)
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE.match(line)
        if not m or m.group("name") not in wanted:
            continue
        try:
            value = float(m.group("value"))
        except ValueError:
            continue
        if math.isnan(value):
            continue
        out[m.group("name")] = out.get(m.group("name"), 0.0) + value
    return out


def scrape(
    base_url: str, timeout: float = 10.0, dialect: MetricsDialect = VLLM
) -> dict[str, float]:
    """Read the server's Prometheus endpoint. Returns {} if it is unavailable."""
    try:
        response = httpx.get(base_url.rstrip("/") + "/metrics", timeout=timeout)
        response.raise_for_status()
    except Exception:  # noqa: BLE001 - metrics are optional context, not the result
        return {}
    return parse_prometheus(response.text, dialect.counters)


def counter_delta(
    before: dict[str, float], after: dict[str, float], key: str, name: str
) -> dict[str, float]:
    """One cumulative counter differenced across a run window.

    Returns {} when the engine does not publish the counter, which is the same
    convention every other consumer here follows.
    """
    if not key or key not in after:
        return {}
    return {name: after.get(key, 0.0) - before.get(key, 0.0)}


def spec_decode_metrics(
    before: dict[str, float],
    after: dict[str, float],
    dialect: MetricsDialect = VLLM,
) -> dict[str, float]:
    """Acceptance statistics for the run window between two scrapes.

    Returns {} when the server has no speculative counters, so a baseline run
    logs nothing rather than logging zeros that would look like total rejection.

    `mean_accepted_length` follows vLLM's definition: 1 + accepted/drafts, the
    leading 1 being the token the target model produces regardless of whether
    any draft token survived. A value of 1.0 means nothing was accepted; the
    ceiling is 1 + num_speculative_tokens.
    """
    key_drafts = dialect.spec_drafts
    key_draft_tokens = dialect.spec_draft_tokens
    key_accepted = dialect.spec_accepted

    if not key_drafts or key_drafts not in after:
        return {}

    drafts = after.get(key_drafts, 0.0) - before.get(key_drafts, 0.0)
    draft_tokens = after.get(key_draft_tokens, 0.0) - before.get(key_draft_tokens, 0.0)
    accepted = after.get(key_accepted, 0.0) - before.get(key_accepted, 0.0)

    out: dict[str, float] = {
        "spec_drafts": drafts,
        "spec_draft_tokens": draft_tokens,
        "spec_accepted_tokens": accepted,
    }
    if draft_tokens > 0:
        out["spec_acceptance_rate"] = accepted / draft_tokens
    if drafts > 0:
        out["spec_mean_accepted_length"] = 1.0 + accepted / drafts
    return out


class GaugeSampler:
    """Polls the server's gauges during a run and keeps the peak of each.

    Reading a gauge after the load has drained is meaningless: cache usage and
    queue depth are both back to zero by then. The peak during the window is
    what shows whether a config was starved of KV cache.

    This matters most for speculative decoding. `gpu_memory_utilization` is
    pinned across configs so a smaller checkpoint cannot quietly receive a
    larger KV cache, but a draft model draws on that same budget, so the
    speculative config gets a *smaller* cache than its own baseline. Without
    this, the resulting preemptions would look like speculation being slow.

    The GGUF ladder has the same hazard in a different shape: llama.cpp splits
    one context across `--parallel` slots, so peak cache usage is what shows a
    cell running against its per-slot ceiling.

    Sampling is best effort. A server without a /metrics endpoint, or one that
    is too busy to answer, yields no peaks rather than an error.
    """

    def __init__(
        self,
        base_url: str,
        interval: float = 0.5,
        dialect: MetricsDialect = VLLM,
    ) -> None:
        self.url = base_url.rstrip("/") + "/metrics"
        self.interval = interval
        self.dialect = dialect
        self.peaks: dict[str, float] = {}
        self._task: "asyncio.Task[None] | None" = None
        self._stop: "asyncio.Event | None" = None

    async def _poll(self) -> None:
        assert self._stop is not None
        async with httpx.AsyncClient(timeout=5.0) as client:
            while not self._stop.is_set():
                try:
                    response = await client.get(self.url)
                    sample = parse_prometheus(response.text, self.dialect.gauges)
                    for key, value in sample.items():
                        if value > self.peaks.get(key, float("-inf")):
                            self.peaks[key] = value
                except Exception:  # noqa: BLE001 - diagnostics must not fail a run
                    pass
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.interval)
                except asyncio.TimeoutError:
                    pass

    async def __aenter__(self) -> "GaugeSampler":
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._poll())
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._task is not None:
            await self._task

    @property
    def observed_load(self) -> bool:
        """Whether any sample landed while requests were actually in flight.

        A cell shorter than the polling interval can be sampled only before and
        after the load, giving a peak of zero. That is not a measurement of an
        empty cache, it is the absence of a measurement, and logging it as 0.0
        would later be read as the former.
        """
        return self.peaks.get(self.dialect.running, 0.0) > 0.0

    def metrics(self) -> dict[str, float]:
        """Peaks, or nothing at all if the sampler never caught the load."""
        if not self.observed_load:
            return {}
        renamed = self.dialect.gauge_names
        return {renamed[k]: v for k, v in self.peaks.items() if k in renamed}


def preemption_delta(
    before: dict[str, float],
    after: dict[str, float],
    dialect: MetricsDialect = VLLM,
) -> dict[str, float]:
    """Preemptions during the window.

    Non-zero means the server ran out of KV cache and evicted running
    sequences, which inflates tail latency for reasons unrelated to the config
    under test. It is a validity check on the run, not a result.

    Absent on an engine that does not preempt. llama.cpp is one: it defers a
    request until a slot frees rather than evicting a running one, so its
    equivalent pressure signal is the queue-depth gauge instead.
    """
    return counter_delta(before, after, dialect.preemptions, "preemptions")

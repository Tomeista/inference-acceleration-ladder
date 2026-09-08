"""Async load generator against an OpenAI-compatible endpoint.

Why not `vllm bench serve` or GuideLLM: neither expresses the tool-calling class
or the multi-turn class this study needs, neither hands back per-class
speculative acceptance, and both would have to be driven as a subprocess whose
JSON we then reparse into MLflow. Roughly two hundred lines buys full control.

The cost of owning the client is that its numbers have to be trusted, so
`bench.run --validate` prints the same workload's TTFT and TPOT for comparison
against `vllm bench serve`. Run that once per server and record it.

Concurrency model: a fixed pool of N workers pulling from a queue, so exactly N
requests are in flight at any moment. That matches vLLM's `--max-concurrency`
semantics rather than a Poisson arrival process; an arrival-rate mode is the
natural next addition and would slot in as an alternative `_worker`.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from ladder.harness.scenarios import Scenario, cycle_scenarios


@dataclass
class RequestRecord:
    """One chat-completions request, start to finish."""

    scenario_id: str
    class_id: str
    turn_index: int
    start_time: float
    end_time: float = 0.0
    ttft: float | None = None
    # Wall-clock gaps between successive content-bearing chunks. Their mean is
    # TPOT for this request; their distribution is the inter-token latency the
    # user actually perceives.
    itls: list[float] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    output_text: str = ""
    success: bool = False
    error: str | None = None

    @property
    def latency(self) -> float:
        return self.end_time - self.start_time

    @property
    def decode_time(self) -> float | None:
        """Time spent generating after the first token arrived."""
        if self.ttft is None:
            return None
        return self.latency - self.ttft

    @property
    def tpot(self) -> float | None:
        """Mean time per output token, excluding the first.

        Defined as (latency - TTFT) / (completion_tokens - 1). Definitions of
        TPOT differ between tools, which is exactly why it is written down here.
        """
        n = self.completion_tokens
        decode = self.decode_time
        if decode is None or not n or n < 2:
            return None
        return decode / (n - 1)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "scenario_id": self.scenario_id,
            "class_id": self.class_id,
            "turn_index": self.turn_index,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "latency": self.latency,
            "ttft": self.ttft,
            "tpot": self.tpot,
            "n_itls": len(self.itls),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "finish_reason": self.finish_reason,
            "success": self.success,
            "error": self.error,
        }
        return d


@dataclass
class LoadResult:
    records: list[RequestRecord]
    started_at: float
    finished_at: float

    @property
    def duration(self) -> float:
        return self.finished_at - self.started_at

    @property
    def successful(self) -> list[RequestRecord]:
        return [r for r in self.records if r.success]


def _extract_delta(chunk: dict[str, Any]) -> tuple[str, bool]:
    """Return (text, is_content_bearing) for one streamed chunk.

    A tool-call chunk carries no `content` but is still a generated token, so it
    has to count toward TTFT and ITL or the tool-calling class would report a
    TTFT equal to its whole latency.
    """
    choices = chunk.get("choices") or []
    if not choices:
        return "", False
    delta = choices[0].get("delta") or {}
    text = delta.get("content") or delta.get("reasoning_content") or ""
    if text:
        return text, True
    if delta.get("tool_calls"):
        fragments = []
        for call in delta["tool_calls"]:
            fn = call.get("function") or {}
            fragments.append(fn.get("arguments") or fn.get("name") or "")
        return "".join(fragments), True
    return "", False


async def _one_request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    record: RequestRecord,
    collect_output: bool,
) -> RequestRecord:
    pieces: list[str] = []
    last_token_at: float | None = None
    record.start_time = time.perf_counter()

    try:
        async with client.stream("POST", url, json=payload) as response:
            if response.status_code != 200:
                body = (await response.aread()).decode("utf-8", "replace")[:400]
                record.error = f"HTTP {response.status_code}: {body}"
                record.end_time = time.perf_counter()
                return record

            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break

                now = time.perf_counter()
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if usage := chunk.get("usage"):
                    record.prompt_tokens = usage.get("prompt_tokens")
                    record.completion_tokens = usage.get("completion_tokens")

                for choice in chunk.get("choices") or []:
                    if choice.get("finish_reason"):
                        record.finish_reason = choice["finish_reason"]

                text, bearing = _extract_delta(chunk)
                if not bearing:
                    continue
                if record.ttft is None:
                    record.ttft = now - record.start_time
                else:
                    record.itls.append(now - last_token_at)
                last_token_at = now
                if collect_output:
                    pieces.append(text)

        record.end_time = time.perf_counter()
        if record.ttft is None:
            record.error = "stream produced no content"
            return record

        # The server is the authority on token counts, but a server that did not
        # return usage should not silently produce a throughput of zero.
        if record.completion_tokens is None:
            record.completion_tokens = len(record.itls) + 1

        record.output_text = "".join(pieces)
        record.success = True
        return record

    except Exception as exc:  # noqa: BLE001 - a failed request is data, not a crash
        record.end_time = time.perf_counter()
        record.error = f"{type(exc).__name__}: {exc}"
        return record


async def _worker(
    queue: "asyncio.Queue[Scenario | None]",
    client: httpx.AsyncClient,
    url: str,
    model: str,
    out: list[RequestRecord],
    collect_output: bool,
) -> None:
    while True:
        scenario = await queue.get()
        if scenario is None:
            queue.task_done()
            return
        try:
            history: list[RequestRecord] = []
            # Turns run sequentially within a scenario: turn N+1 cannot be built
            # until turn N has returned. Single-turn classes execute this once.
            while (turn := scenario.next_turn(history)) is not None:
                record = RequestRecord(
                    scenario_id=scenario.scenario_id,
                    class_id=scenario.class_id,
                    turn_index=len(history),
                    start_time=time.perf_counter(),
                )
                await _one_request(
                    client, url, turn.to_payload(model), record, collect_output
                )
                history.append(record)
                out.append(record)
                if not record.success:
                    break
        finally:
            queue.task_done()


async def run_load(
    scenarios: list[Scenario],
    *,
    base_url: str,
    model: str,
    concurrency: int,
    num_requests: int,
    api_key: str | None = None,
    request_timeout: float = 600.0,
    collect_output: bool = False,
) -> LoadResult:
    """Drive `num_requests` scenarios at fixed `concurrency`."""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    queue: "asyncio.Queue[Scenario | None]" = asyncio.Queue()
    for scenario in cycle_scenarios(scenarios, num_requests):
        queue.put_nowait(scenario)
    for _ in range(concurrency):
        queue.put_nowait(None)

    records: list[RequestRecord] = []
    limits = httpx.Limits(
        max_connections=concurrency + 8, max_keepalive_connections=concurrency + 8
    )
    timeout = httpx.Timeout(request_timeout, connect=30.0)

    async with httpx.AsyncClient(
        headers=headers, limits=limits, timeout=timeout
    ) as client:
        started_at = time.perf_counter()
        workers = [
            asyncio.create_task(
                _worker(queue, client, url, model, records, collect_output)
            )
            for _ in range(concurrency)
        ]
        await asyncio.gather(*workers)
        finished_at = time.perf_counter()

    return LoadResult(records=records, started_at=started_at, finished_at=finished_at)


async def warmup(
    scenarios: list[Scenario], *, base_url: str, model: str, num_requests: int = 4
) -> None:
    """Discarded requests that pay for CUDA graph capture and kernel autotuning.

    The first requests a fresh vLLM sees are markedly slower than steady state.
    Folding them into the measurement would penalize whichever config happened
    to be measured first.
    """
    await run_load(
        scenarios,
        base_url=base_url,
        model=model,
        concurrency=min(num_requests, 4),
        num_requests=num_requests,
    )

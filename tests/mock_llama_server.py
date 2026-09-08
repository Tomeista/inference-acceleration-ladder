"""A fake llama-server, so the ladder harness can be verified without a GPU.

Same purpose as bench's mock vLLM and deliberately not shared with it: a test
double whose job is to imitate one engine's quirks should not be parameterized
over two engines, or a bug in the shared part hides behind the fake in both
studies at once. This is the one place duplication is the cheaper mistake.

What it imitates that vLLM's mock does not:

  * `llamacpp:`-prefixed Prometheus names, and the absence of preemption and
    speculative counters. Exercises the "absent rather than zero" path in
    `bench.metrics` through a real scrape rather than a hand-built dict.
  * `/props` instead of `/version`, with the slot and context accounting
    `ladder.run` checks its config against.
  * `llamacpp:prompt_tokens_total` counting only *newly processed* prompt
    tokens. With MOCK_PREFIX_REUSE set it under-counts, which is how the
    prefill-reuse check is tested without a real slot cache.

It is a protocol stub, not a simulator. Its latencies are configured, not
modeled, so nothing it produces is a performance result.

    uvicorn mock_llama_server:app --port 8080
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

app = FastAPI()

TTFT_MS = float(os.environ.get("MOCK_TTFT_MS", "40"))
ITL_MS = float(os.environ.get("MOCK_ITL_MS", "8"))
CONTENTION = float(os.environ.get("MOCK_CONTENTION", "0.02"))
MODEL_NAME = os.environ.get("MOCK_MODEL", "qwen3-8b")
N_CTX = int(os.environ.get("MOCK_N_CTX", "65536"))
TOTAL_SLOTS = int(os.environ.get("MOCK_SLOTS", "8"))


def prefix_reuse() -> float:
    """Fraction of prompt tokens the server pretends to have found cached.

    0.0 is an honest server; 0.5 is one silently skipping half of every
    prefill. Read per request, not at import, so one server instance can play
    both roles across a test session.
    """
    return float(os.environ.get("MOCK_PREFIX_REUSE", "0"))


def honours_ignore_eos() -> bool:
    """Whether this build reads ignore_eos out of the OpenAI request body.

    False is the silent failure --preflight exists to catch. Read per request
    for the same reason as above.
    """
    return os.environ.get("MOCK_IGNORE_EOS", "1") == "1"

_state = {
    "in_flight": 0,
    "deferred": 0,
    "prompt_tokens": 0.0,
    "tokens_predicted": 0.0,
    "n_decode": 0.0,
}

WORDS = "the quick brown fox jumps over a lazy dog while parsing tokens".split()


def _estimate_prompt_tokens(messages: list[dict], tools: list | None) -> int:
    """Character-based estimate, or an exact count when a test pins one.

    The mock has no tokenizer, so its estimate can never land on a class's exact
    input budget. MOCK_PROMPT_TOKENS lets a test play a server whose chat
    template agrees with the one the prompt set was built against -- which is
    the condition --preflight checks and could otherwise never be met here.
    """
    pinned = os.environ.get("MOCK_PROMPT_TOKENS")
    if pinned:
        return int(pinned)
    blob = json.dumps(messages) + (json.dumps(tools) if tools else "")
    return max(1, len(blob) // 4)


@app.get("/health")
async def health() -> PlainTextResponse:
    return PlainTextResponse("", status_code=200)


@app.get("/props")
async def props() -> JSONResponse:
    """llama-server's answer to /version, plus the context accounting."""
    return JSONResponse(
        {
            "model_path": f"/models/gguf/{MODEL_NAME}.gguf",
            "total_slots": TOTAL_SLOTS,
            "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}",
            "build_info": "mock-b0000",
            "default_generation_settings": {"n_ctx": N_CTX},
        }
    )


@app.get("/v1/models")
async def models() -> JSONResponse:
    return JSONResponse({"data": [{"id": MODEL_NAME, "object": "model"}]})


@app.get("/metrics")
async def prometheus() -> PlainTextResponse:
    # No preemption counter and no speculative counters, because llama.cpp has
    # neither. Their absence is the point: it drives the {} return in
    # bench.metrics.preemption_delta and spec_decode_metrics.
    lines = [
        "# TYPE llamacpp:prompt_tokens_total counter",
        f'llamacpp:prompt_tokens_total{{model_name="{MODEL_NAME}"}} {_state["prompt_tokens"]}',
        "# TYPE llamacpp:tokens_predicted_total counter",
        f'llamacpp:tokens_predicted_total{{model_name="{MODEL_NAME}"}} {_state["tokens_predicted"]}',
        "# TYPE llamacpp:n_decode_total counter",
        f'llamacpp:n_decode_total{{model_name="{MODEL_NAME}"}} {_state["n_decode"]}',
        "# TYPE llamacpp:prompt_seconds_total counter",
        f'llamacpp:prompt_seconds_total{{model_name="{MODEL_NAME}"}} 0.0',
        "# TYPE llamacpp:tokens_predicted_seconds_total counter",
        f'llamacpp:tokens_predicted_seconds_total{{model_name="{MODEL_NAME}"}} 0.0',
        "# TYPE llamacpp:requests_processing gauge",
        f'llamacpp:requests_processing{{model_name="{MODEL_NAME}"}} {_state["in_flight"]}',
        "# TYPE llamacpp:requests_deferred gauge",
        f'llamacpp:requests_deferred{{model_name="{MODEL_NAME}"}} {_state["deferred"]}',
        "# TYPE llamacpp:kv_cache_usage_ratio gauge",
        f'llamacpp:kv_cache_usage_ratio{{model_name="{MODEL_NAME}"}} '
        f'{min(1.0, _state["in_flight"] / max(1, TOTAL_SLOTS)):.4f}',
    ]
    return PlainTextResponse("\n".join(lines) + "\n")


def _chunk(request_id: str, created: int, model: str, delta: dict, finish: str | None = None) -> str:
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", MODEL_NAME)
    max_tokens = int(body.get("max_tokens", 16))
    messages = body.get("messages", [])
    tools = body.get("tools")
    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
    prompt_tokens = _estimate_prompt_tokens(messages, tools)

    # The parameter is a llama.cpp native riding in the OpenAI body. A build
    # that drops it stops early on its own; that is what --preflight looks for.
    ignore_eos = bool(body.get("ignore_eos")) and honours_ignore_eos()
    n_out = max_tokens if ignore_eos else min(max_tokens, 7)

    if not body.get("stream"):
        return JSONResponse({"error": "this mock only implements streaming"}, status_code=400)

    request_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())

    async def generate():
        if _state["in_flight"] >= TOTAL_SLOTS:
            # llama.cpp defers rather than preempting. Reflected here so the
            # queue-depth warning in ladder.run has something to fire on.
            _state["deferred"] += 1
            while _state["in_flight"] >= TOTAL_SLOTS:
                await asyncio.sleep(0.005)
            _state["deferred"] -= 1

        _state["in_flight"] += 1
        factor = 1.0 + CONTENTION * max(0, _state["in_flight"] - 1)
        try:
            await asyncio.sleep(TTFT_MS / 1000.0 * factor)
            yield _chunk(request_id, created, model, {"role": "assistant", "content": ""})

            for i in range(n_out):
                if i:
                    await asyncio.sleep(ITL_MS / 1000.0 * factor)
                yield _chunk(request_id, created, model, {"content": WORDS[i % len(WORDS)] + " "})

            yield _chunk(
                request_id, created, model, {}, finish="length" if ignore_eos else "stop"
            )

            # Counts tokens *processed*, so a cached prefix is not counted. This
            # is the shortfall prefill_reuse_check measures.
            _state["prompt_tokens"] += prompt_tokens * (1.0 - prefix_reuse())
            _state["tokens_predicted"] += n_out
            _state["n_decode"] += n_out

            if include_usage:
                yield "data: " + json.dumps(
                    {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": n_out,
                            "total_tokens": prompt_tokens + n_out,
                        },
                    }
                ) + "\n\n"

            yield "data: [DONE]\n\n"
        finally:
            _state["in_flight"] -= 1

    return StreamingResponse(generate(), media_type="text/event-stream")

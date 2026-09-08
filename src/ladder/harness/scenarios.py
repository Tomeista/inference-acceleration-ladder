"""Scenario and Turn: the unit of work the load client executes.

A Scenario is a *sequence* of requests rather than a single request. Every MVP
class is the degenerate one-turn case, but building it this way is what keeps
c7 (multi-turn ReAct) from forcing a rewrite of the concurrency logic later: a
multi-turn class overrides `next_turn` to construct turn N+1 from the model's
output in turn N, and the client loop is unchanged.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class Turn:
    """One chat-completions request."""

    messages: list[dict[str, Any]]
    max_tokens: int
    temperature: float
    tools: list[dict[str, Any]] | None = None
    # Non-OpenAI fields passed straight through to vLLM (ignore_eos,
    # chat_template_kwargs, ...). Kept as an open dict so a new vLLM knob does
    # not need a schema change here.
    extra_body: dict[str, Any] = field(default_factory=dict)

    def to_payload(self, model: str, *, stream: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": self.messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": stream,
        }
        if stream:
            # Without this the final chunk carries no usage block and we would
            # have to trust our own token counting instead of the server's.
            payload["stream_options"] = {"include_usage": True}
        if self.tools:
            payload["tools"] = self.tools
        payload.update(self.extra_body)
        return payload


@dataclass
class Scenario:
    """A prompt instance: one or more sequential turns against one session."""

    scenario_id: str
    class_id: str
    turns: list[Turn]
    # Exact chat-templated prompt length of turn 0, measured at build time by
    # the real Qwen3 tokenizer. Recorded so a run can assert that the server
    # saw the length the class promised.
    prompt_tokens_expected: int | None = None

    def next_turn(self, history: list[Any]) -> Turn | None:
        """Next turn given the completed turns so far, or None when finished.

        `history` holds the RequestRecord of each completed turn. Static
        classes ignore it and walk the list; a dynamic class overrides this.
        """
        idx = len(history)
        return self.turns[idx] if idx < len(self.turns) else None

    # -- serialization ----------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scenario":
        turns = [Turn(**t) for t in data["turns"]]
        return cls(
            scenario_id=data["scenario_id"],
            class_id=data["class_id"],
            turns=turns,
            prompt_tokens_expected=data.get("prompt_tokens_expected"),
        )


def write_scenarios(path: Path, scenarios: list[Scenario]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for scenario in scenarios:
            fh.write(scenario.to_json() + "\n")


def read_scenarios(path: Path) -> list[Scenario]:
    if not path.exists():
        raise FileNotFoundError(
            f"no frozen prompt set at {path}. Run: python -m bench.build_prompts"
        )
    with path.open("r", encoding="utf-8") as fh:
        return [Scenario.from_dict(json.loads(line)) for line in fh if line.strip()]


def cycle_scenarios(scenarios: list[Scenario], count: int) -> Iterator[Scenario]:
    """Yield `count` scenarios, repeating the set if it is smaller.

    Repetition is a measurement hazard with prefix caching on, so the sweep
    warns when it has to wrap.
    """
    if not scenarios:
        raise ValueError("empty scenario set")
    for i in range(count):
        yield scenarios[i % len(scenarios)]

"""Loader for config/classes.yaml.

A PromptClass is the frozen definition of one workload shape. It is the unit
that a benchmark result is reported against, so every field here ends up as an
MLflow parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# One level deeper than in bench (src/bench/ vs src/ladder/harness/), so the
# hop to the repo root is parents[3] rather than parents[2]. This is the only
# behavioural edit made to any vendored file; see harness/__init__.py.
_REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = _REPO_ROOT / "config"
PROMPTS_DIR = _REPO_ROOT / "prompts"


@dataclass(frozen=True)
class PromptClass:
    id: str
    name: str
    source: str
    input_tokens: int
    output_tokens: int
    temperature: float
    description: str = ""
    thinking: bool = False
    uses_tools: bool = False
    turns: int = 1
    num_prompts: int = 128
    enabled: bool = True

    @property
    def prompt_file(self) -> Path:
        return PROMPTS_DIR / f"{self.id}.jsonl"

    def as_params(self) -> dict[str, object]:
        """The subset worth recording on an MLflow run."""
        return {
            "class_id": self.id,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "temperature": self.temperature,
            "thinking": self.thinking,
            "uses_tools": self.uses_tools,
            "turns": self.turns,
        }


def load_classes(path: Path | None = None) -> dict[str, PromptClass]:
    """Load every class, enabled or not, keyed by id."""
    path = path or CONFIG_DIR / "classes.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = raw.get("defaults") or {}

    valid = set(PromptClass.__dataclass_fields__)

    classes: dict[str, PromptClass] = {}
    for entry in raw["classes"]:
        merged = {**defaults, **entry}
        unknown = set(merged) - valid
        if unknown:
            raise ValueError(f"class {entry.get('id')!r} has unknown keys: {sorted(unknown)}")
        cls = PromptClass(**merged)
        if cls.id in classes:
            raise ValueError(f"duplicate class id {cls.id!r}")
        classes[cls.id] = cls
    return classes


def select_classes(
    requested: list[str] | None = None, path: Path | None = None
) -> list[PromptClass]:
    """Resolve a CLI --classes selection.

    With no selection, returns the enabled classes in file order. An explicit
    selection may name a disabled class, so that a half-built class can be
    exercised without editing the config.
    """
    classes = load_classes(path)
    if not requested:
        return [c for c in classes.values() if c.enabled]

    missing = [r for r in requested if r not in classes]
    if missing:
        raise KeyError(f"unknown class id(s) {missing}; known: {sorted(classes)}")
    return [classes[r] for r in requested]

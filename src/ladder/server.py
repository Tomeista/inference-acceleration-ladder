"""GGUF ladder configs, llama-server script generation, and the readiness probe.

The sibling `bench/` study measures seven vLLM servers along three axes. This
one measures a single axis, bits per weight, as densely as the format allows:
llama.cpp has working CUDA kernels at roughly eighteen distinct widths between
16.0 and 1.75 bits, where vLLM has three.

Structured as a deliberate mirror of `bench.server`, because the two are read
side by side: same `defaults` + `configs` YAML shape, same "unknown key is an
error" loading, same generated-scripts-in-version-control rule, same refusal to
launch anything. What differs is llama.cpp's context accounting, which is
different enough to be the main thing this module exists to get right.

The context trap
----------------
`--ctx-size` in llama.cpp is the size of the *whole* KV cache, divided evenly
across `--parallel` slots. A request longer than ctx_size/parallel is rejected
or truncated. So the per-sequence budget that `bench` pins as `max_model_len`
is here a derived quantity, and pinning the wrong one of the two silently
changes what every config is allowed to do. `n_ctx_per_slot` is therefore the
pinned field and `--ctx-size` is computed from it.
"""

from __future__ import annotations

import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PACKAGE_ROOT / "config"
SCRIPTS_DIR = PACKAGE_ROOT / "scripts"

# Qwen3-8B parameter count, summed from its config.json rather than quoted from
# the model card. Used only to turn a GGUF file size into bits per weight, the
# x-axis of the whole study, so it wants to be exact and auditable:
#   36 layers x (4 attn projections + 3 MLP projections) + embed + lm_head
QWEN3_8B_PARAMS = 8_190_427_136


@dataclass(frozen=True)
class LadderConfig:
    """One rung: a single GGUF file served by llama-server."""

    id: str
    name: str
    # Where the file came from. Recorded because quantization quality depends on
    # the imatrix calibration set the publisher used, so two repos' Q4_K_M are
    # not the same checkpoint and must not be pooled into one curve.
    repo: str
    gguf_file: str
    quant_type: str
    # Published bits-per-weight for this quant type. Nominal: `ladder.models`
    # measures the real one from the file on disk and that is what gets plotted.
    bpw_nominal: float
    # kquant | iquant | fp. The k/I split is the one axis besides width that
    # changes the answer: I-quants are codebook lookups and trade compute for
    # size, so they can be smaller and slower at once.
    family: str
    # anchor | ladder | unsloth_dynamic. Groups are reported separately; see
    # the README on why the Unsloth rows are not points on the same curve.
    group: str = "ladder"
    served_model_name: str = "qwen3-8b"
    model_dir: str = "./models/gguf"

    # -- pinned serving parameters ----------------------------------------
    host: str = "127.0.0.1"
    port: int = 8080
    n_gpu_layers: int = 99
    # Per-sequence context. The equivalent of bench's max_model_len, and pinned
    # for the same reason: a smaller checkpoint must not quietly be allowed a
    # longer sequence and be credited for it.
    n_ctx_per_slot: int = 8192
    parallel: int = 8
    batch_size: int = 2048
    ubatch_size: int = 512
    # Pinned across every rung. Quantizing the KV cache as well would put a
    # second variable on the x-axis and make the curve unreadable.
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    threads: int = 8
    seed: int = 0
    extra_args: list[str] = field(default_factory=list)
    enabled: bool = True

    @property
    def n_ctx(self) -> int:
        """`--ctx-size`: the whole cache, which llama.cpp splits across slots."""
        return self.n_ctx_per_slot * self.parallel

    @property
    def model_file(self) -> Path:
        return Path(self.model_dir) / self.gguf_file

    @property
    def base_url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{host}:{self.port}"

    def as_params(self) -> dict[str, object]:
        return {
            "config_id": self.id,
            "engine": "llama.cpp",
            "repo": self.repo,
            "gguf_file": self.gguf_file,
            "quant_type": self.quant_type,
            "bpw_nominal": self.bpw_nominal,
            "quant_family": self.family,
            "ladder_group": self.group,
            "n_ctx_per_slot": self.n_ctx_per_slot,
            "n_ctx": self.n_ctx,
            "parallel": self.parallel,
            "n_gpu_layers": self.n_gpu_layers,
            "cache_type_k": self.cache_type_k,
            "cache_type_v": self.cache_type_v,
            "batch_size": self.batch_size,
            "ubatch_size": self.ubatch_size,
            "server_seed": self.seed,
        }


def load_configs(path: Path | None = None) -> dict[str, LadderConfig]:
    path = path or CONFIG_DIR / "ladder.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = raw.get("defaults") or {}
    valid = set(LadderConfig.__dataclass_fields__)

    configs: dict[str, LadderConfig] = {}
    for entry in raw["configs"]:
        merged = {**defaults, **entry}
        unknown = set(merged) - valid
        if unknown:
            raise ValueError(f"config {entry.get('id')!r} has unknown keys: {sorted(unknown)}")
        cfg = LadderConfig(**merged)
        if cfg.id in configs:
            raise ValueError(f"duplicate config id {cfg.id!r}")
        configs[cfg.id] = cfg
    return configs


def serve_command(cfg: LadderConfig) -> list[str]:
    """The llama-server argv for this rung."""
    argv = [
        "llama-server",
        "--model",
        str(cfg.model_file).replace("\\", "/"),
        # /v1/models reports this, which is how a run proves it measured the
        # server it thinks it did.
        "--alias",
        cfg.served_model_name,
        "--host",
        cfg.host,
        "--port",
        str(cfg.port),
        "--n-gpu-layers",
        str(cfg.n_gpu_layers),
        "--ctx-size",
        str(cfg.n_ctx),
        "--parallel",
        str(cfg.parallel),
        "--batch-size",
        str(cfg.batch_size),
        "--ubatch-size",
        str(cfg.ubatch_size),
        "--cache-type-k",
        cfg.cache_type_k,
        "--cache-type-v",
        cfg.cache_type_v,
        "--threads",
        str(cfg.threads),
        "--seed",
        str(cfg.seed),
        # Without this there is no /metrics endpoint and the run loses every
        # validity check it has.
        "--metrics",
        # llama.cpp's answer to bench's --no-enable-prefix-caching. Chunked
        # prefix reuse off, so a later request cannot skip prefill that an
        # earlier one already paid for and be credited with a faster TTFT.
        # Slot-level reuse of an identical prompt is not covered by this flag,
        # which is why `ladder.dialect.prefill_reuse_check` measures the
        # residual directly instead of trusting the flag.
        "--cache-reuse",
        "0",
    ]
    argv += list(cfg.extra_args)
    return argv


def render_serve_script(cfg: LadderConfig) -> str:
    argv = serve_command(cfg)
    pairs: list[str] = []
    rest = argv[1:]
    i = 0
    while i < len(rest):
        if rest[i].startswith("--") and i + 1 < len(rest) and not rest[i + 1].startswith("--"):
            pairs.append(f"{rest[i]} {shlex.quote(rest[i + 1])}")
            i += 2
        else:
            pairs.append(rest[i])
            i += 1
    body = " \\\n    ".join(["llama-server"] + pairs)

    return f"""#!/usr/bin/env bash
# Generated by ladder.server. Do not edit by hand; edit config/ladder.yaml.
#
# Rung   : {cfg.id}  ({cfg.name})
# Quant  : {cfg.quant_type}  ({cfg.family}, ~{cfg.bpw_nominal} bpw nominal)
# Source : {cfg.repo} :: {cfg.gguf_file}
# Group  : {cfg.group}
#
# --ctx-size is {cfg.n_ctx_per_slot} per slot x {cfg.parallel} slots = {cfg.n_ctx}.
# llama.cpp divides one context across --parallel slots, so the per-sequence
# budget is the derived quantity and it is what is pinned across rungs. Raising
# --parallel without raising --ctx-size shortens every sequence and would show
# up as c3_rag failing rather than as a slow config.
set -euo pipefail

{body}
"""


def write_serve_scripts(
    configs: dict[str, LadderConfig], out_dir: Path | None = None
) -> list[Path]:
    out_dir = out_dir or SCRIPTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for cfg in configs.values():
        path = out_dir / f"serve_{cfg.id}.sh"
        path.write_text(render_serve_script(cfg), encoding="utf-8", newline="\n")
        written.append(path)
    return written


# --------------------------------------------------------------------------
# probing
# --------------------------------------------------------------------------


def wait_for_ready(base_url: str, timeout: float = 900.0, interval: float = 3.0) -> None:
    """Block until /health answers, or raise.

    Longer default than bench's: llama.cpp reads the whole GGUF off disk before
    it answers, and the F16 anchor is ~16 GB.
    """
    deadline = time.time() + timeout
    last: str = "no attempt made"
    while time.time() < deadline:
        try:
            response = httpx.get(base_url.rstrip("/") + "/health", timeout=5.0)
            if response.status_code == 200:
                return
            last = f"HTTP {response.status_code}"
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(interval)
    raise TimeoutError(f"server at {base_url} not ready after {timeout:.0f}s ({last})")


def server_info(base_url: str) -> dict[str, Any]:
    """What the running server says about itself.

    llama-server has no /version. The equivalent is /props, which additionally
    reports the context and slot count it actually allocated -- the analogue of
    the `GPU KV cache size` line bench tells you to record off the vLLM startup
    log, and the one number that proves the context pin took effect.

    Every field is optional: /props has gained keys across llama.cpp releases
    and this must degrade to a thinner record rather than to a failed run.
    """
    info: dict[str, Any] = {}
    try:
        models = httpx.get(base_url.rstrip("/") + "/v1/models", timeout=10.0).json()
        info["served_models"] = [m.get("id") for m in models.get("data", [])]
    except Exception as exc:  # noqa: BLE001
        info["served_models_error"] = str(exc)

    try:
        props = httpx.get(base_url.rstrip("/") + "/props", timeout=10.0).json()
    except Exception:  # noqa: BLE001
        return info

    info["model_path"] = props.get("model_path")
    info["total_slots"] = props.get("total_slots")
    info["chat_template_present"] = bool(props.get("chat_template"))
    settings = props.get("default_generation_settings") or {}
    # Recent builds nest this under the slot's own record.
    info["n_ctx_reported"] = settings.get("n_ctx") or props.get("n_ctx")
    for key in ("build_info", "model_n_params", "model_size"):
        if key in props:
            info[key] = props[key]
    return info


def check_context_budget(cfg: LadderConfig, concurrency: int, longest_class: int) -> str | None:
    """Reasons this rung cannot honestly run this cell, or None.

    Two failure modes that both surface as garbage rather than as an error:

    Asking for more concurrency than there are slots does not fail. llama.cpp
    queues the excess, so the cell still returns numbers -- they are just the
    numbers for `parallel` concurrency with a queue in front, not for the
    concurrency the run claims to have measured.

    A prompt longer than the per-slot context is truncated from the left by
    some builds rather than rejected, which turns c3_rag into a shorter class
    that still reports 4000 expected tokens.
    """
    if concurrency > cfg.parallel:
        return (
            f"concurrency {concurrency} exceeds --parallel {cfg.parallel}: the "
            f"excess would queue, and the cell would report queued latency as "
            f"though it were concurrent latency"
        )
    if longest_class > cfg.n_ctx_per_slot:
        return (
            f"class needs {longest_class} tokens but each of {cfg.parallel} slots "
            f"has only {cfg.n_ctx_per_slot} (--ctx-size {cfg.n_ctx}); raise "
            f"n_ctx_per_slot or lower parallel"
        )
    return None

"""The driver, end to end against the mock llama-server.

The load is fake, so none of these assert on speed. What they assert on is the
plumbing that would otherwise only be exercised for the first time on the GPU
box, halfway through a sweep that takes most of a day.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ladder.harness.classes import PROMPTS_DIR, select_classes

from ladder import run as run_mod
from ladder.server import load_configs, server_info


@pytest.fixture(scope="module")
def cfg():
    return load_configs()["q4_k_m"]


@pytest.fixture
def args():
    return run_mod.build_parser().parse_args([])


# --------------------------------------------------------------------------
# the shared prompt set
# --------------------------------------------------------------------------


def test_the_prompt_files_are_the_bytes_the_manifest_recorded():
    """The single most expensive mistake this study could have made.

    The BF16 anchor ties the llama.cpp curve to the vLLM study only if both
    engines were shown byte-identical prompts. These files are a copy of that
    set, so "same directory" is no longer what guarantees it -- the digests
    are. A prompt set that was regenerated, re-tokenized, or line-ending
    mangled in transit would still load and still look plausible; only this
    test would notice.
    """
    manifest = json.loads((PROMPTS_DIR / "manifest.json").read_text(encoding="utf-8"))
    recorded = manifest["classes"]
    assert recorded, "manifest lists no classes"

    for class_id, stats in sorted(recorded.items()):
        path = PROMPTS_DIR / f"{class_id}.jsonl"
        assert path.exists(), f"{class_id} is in the manifest but not on disk"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        assert actual == stats["sha256_16"], (
            f"{class_id}.jsonl is not the frozen set: manifest says "
            f"{stats['sha256_16']}, file is {actual}. Results measured against "
            f"it cannot be compared with the vLLM study."
        )


def test_every_enabled_class_has_its_prompt_file():
    for cls in select_classes():
        assert cls.prompt_file.exists(), f"{cls.id} has no prompt file"


def test_the_manifest_digest_identifies_the_prompt_sets():
    """Recorded on every run, so a result can be traced to the exact prompts."""
    digest = run_mod._manifest_digest()
    assert digest != "missing"
    assert "c1_chat:" in digest


# --------------------------------------------------------------------------
# probing
# --------------------------------------------------------------------------


def test_server_info_reads_props_where_vllm_had_version(mock_llama):
    info = server_info(mock_llama)
    assert info["served_models"] == ["qwen3-8b"]
    assert info["total_slots"] == 8
    assert info["n_ctx_reported"] == 65536
    assert info["build_info"] == "mock-b0000"


def test_server_info_degrades_rather_than_raising():
    """A build with a thinner /props must cost a field, not the run."""
    info = server_info("http://127.0.0.1:1")
    assert "served_models_error" in info
    assert "total_slots" not in info


# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------


async def test_preflight_passes_against_a_conforming_server(mock_llama, cfg, monkeypatch):
    monkeypatch.setenv("MOCK_IGNORE_EOS", "1")
    monkeypatch.setenv("MOCK_PREFIX_REUSE", "0")
    # A server whose chat template agrees with the one c1_chat was built for.
    monkeypatch.setenv("MOCK_PROMPT_TOKENS", "64")
    assert await run_mod.preflight(cfg, mock_llama) == 0


async def test_preflight_catches_a_divergent_chat_template(mock_llama, cfg, monkeypatch):
    """The GGUF carries its own copy of the template, and the two have diverged.

    A prompt set tokenized to exactly 4000 tokens against the HF template is not
    4000 tokens against a different one, so c3_rag would stop being the prefill
    class it claims to be -- silently, since every request still succeeds.
    """
    monkeypatch.setenv("MOCK_IGNORE_EOS", "1")
    monkeypatch.setenv("MOCK_PROMPT_TOKENS", "96")  # c1_chat is built for 64
    assert await run_mod.preflight(cfg, mock_llama) == 1


async def test_preflight_catches_a_build_that_drops_ignore_eos(mock_llama, cfg, monkeypatch):
    """The failure that would otherwise be found after the sweep, in the data.

    Without ignore_eos each rung generates a different number of tokens, so
    every tokens/sec figure in the study is computed over a different amount of
    work. It fails silently: the requests all succeed.
    """
    monkeypatch.setenv("MOCK_IGNORE_EOS", "0")
    monkeypatch.setenv("MOCK_PROMPT_TOKENS", "64")
    assert await run_mod.preflight(cfg, mock_llama) == 1


async def test_preflight_fails_when_there_is_no_metrics_endpoint(cfg, mock_llama, monkeypatch):
    """No /metrics means no validity checks, which is worse than no numbers."""
    monkeypatch.setenv("MOCK_IGNORE_EOS", "1")
    monkeypatch.setenv("MOCK_PROMPT_TOKENS", "64")
    monkeypatch.setattr(run_mod.metrics_mod, "scrape", lambda *a, **k: {})
    assert await run_mod.preflight(cfg, mock_llama) == 1


# --------------------------------------------------------------------------
# a whole cell
# --------------------------------------------------------------------------


async def test_run_cell_produces_the_metric_set(mock_llama, cfg, args, monkeypatch):
    monkeypatch.setenv("MOCK_PREFIX_REUSE", "0")
    args.requests_per_cell = 8
    cls = select_classes(["c1_chat"])[0]

    values, notes, records = await run_mod.run_cell(cls, 2, cfg, mock_llama, args)

    assert values["requests_ok"] == 8
    assert values["ttft_s_p50"] > 0
    assert values["output_tps"] > 0
    assert len(records) == 8

    # llama.cpp has neither, and the cell must not invent them.
    assert "preemptions" not in values
    assert "spec_acceptance_rate" not in values

    # The ladder's own additions.
    assert values["prefill_processed_ratio"] == pytest.approx(1.0, abs=0.05)
    assert values["server_generation_tokens"] > 0
    assert notes["prefill_reuse_free"] is True
    assert notes["length_capped"] is True


async def test_run_cell_flags_reused_prefill_in_its_notes(mock_llama, cfg, args, monkeypatch):
    monkeypatch.setenv("MOCK_PREFIX_REUSE", "0.5")
    args.requests_per_cell = 8
    cls = select_classes(["c1_chat"])[0]

    values, notes, _ = await run_mod.run_cell(cls, 1, cfg, mock_llama, args)
    assert notes["prefill_reuse_free"] is False
    assert values["prefill_processed_ratio"] < 0.6


async def test_a_cell_carries_no_boolean_into_the_metric_set(mock_llama, cfg, args, monkeypatch):
    """MLflow metrics are floats; the booleans belong in tags.

    bench's _clean_metrics would drop them silently, so this is the only place
    the split gets checked.
    """
    monkeypatch.setenv("MOCK_PREFIX_REUSE", "0")
    args.requests_per_cell = 4
    cls = select_classes(["c1_chat"])[0]
    values, _, _ = await run_mod.run_cell(cls, 1, cfg, mock_llama, args)
    assert not any(isinstance(v, bool) for v in values.values())


# --------------------------------------------------------------------------
# generated artefacts
# --------------------------------------------------------------------------


def test_emit_scripts_writes_one_per_rung(tmp_path: Path):
    from ladder.server import write_serve_scripts

    configs = load_configs()
    written = write_serve_scripts(configs, out_dir=tmp_path)
    assert len(written) == len(configs)
    assert (tmp_path / "serve_q4_k_m.sh").exists()

    body = (tmp_path / "serve_ud_iq1_s.sh").read_text(encoding="utf-8")
    assert "--metrics" in body
    assert "UD-IQ1_S" in body


def test_download_script_covers_every_rung():
    from ladder.models import render_download_script

    configs = load_configs()
    script = render_download_script(configs)
    for cfg in configs.values():
        assert cfg.gguf_file in script
        assert cfg.repo in script


def test_measured_bpw_is_computed_from_the_file_on_disk(tmp_path: Path):
    """Nominal bpw is a label. The x-axis is what the bytes actually say."""
    from ladder.models import measured_bpw

    path = tmp_path / "fake.gguf"
    # A tenth of Qwen3-8B's parameter count, one byte each -> 8 bits per weight.
    path.write_bytes(b"\0" * 819_042_713)
    assert measured_bpw(path, n_params=819_042_713) == pytest.approx(8.0, abs=0.01)
    assert measured_bpw(tmp_path / "absent.gguf") is None

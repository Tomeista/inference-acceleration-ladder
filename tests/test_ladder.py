"""The ladder itself: does the config describe a curve worth measuring?

These are the ladder's equivalent of bench's `test_speculative_pairs_with_a_
matching_non_speculative_control`. The fields they assert on are almost all
inert at run time -- llama.cpp reads the quantization out of the GGUF header
and would not notice if `family` or `bpw_nominal` were wrong -- so their only
job is to reach MLflow correctly and to keep the design property they encode
from being lost when somebody appends a rung.
"""

from __future__ import annotations

import pytest

from ladder.server import (
    LadderConfig,
    check_context_budget,
    load_configs,
    render_serve_script,
    serve_command,
)


@pytest.fixture(scope="module")
def configs():
    return load_configs()


def _group(configs, name: str) -> list[LadderConfig]:
    return [c for c in configs.values() if c.group == name]


# --------------------------------------------------------------------------
# the shape of the curve
# --------------------------------------------------------------------------


def test_the_ladder_spans_the_range_vllm_cannot_reach(configs):
    """The reason this study exists.

    vLLM has three usable widths for this model: 16, 8 and 4. The floor here is
    what one publisher actually ships for Qwen3-8B doing uniform k- and
    I-quants, which is 2.98 bpw measured -- below that nobody publishes at 8B,
    and the sub3_extension group picks up what does exist.
    """
    ladder = _group(configs, "ladder")
    widths = sorted(c.bpw_nominal for c in ladder)
    assert widths[0] < 3.0, "the ladder does not reach past vLLM's 4-bit floor"
    assert widths[-1] > 8.0, "no high-precision rung to anchor the top of the curve"
    assert len(ladder) >= 20, f"only {len(ladder)} rungs; the curve will be coarse"


def test_something_exists_below_the_published_floor(configs):
    """2.98 bpw is where uniform quantization of an 8B stops being published.

    Everything below it is Unsloth dynamic, so it changes publisher and mixture
    policy at once and cannot join the curve. It gets its own group rather than
    being dropped, because the question was how far down the axis goes.
    """
    sub3 = _group(configs, "sub3_extension")
    assert sub3, "nothing below the floor; the axis stops early for no reason"
    ladder_floor = min(c.bpw_nominal for c in _group(configs, "ladder"))
    for cfg in sub3:
        assert cfg.bpw_nominal < ladder_floor
        assert cfg.repo.startswith("unsloth/")


def test_the_matched_size_trio_survives(configs):
    """Three formats at the same measured width: the cleanest test in the set.

    Q4_0 (legacy RTN, no imatrix), Q4_K_S (k-quant) and IQ4_NL (codebook
    I-quant) all land within 0.01 bpw. Bytes read per token are identical, so
    any speed difference between them is kernel, not information content. If a
    re-upload moves one of them, this is no longer that experiment.
    """
    trio = [configs[i].bpw_nominal for i in ("q4_0", "q4_k_s", "iq4_nl")]
    assert max(trio) - min(trio) <= 0.05, f"the trio has drifted apart: {trio}"
    assert configs["iq4_nl"].family == "iquant"
    assert configs["q4_0"].family == "kquant"
    assert configs["q4_k_s"].family == "kquant"


def test_no_gap_in_the_ladder_is_wide_enough_to_hide_a_knee(configs):
    """A curve is only as good as its worst gap.

    Between 16 and 8 bits there is nothing to sample -- no such GGUF type
    exists -- so the check starts at the top of the k-quant range.
    """
    widths = sorted(c.bpw_nominal for c in _group(configs, "ladder"))
    gaps = [(hi, lo, hi - lo) for lo, hi in zip(widths, widths[1:])]
    worst = max(gaps, key=lambda g: g[2])
    assert worst[2] <= 2.0, f"gap of {worst[2]:.2f} bpw between {worst[1]} and {worst[0]}"


def test_every_iquant_has_a_kquant_of_almost_the_same_width(configs):
    """The comparison the ladder is really for.

    I-quants are codebook lookups: more compute per byte than a k-quant. The
    expectation worth testing on the GPU is that some I-quant is *slower* than
    a physically larger k-quant, which inverts the bits-to-speed curve. That
    only shows up if the set actually contains matched-width pairs, so the
    config is required to carry them.
    """
    ladder = _group(configs, "ladder")
    kquants = [c for c in ladder if c.family == "kquant"]
    iquants = [c for c in ladder if c.family == "iquant"]
    assert iquants, "no I-quants: the inversion cannot be observed"

    # Q2_K is the narrowest k-quant llama.cpp has. Below it only I-quants
    # exist -- that is what they are for -- so there is nothing to pair the
    # sub-3-bit rungs against, and requiring it would assert something false
    # about the format rather than something true about this config.
    floor = min(k.bpw_nominal for k in kquants)
    pairable = [c for c in iquants if c.bpw_nominal >= floor]
    assert len(pairable) >= 3, (
        f"only {len(pairable)} I-quant(s) sit above the {floor} bpw k-quant "
        f"floor; the inversion needs several matched pairs to be visible"
    )

    for iq in pairable:
        nearest = min(kquants, key=lambda k: abs(k.bpw_nominal - iq.bpw_nominal))
        delta = abs(nearest.bpw_nominal - iq.bpw_nominal)
        assert delta <= 0.35, (
            f"{iq.id} at {iq.bpw_nominal} bpw has no k-quant partner within "
            f"0.35 bpw (nearest is {nearest.id} at {nearest.bpw_nominal})"
        )


def test_the_curve_comes_from_a_single_publisher(configs):
    """Quantization quality depends on the imatrix calibration set.

    Mixing publishers inside one curve would mean sampling two curves once each
    and plotting them as though they were one.
    """
    repos = {c.repo for c in _group(configs, "ladder")}
    assert len(repos) == 1, f"the ladder group draws on several repos: {sorted(repos)}"


def test_the_anchor_exists_and_is_full_precision(configs):
    """Without it the two studies cannot be related at all."""
    anchors = _group(configs, "anchor")
    assert len(anchors) == 1
    assert anchors[0].family == "fp"
    # 16.01, not 16.00: the GGUF carries metadata and a token table alongside
    # the weights, which is exactly why bpw is measured rather than assumed.
    assert 15.9 <= anchors[0].bpw_nominal <= 16.1


def test_unsloth_dynamic_is_not_part_of_the_curve(configs):
    """A UD quant varies width per tensor, so its bpw is a mean, not a width.

    Reported alongside, never merged: it moves publisher and mixture policy at
    the same time as width.
    """
    unsloth = _group(configs, "unsloth_dynamic")
    assert unsloth, "the dynamic group is empty"
    for cfg in unsloth:
        assert cfg.repo.startswith("unsloth/")
        assert cfg.group != "ladder"
    assert not any(c.repo.startswith("unsloth/") for c in _group(configs, "ladder"))


def test_each_dynamic_rung_has_a_ladder_rung_to_compare_against(configs):
    """The dynamic group is only meaningful as a paired comparison.

    "Does spending the same bytes non-uniformly change speed" needs a uniform
    rung of the same measured width to ask against. Without the pair, a UD row
    is an isolated number and the Unsloth question goes unanswered.
    """
    ladder = _group(configs, "ladder")
    for cfg in _group(configs, "unsloth_dynamic"):
        nearest = min(ladder, key=lambda c: abs(c.bpw_nominal - cfg.bpw_nominal))
        delta = abs(nearest.bpw_nominal - cfg.bpw_nominal)
        assert delta <= 0.25, (
            f"{cfg.id} at {cfg.bpw_nominal} bpw has no uniform rung within "
            f"0.25 bpw (nearest is {nearest.id} at {nearest.bpw_nominal})"
        )


# --------------------------------------------------------------------------
# the pins
# --------------------------------------------------------------------------


def test_serving_parameters_are_identical_across_every_rung(configs):
    """A smaller checkpoint must not receive a longer context and get credit.

    bench pins max_model_len and gpu_memory_utilization for this. The llama.cpp
    equivalents are the per-slot context, the slot count and the KV cache dtype.
    """
    pinned = {
        cfg.id: (
            cfg.n_ctx_per_slot,
            cfg.parallel,
            cfg.cache_type_k,
            cfg.cache_type_v,
            cfg.n_gpu_layers,
            cfg.batch_size,
            cfg.ubatch_size,
        )
        for cfg in configs.values()
    }
    assert len(set(pinned.values())) == 1, pinned


def test_the_kv_cache_is_never_quantized(configs):
    """Quantizing it too would put a second variable on the x-axis."""
    for cfg in configs.values():
        assert cfg.cache_type_k == "f16"
        assert cfg.cache_type_v == "f16"


def test_every_rung_is_fully_offloaded(configs):
    """A rung left partly on CPU would measure PCIe, not quantization."""
    for cfg in configs.values():
        assert cfg.n_gpu_layers >= 99


# --------------------------------------------------------------------------
# the serve command
# --------------------------------------------------------------------------


def test_ctx_size_is_per_slot_times_parallel(configs):
    """llama.cpp's context is the whole cache, split across slots.

    Passing the per-sequence budget straight to --ctx-size would give each slot
    1/parallel of it, and c3_rag's 4000-token prompts would not fit.
    """
    cfg = configs["q4_k_m"]
    argv = serve_command(cfg)
    ctx = int(argv[argv.index("--ctx-size") + 1])
    parallel = int(argv[argv.index("--parallel") + 1])
    assert ctx == cfg.n_ctx_per_slot * parallel
    assert ctx // parallel >= 4128, "a slot cannot hold c3_rag's 4000 in + 128 out"


def test_metrics_endpoint_is_always_requested(configs):
    """Without --metrics the sweep loses every validity check it has."""
    for cfg in configs.values():
        assert "--metrics" in serve_command(cfg)


def test_prefix_reuse_is_disabled(configs):
    """bench's --no-enable-prefix-caching, in llama.cpp spelling."""
    argv = serve_command(configs["q4_k_m"])
    assert argv[argv.index("--cache-reuse") + 1] == "0"


def test_the_serve_script_records_the_context_arithmetic(configs):
    """The generated script has to explain the one number that is derived."""
    script = render_serve_script(configs["q4_k_m"])
    assert "per slot" in script
    assert "llama-server" in script


def test_quantization_reaches_mlflow_params(configs):
    """Inert at run time; the params are the only reason the fields exist."""
    params = configs["iq2_m"].as_params()
    assert params["quant_type"] == "IQ2_M"
    assert params["quant_family"] == "iquant"
    assert params["ladder_group"] == "ladder"
    assert params["engine"] == "llama.cpp"


# --------------------------------------------------------------------------
# the guards
# --------------------------------------------------------------------------


def test_a_cell_beyond_the_slot_count_is_refused(configs):
    """Over-subscribing does not fail, it queues, and reports the wait as latency."""
    cfg = configs["q4_k_m"]
    assert check_context_budget(cfg, cfg.parallel, 4128) is None
    problem = check_context_budget(cfg, cfg.parallel + 1, 4128)
    assert problem is not None and "queue" in problem


def test_a_class_longer_than_a_slot_is_refused(configs):
    cfg = configs["q4_k_m"]
    problem = check_context_budget(cfg, 1, cfg.n_ctx_per_slot + 1)
    assert problem is not None and "slots" in problem


def test_unknown_key_in_a_config_is_rejected(tmp_path):
    path = tmp_path / "ladder.yaml"
    path.write_text(
        "defaults: {}\n"
        "configs:\n"
        "  - id: x\n"
        "    name: x\n"
        "    repo: r\n"
        "    gguf_file: f.gguf\n"
        "    quant_type: Q4_K_M\n"
        "    bpw_nominal: 4.83\n"
        "    family: kquant\n"
        "    bpw_nominl: 4.83\n",  # typo
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown keys"):
        load_configs(path)


def test_duplicate_rung_ids_are_rejected(tmp_path):
    path = tmp_path / "ladder.yaml"
    entry = (
        "  - id: x\n"
        "    name: x\n"
        "    repo: r\n"
        "    gguf_file: f.gguf\n"
        "    quant_type: Q4_K_M\n"
        "    bpw_nominal: 4.83\n"
        "    family: kquant\n"
    )
    path.write_text("defaults: {}\nconfigs:\n" + entry + entry, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_configs(path)

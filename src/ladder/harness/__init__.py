r"""Vendored measurement code, copied from the `bench` study.

These five modules are what actually produces a number: the async load client,
the percentile math, the Prometheus scrape and gauge sampler, the prompt-class
loader and the MLflow wrappers. They were written for the vLLM study and are
reused here unchanged so that a tokens/sec figure from the ladder and one from
bench mean the same thing -- same timing points, same percentile definition,
same "absent rather than zero" convention for metrics an engine does not
publish.

They are *copies*, not an import of a sibling package, so this repository
stands alone. The cost of that choice is real and worth stating: a fix made to
bench's client does not arrive here on its own. The files are therefore kept
byte-identical to their originals apart from two mechanical changes, which is
what keeps a resync to a `diff` rather than a merge:

    sed 's/^from bench\./from ladder.harness./' bench/src/bench/<f>.py

and, in `classes.py` only, the repo-root hop moves from `parents[2]` to
`parents[3]` because this package sits one directory deeper. To check a copy
against an upstream checkout:

    diff <(sed 's/ladder\.harness/bench/' src/ladder/harness/metrics.py) \
         ../bench/src/bench/metrics.py

`metrics.py` still carries the `VLLM` dialect it was written around. That is
dead code here -- the ladder uses `LLAMACPP` from `ladder.dialect` -- and it is
deliberately left in place rather than stripped, because removing it would make
every future diff against bench noisy for no gain.

Not vendored: `build_prompts.py` and `corpus.py`, which regenerate the prompt
sets. Those need `transformers` and a tokenizer checkout, and this study does
not rebuild prompts -- it reads the frozen files in `prompts/`, whose sha256
digests are checked against `prompts/manifest.json` by the test suite.
"""

# Contributing to GraphNetz

Thanks for your interest in contributing. GraphNetz is a research-grade
benchmarking framework, so the bar for new code is *correctness, statistical
honesty, and clarity* — in that order.

## Ground rules

- **No silent baselines.** Every cell in `BENCHMARK_TASKS` must have a real
  held-out metric (test accuracy, test AUC, validation MAE). Self-supervised
  losses are not benchmark metrics; use `train_dgi` / `DGIWrapper` as a
  pre-training utility instead.
- **Statistics first.** New evaluation paths must thread through the
  multi-seed pipeline so the report still produces per-cell CIs, Holm-corrected
  pairwise tests, and Friedman–Nemenyi diagrams without bespoke code.
- **Determinism.** Seed every RNG (`torch`, `numpy`, Python `random`). A run
  with the same seed list and software stack must reproduce bit-for-bit on
  the same hardware.
- **Small, focused PRs.** One loader, one model, or one bug per PR. Keep
  unrelated reformatting out.

## Quick development loop

```bash
git clone https://github.com/quant-sci/graphnetz
cd graphnetz
uv sync --group dev          # install + dev dependencies
uv run pytest                # smoke tests
uv run ruff check            # lint
PYTHONPATH=src uv run python examples/experiment.py  # regenerate paper figures
```

## Adding a dataset loader

1. Pick the right category module under `src/graphnetz/datasets/` (or open an
   issue if a new category is needed — the taxonomy is intentionally small).
2. Add a thin loader function that returns a PyG dataset. Keep it stateless,
   one network per call. Examples in `social.py` and `biology.py`.
3. Register it in `LOADER_REGISTRY` (in `src/graphnetz/datasets/__init__.py`)
   under each task it can serve. A single loader may appear under
   multiple tasks (Cora is both `node_cls` and `link_pred`).
4. If the loader is appropriate for the curated benchmark, add a `Task(...)`
   to `BENCHMARK_TASKS` in `src/graphnetz/benchmark.py` and pick an epoch
   budget that converges on a laptop.
5. Add a one-line entry in `tests/test_smoke.py` so the loader is exercised
   in CI.

## Adding a model

The benchmark dispatches by *task*, not by model name, so models declare
which tasks they support.

```python
from graphnetz import register_model

@register_model(task_type={"node_cls", "graph_cls"})
class MyGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        ...
```

The default factory calls `cls(in_channels, hidden_channels, out_channels)`.
For non-standard signatures, pass a `factory=` callable to `register_model`.

If the model is a node-level encoder that should also work as a
graph-classifier, regressor, or link-predictor, prefer wrapping it with
`graphnetz.benchmark._multi_task_factory` rather than maintaining a separate
implementation per task.

## Adding a task type

Adding a new task type (e.g. `node_reg`, `temporal`) is a four-step change:

1. Append it to `TASK_TYPES` in `benchmark.py`.
2. Add a training routine in `training.py` returning a per-epoch metric dict.
3. Add an adapter in `models/_adapters.py` if node-level encoders should
   plug into the new task via the multi-task factory.
4. Extend `_run_task` in `benchmark.py` with the dispatch branch.

Document the new task in the README's *Task* table.

## Adding a statistical test

Stay in `BenchmarkReport` (`benchmark.py`). New tests should:

- Operate on the per-seed `final_metrics()` tensor, not on training loss.
- Return a structured object (DataFrame / dict) and a LaTeX export method.
- Use the closed-form null distributions in `scipy.stats` rather than
  bootstrap simulations unless the paired-by-seed structure makes the
  bootstrap clearly preferable.

## Code style

- Python 3.10+; type hints on every public function.
- `ruff` is the source of truth for lint; PRs must be `ruff` clean.
- No comments that explain *what* the code does — only *why* a non-obvious
  choice was made.
- Docstrings on public symbols only. One-line summary, optional body, no
  multi-paragraph essays.
- Tests under `tests/`. Smoke tests are fine for new loaders; full coverage
  is required for new statistical helpers.

## Reproducing the paper

```bash
PYTHONPATH=src uv run python paper/experiment.py
latexmk -pdf paper/main.tex
```

If you change `BENCHMARK_TASKS` or any default in `benchmark.py`, please also
re-run `paper/experiment.py` so the cached histories
(`paper/_cache_*.pkl`), figures, and tables stay in sync with the prose.

## Reporting issues

Please include:

- Minimal reproducer (`python -c "..."` is best).
- `python --version`, `pip freeze | grep -E "torch|geometric|graphnetz"`.
- The full traceback, not just the last line.

Security-sensitive issues: open a private issue or email the maintainer
listed in `pyproject.toml` instead of a public issue.

## Code of conduct

Be specific, attack the code not the person. The maintainers reserve
the right to close threads that drift outside that.

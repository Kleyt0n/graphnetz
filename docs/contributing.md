# Contributing

Thanks for your interest in contributing. The bar for new code is
**correctness, statistical honesty, and clarity** — in that order. New
loaders and architectures are welcome; new evaluation shortcuts are not.

## Ground rules

1. **No silent baselines.** Every cell in `BENCHMARK_TASKS` must carry a
   real held-out metric (test accuracy, test AUC, validation MAE).
   Self-supervised losses are not benchmark metrics — use
   `train_dgi` / `DGIWrapper` as a pre-training utility instead.
2. **Statistics first.** New evaluation paths must thread through the
   multi-seed pipeline so the report still produces CIs, Holm-corrected
   pairwise tests, and Friedman–Nemenyi diagrams without bespoke code.
3. **Determinism.** Seed every RNG. A run with the same seed list and
   software stack must reproduce bit-for-bit on the same hardware. The
   benchmark dispatcher already reseeds Python `random`, NumPy, Torch CPU,
   and Torch CUDA; new code paths must not introduce ungated stochasticity.
4. **Small, focused PRs.** One loader, one model, or one bug per PR. Keep
   unrelated reformatting out.

## Quick development loop

```bash
git clone https://github.com/quant-sci/graphnetz
cd graphnetz
uv sync --group dev
uv run pytest          # smoke tests
uv run ruff check      # lint (must be clean before review)
```

## Adding a dataset loader

1. Pick the right category module under `src/graphnetz/datasets/`. Open an
   issue first if a new top-level category is needed — the taxonomy is
   intentionally small.
2. Write a thin loader function that returns a PyG dataset. Keep it
   stateless and one network per call. See `social.py` and `biology.py`
   for reference shapes.
3. Register it in `LOADER_REGISTRY` under each task kind it can serve. A
   single loader may appear under multiple kinds (e.g. `cora` is both
   `node_cls` and `link_pred`).
4. If the loader is appropriate for the curated benchmark, add a
   `Task(...)` entry to `BENCHMARK_TASKS` in `benchmark.py` and pick an
   epoch budget that converges on a laptop.
5. Add a one-line entry in `tests/test_smoke.py` so the loader is
   exercised in CI.

## Adding a model

The dispatcher routes by *task kind*, not by model name, so models declare
which kinds they support up front:

```python
from graphnetz import register_model

@register_model(kinds={"node_cls", "graph_cls"})
class MyGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        ...
```

The default factory calls `cls(in_channels, hidden_channels, out_channels)`.
For non-standard signatures, pass a `factory=` callable. For node-level
encoders that should plug into every task kind, prefer wrapping with
{py:func}`graphnetz.benchmark._multi_kind_factory` rather than maintaining
a separate implementation per kind.

## Adding a task kind

Adding a new kind (e.g. `node_reg`, `temporal`) is a four-step change:

1. Append the new kind to `TASK_KINDS` in `benchmark.py`.
2. Add a training routine in `training.py` returning a per-epoch metric
   dict (the existing trainers are the template).
3. Add an adapter in `models/_adapters.py` if node-level encoders should
   plug into the new kind via the multi-kind factory.
4. Extend `_run_task` in `benchmark.py` with the dispatch branch.

Then document the new kind in [Dataset taxonomy →
Task kinds](datasets.md#task-kinds).

## Adding a statistical test

Stay inside `BenchmarkReport` (`benchmark.py`). New tests should:

- Operate on the per-seed `final_metrics()` table, not on training loss.
- Return a structured object (DataFrame / dict) and ship a matching LaTeX
  exporter.
- Prefer closed-form null distributions from `scipy.stats` over bootstrap
  simulation, unless the paired-by-seed structure makes the bootstrap
  clearly preferable (see the percentile-bootstrap CI helper for the
  pattern).

## Building the docs

```bash
uv sync --group docs
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
open docs/_build/html/index.html
```

The `-W` flag treats warnings as errors; CI also runs the docs build, so
keep it warning-clean.

## Code style

- Python 3.10+; type hints on every public function.
- `ruff` is the source of truth for lint and formatting — PRs must be
  `ruff` clean.
- Docstrings on **public** symbols only: one-line summary, optional body,
  no multi-paragraph essays.
- Comments explain *why*, not *what* — well-named identifiers cover the
  *what*.
- Tests under `tests/`. Smoke tests are fine for new loaders; full
  coverage is required for new statistical helpers.

## Reporting issues

Please include:

- A minimal reproducer (`python -c "..."` is best).
- `python --version`, `pip freeze | grep -E "torch|geometric|graphnetz"`.
- The full traceback, not just the last line.

For security-sensitive issues, please open a private issue or email the
maintainer listed in `pyproject.toml` rather than opening a public issue.

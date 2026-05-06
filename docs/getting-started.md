# Getting started

This page takes you from a clean environment to a five-seed benchmark with a
LaTeX-ready summary table and a critical-difference diagram. It assumes
familiarity with PyTorch and PyG; everything else is covered as we go.

## Install

```bash
uv add graphnetz
# or, in an existing environment:
pip install graphnetz
```

For local development, clone the repo and use the `dev` group:

```bash
git clone https://github.com/quant-sci/graphnetz
cd graphnetz
uv sync --group dev
```

Requires **Python ≥ 3.10**, **PyTorch ≥ 2.6**, and **torch-geometric ≥ 2.6**.
Optional extras: `graphnetz[ogb]` for OGB loaders, `graphnetz[chem]` to
pull in RDKit (required by OGB molecular loaders such as `ogbg-molhiv`).

## Train one model

The single-task trainers accept any `nn.Module` and return a per-epoch
history dict ready for plotting:

```python
from graphnetz import GCN, train_node_classification, plot_history
from graphnetz.datasets.social import cora

ds = cora("data/cora")
model = GCN(ds.num_features, 64, ds.num_classes)
history = train_node_classification(model, ds[0], epochs=200)
fig, ax = plot_history(history, title="GCN on Cora")
```

Use this when you only need *one* model on *one* dataset and don't care about
cross-seed variance. For everything else — multi-seed, multi-task,
multi-model — reach for `run_benchmark`.

```{tip}
**GPU is automatic.** Both the standalone trainers and `run_benchmark`
accept `device='auto'` (the default). The runtime picks CUDA when
available, then Apple-silicon MPS, then CPU, and moves the model and
data onto it for you. Pin placement explicitly with `device='cpu'` (or
any `torch.device`) when you need to.
```

## Run a multi-seed benchmark

```python
from graphnetz import GAT, GCN, GraphSAGE, GraphTransformer, run_benchmark

report = run_benchmark(
    "social",
    {"GCN": GCN, "GAT": GAT, "GraphSAGE": GraphSAGE, "GraphTransformer": GraphTransformer},
    seeds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    task_type="node_cls",
)

print(report.summary())          # per-(task, model) mean ± t-CI
print(report.pairwise())         # Holm-corrected paired t-tests
fig, _ = report.plot_critical_difference(alpha=0.05)
report.to_latex("results.tex")   # publication-ready table
```

The same call works for any category and task family: pass `task_type="graph_cls"`
to benchmark on graph classification, `task_type="link_pred"` for link
prediction, and so on. Pass `only=[task_name, ...]` to restrict to specific
loaders.

## Plug in your own model

Custom models go through the same statistical pipeline as the built-ins —
multi-seed, Holm-corrected, CD-diagrammed — once they declare which task they support. Three integration paths cover the common cases:

**Decorator** — permanent registration at import time. Best for libraries
or shared modules:

```python
import torch
from torch_geometric.nn import GCNConv
from graphnetz import register_model

@register_model(task_type={"node_cls"})
class MyGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, ei = data.x, data.edge_index
        return self.conv2(torch.relu(self.conv1(x, ei)), ei)

run_benchmark("social", {"MyGNN": MyGNN}, task_type="node_cls", seeds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
```

**Class attribute** — same effect, no decorator dependency:

```python
class MyGNN(torch.nn.Module):
    task_types = {"node_cls"}
    ...
```

**Inline tuple** — one-shot variants for hyperparameter sweeps. The third
slot is a factory `(in_channels, hidden_channels, out_channels) -> Module`:

```python
run_benchmark(
    "social",
    {
        "MyGNN-d0.3": (MyGNN, "node_cls", lambda i, h, o: MyGNN(i, h, o, dropout=0.3)),
        "MyGNN-d0.5": (MyGNN, "node_cls", lambda i, h, o: MyGNN(i, h, o, dropout=0.5)),
    },
)
```

For node-level encoders that should run on **all four** task types without
writing the adapter glue, see the
[multi-tasks factory](models.md#multi-task-factory).

## Plug in your own dataset

Custom datasets get the same statistical pipeline as the built-ins.
The minimal contract is the standard PyG one — your dataset object exposes
`ds[0]` returning a `Data`, plus the relevant attributes for the task
(`num_features`, `num_classes`, or `num_relations`).

**Quickest path** — wrap an already-loaded dataset and pass it via
`tasks=`:

```python
from graphnetz import GCN, run_benchmark, task_from_dataset

# Your dataset (any PyG-shaped object).
ds = my_loader("data/my_dataset")

task = task_from_dataset("my_dataset", "node_cls", ds, epochs=100)
report = run_benchmark(
    models={"GCN": GCN},
    tasks=[task],
    seeds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
)
```

No `BENCHMARK_TASKS` mutation, no global state — `tasks=` bypasses the
registry entirely. ``category`` defaults to ``"custom"`` for cache-path
namespacing.

**Permanent registration** — make your dataset visible to
`run_benchmark(category, ...)` and `iter_benchmark_tasks`:

```python
from graphnetz import register_task, task_from_dataset, unregister_task

register_task("biology", task_from_dataset("my_assay", "graph_cls", ds, epochs=50))

# ... later, if you want to remove it:
unregister_task("biology", "my_assay")
```

**Seed-aware loaders** — for synthetic datasets where each seed should
produce a fresh sample, write a loader that takes a `seed` keyword. The
dispatcher detects it via `inspect.signature` and passes the benchmark
seed in:

```python
from graphnetz.benchmark import Task

def my_loader(root: str, *, seed: int):
    return MySyntheticDataset(root, num_graphs=100, seed=seed)

task = Task("synthetic_g100", "graph_cls", my_loader, epochs=20)
report = run_benchmark(models={"GCN": GCN}, tasks=[task], seeds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
```

```{tip}
The conventions for each task (which attributes the dataset must
expose, how splits are encoded) live in the trainer docstrings:
{py:func}`graphnetz.train_node_classification`,
{py:func}`graphnetz.train_graph_classification`,
{py:func}`graphnetz.train_graph_regression`,
{py:func}`graphnetz.train_link_prediction`, and
{py:func}`graphnetz.train_relational_link_prediction`.
```

## Five-minute tour

1. **Pick a category.** `combinatorial`, `biology`, `social`, `knowledge`,
   `infrastructure`, `finance`, `computing`, `vision`, `physics`, `security`.
   Installing the optional `ogb` extra adds OGB datasets to the existing
   domain categories (e.g. `ogbn-arxiv` joins `social/node_cls`,
   `ogbg-molhiv` joins `biology/graph_cls`).
2. **Pick a task.** `node_cls`, `graph_cls`, `graph_reg`, or
   `link_pred`. The runner skips models that don't declare support for the
   chosen task.
3. **Pick architectures.** Any subset of the five built-ins, or your own —
   see [Models & adapters](models.md#custom-models).
4. **Run.** `run_benchmark(category, models, task_type=..., seeds=...)`. Use
   `seeds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)` for the default reproducible 10-seed sweep.
5. **Report.** Call `summary`, `pairwise`, `plot_critical_difference`,
   `plot_pairwise`, `plot_forest`, `plot_learning_curves`, `to_latex`,
   `pairwise_to_latex`. Every method works on the same
   {py:class}`~graphnetz.benchmark.BenchmarkReport`.

## Next steps

- Browse the [dataset taxonomy](datasets.md) to find loaders that match
  your domain.
- Read [Reading the report](report.md) to learn which plot or table answers
  which question.
- Skim [Contributing](contributing.md) before adding a new loader, model,
  or task so your additions thread through the same pipeline.

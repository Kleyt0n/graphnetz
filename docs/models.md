# Models & adapters

GraphNetz follows a simple rule: **a node-level encoder should run on every
task kind without rewriting code**. The five built-in architectures plug into
node classification, graph classification, graph regression, and link
prediction through three small adapter wrappers; GIN keeps its native
graph-level pooling.

## Built-in architectures

| Model | Supported kinds | Reference |
|---|---|---|
| {py:class}`~graphnetz.models.GCN` | all four | Kipf & Welling, ICLR 2017 |
| {py:class}`~graphnetz.models.GAT` | all four | Veličković et al., ICLR 2018 |
| {py:class}`~graphnetz.models.GIN` | `graph_cls`, `graph_reg` | Xu et al., ICLR 2019 |
| {py:class}`~graphnetz.models.GraphSAGE` | all four | Hamilton et al., NeurIPS 2017 |
| {py:class}`~graphnetz.models.GraphTransformer` | all four | Shi et al., 2021 |
| {py:class}`~graphnetz.models.DGI` | *(utility)* | Veličković et al., ICLR 2019 |

```{note}
DGI is exposed as a self-supervised *pre-training* utility, not a benchmark
task. Its loss is its own metric — there is no held-out signal — so the
runner routes unlabelled graphs through `link_pred` (a real edge-split AUC)
instead of DGI loss.
```

## Task-kind adapters

Three thin wrappers make a node-level encoder fluent in every task family.
The benchmark dispatcher picks the right one automatically based on the
chosen `kind`:

```{eval-rst}
.. autoclass:: graphnetz.models._adapters.GraphLevelWrapper
   :no-index:
.. autoclass:: graphnetz.models._adapters.LinkPredWrapper
   :no-index:
.. autoclass:: graphnetz.models._adapters.DGIWrapper
   :no-index:
```

You won't usually instantiate them directly — `run_benchmark` wraps your
encoder for you:

```python
from graphnetz import GAT
model = GAT(in_channels=8, hidden_channels=64, out_channels=4)
# run_benchmark wraps `model` in the right adapter for the requested kind.
```

## Custom models

Models declare which task kinds they support and the dispatcher skips
incompatible *(model, task)* pairs. Three integration paths cover the common
cases:

### 1. Decorator (recommended for libraries)

Permanent registration; the model is then visible to every `run_benchmark`
call by name:

```python
import torch
from graphnetz import register_model

@register_model(kinds={"node_cls", "graph_cls"})
class MyGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        ...
```

### 2. Class attribute (no decorator)

If you'd rather not depend on `register_model` at import time:

```python
class MyGNN(torch.nn.Module):
    task_kinds = {"node_cls", "graph_cls"}
    ...
```

### 3. Inline tuple (one-shot experiments)

Useful for hyperparameter sweeps where each variant needs a different
factory:

```python
from graphnetz import run_benchmark

run_benchmark(
    "social",
    {
        "MyGNN-d0.3": (MyGNN, "node_cls", lambda i, h, o: MyGNN(i, h, o, dropout=0.3)),
        "MyGNN-d0.5": (MyGNN, "node_cls", lambda i, h, o: MyGNN(i, h, o, dropout=0.5)),
    },
)
```

### Multi-kind factory

For a node-level encoder that should work across **all four** task kinds
without you writing the adapter glue:

```python
from graphnetz.benchmark import _multi_kind_factory, register_model

class MyEncoder(torch.nn.Module):
    """Returns per-node embeddings of shape [N, hidden_channels]."""
    ...

_ALL_KINDS = {"node_cls", "graph_cls", "graph_reg", "link_pred"}
register_model(MyEncoder, kinds=_ALL_KINDS, factory=_multi_kind_factory(MyEncoder))
```

## Choosing an integration path

| Scenario | Use |
|---|---|
| You're publishing a new architecture | **Decorator** — clean import surface, name-based discovery |
| You're benchmarking someone else's encoder | **Class attribute** — no edits to upstream code beyond adding `task_kinds` |
| You're sweeping over hyperparameters | **Inline tuple** — one factory per variant in the same `run_benchmark` call |
| You have a node-level encoder that should run everywhere | **Multi-kind factory** — `_multi_kind_factory(cls)` does the wrapping |

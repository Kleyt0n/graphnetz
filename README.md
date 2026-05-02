<p align="center">
  <img src="assets/logo-banner.svg" alt="GraphNetz" width="460">
</p>

<p align="center"><em>Statistically rigorous GNN benchmarking</em></p>

<p align="center">
  <a href="https://github.com/quant-sci/graphnet/actions"><img alt="Build" src="https://img.shields.io/badge/build-passing-22333B?style=flat-square&labelColor=EAE0D5"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-22333B?style=flat-square&labelColor=EAE0D5"></a>
  <a href="LICENCE.txt"><img alt="License" src="https://img.shields.io/badge/license-MIT-22333B?style=flat-square&labelColor=EAE0D5"></a>
  <a href="https://arxiv.org/"><img alt="Paper" src="https://img.shields.io/badge/paper-PDF-22333B?style=flat-square&labelColor=EAE0D5"></a>
</p>

---

## Why GraphNetz

Most GNN benchmarks report point-estimate accuracies on a handful of citation graphs and declare a winner without confidence intervals, multiple-comparison correction, or rank aggregation across datasets. GraphNetz's default output is a **structured statistical report**, not a raw accuracy table:

- multi-seed Student's *t* confidence intervals per cell,
- Holm–Bonferroni paired *t*-tests (or Wilcoxon signed-rank) within each task,
- Demšar critical-difference diagrams from Friedman ranks with a Nemenyi
  post-hoc.

The catalogue is organised along a **category × task** taxonomy: 58 dataset
loaders across 10 scientific categories crossed with 4 task kinds (node
classification, graph classification, graph regression, link prediction). Five
canonical architectures (GCN, GAT, GIN, GraphSAGE, Graph Transformer) plug into
every kind via a small set of task-kind adapters; Deep Graph Infomax is
exposed as an optional pre-training utility.

## Install

```bash
uv add graphnetz
# or, in an existing environment:
pip install graphnetz
```

For local development:

```bash
git clone https://github.com/quant-sci/graphnet
cd graphnet
uv sync --group dev
```

GraphNetz requires Python ≥ 3.10, `torch ≥ 2.6`, and `torch-geometric ≥ 2.6`.

## Quick start

```python
from graphnetz import GCN, train_node_classification, plot_history
from graphnetz.datasets.social import cora

ds = cora("data/cora")
model = GCN(ds.num_features, 64, ds.num_classes)
history = train_node_classification(model, ds[0], epochs=200)
fig, ax = plot_history(history, title="GCN on Cora")
```

For a full benchmark run with the default statistical report:

```python
from graphnetz import GAT, GCN, GraphSAGE, GraphTransformer, run_benchmark

report = run_benchmark(
    "social",
    {"GCN": GCN, "GAT": GAT, "GraphSAGE": GraphSAGE, "GraphTransformer": GraphTransformer},
    seeds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    kind="node_cls",          # restrict to one task family
)
print(report.summary())       # per-(task, model) mean ± t-CI
print(report.pairwise())      # Holm-corrected paired t-tests (or Wilcoxon)
fig, _ = report.plot_critical_difference(alpha=0.05)
```

## Task kinds

| Kind | Symbol | Metric | Examples |
|---|---|---|---|
| Node classification | `node_cls` | test accuracy | Cora, Roman-empire |
| Graph classification | `graph_cls` | val accuracy | MUTAG, MNIST-superpixels |
| Graph regression | `graph_reg` | val MAE | ZINC, QM9 |
| Link prediction | `link_pred` | test AUC | FB15k-237, Internet AS |

Unlabelled graphs (Netzschleuder, synthetic combinatorial, Ising lattice)
enter the benchmark through link prediction on a held-out edge split, so
every cell carries a real test-time metric — there is no self-supervised
*pretext* loss in the headline report.

## Dataset categories

| Category | # | Task kinds | Loaders |
|---|---:|---|---|
| Combinatorial | 6 | GC, GR, LP | random TSP, VRP, max-flow, bipartite matching, coloring, max-cut |
| Biology | 10 | GC, GR, NC, LP | MUTAG, PROTEINS, ENZYMES, Peptides-func/struct, PPI, C. elegans, Budapest connectome, hospital/high-school contacts |
| Social | 14 | NC, LP | Cora, CiteSeer, PubMed, WikiCS, Roman-empire, Amazon-ratings, Minesweeper, Tolokers, Questions, MovieLens-100k, Karate, Facebook friends, DBLP coauthor, DNC emails |
| Knowledge | 3 | LP | FB15k-237, WordNet18-RR, WordNet (Netz) |
| Infrastructure | 6 | LP | power grid, EuroRoad, US roads, EU airlines, London transport, urban streets |
| Finance | 4 | NC, LP | Elliptic Bitcoin, product space, board of directors, US patents |
| Computing | 4 | LP | Internet AS, Internet topology, AS-Skitter, route views |
| Vision | 5 | GC, NC | MNIST/CIFAR-10 superpixels, ModelNet10/40, ShapeNet |
| Physics | 3 | GR, LP | QM9, ZINC, Ising lattice |
| Security | 3 | GC, LP | MalNet-Tiny, 9/11 terrorists, train terrorists |
| OGB | 2 | NC, GC | ogbn-arxiv, ogbg-molhiv (requires `pip install graphnetz[ogb]`) |

```python
from graphnetz.datasets.social import cora, roman_empire
from graphnetz.datasets.biology import peptides_func
from graphnetz.datasets.computing import internet_as
from graphnetz.datasets.ogb import ogbn_arxiv, ogbg_molhiv

ds_cora = cora("data/cora")
ds_rom  = roman_empire("data/roman_empire")        # heterophilic
ds_pep  = peptides_func("data/peptides_func")      # LRGB
ds_inet = internet_as("data/internet_as")          # Netzschleuder
```

For arbitrary [Netzschleuder](https://networks.skewed.de/) networks:

```python
from graphnetz import Netz
ds = Netz(root="data", dataset_name="urban_streets", network_name="brasilia")
```

## Models

| Model | Kinds | Source |
|---|---|---|
| `GCN`  | all four | Kipf & Welling, ICLR 2017 |
| `GAT`  | all four | Veličković et al., ICLR 2018 |
| `GIN`  | `graph_cls`, `graph_reg` | Xu et al., ICLR 2019 |
| `GraphSAGE` | all four | Hamilton et al., NeurIPS 2017 |
| `GraphTransformer` | all four | Shi et al., 2021 |
| `DGI` | *(utility)* | Veličković et al., ICLR 2019 |

Node-level encoders enter every task kind through three small adapters:
graph-level pooling head, dot-product link-prediction head, and the DGI
self-supervised wrapper for optional unsupervised pre-training.

## Custom models

```python
from graphnetz import register_model

# 1. Decorator
@register_model(kinds="node_cls")
class MyGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels): ...

# 2. Class attribute (no decorator)
class MyGNN(torch.nn.Module):
    task_kinds = {"node_cls", "graph_cls"}

# 3. Inline tuple at run-time
run_benchmark(
    "social",
    {"MyGNN": (MyGNN, "node_cls",
               lambda i, h, o: MyGNN(i, h, o, dropout=0.3))},
)
```

## The statistical report

`run_benchmark(...)` returns a `BenchmarkReport` with the following methods:

| Method | Output |
|---|---|
| `report.summary(ci=0.95)` | per-(task, model) mean ± *t*-CI half-width DataFrame |
| `report.pairwise(alpha=0.05)` | Holm-corrected paired *t*-tests or Wilcoxon signed-rank tests within each task |
| `report.plot_critical_difference()` | Demšar / Nemenyi CD diagram across tasks |
| `report.plot_pairwise(layout=...)` | matrix or list view of pairwise significance |
| `report.plot_forest()` | per-task forest plot of mean ± CI |
| `report.plot_learning_curves()` | shared-y learning curves with t-CI bands |
| `report.to_latex(path)` | publication-ready bold-best LaTeX table |
| `report.pairwise_to_latex(path)` | Holm pairwise LaTeX table (parametric or non-parametric) |

## Notebooks

Worked examples live under `examples/`:

- `01_benchmark.ipynb` — the cross-category dashboard (multi-seed report,
  bootstrap CIs, custom-model integration).
- `02_knowledge.ipynb` — relational link prediction on FB15k-237 / WN18-RR
  using the DistMult decoder.

## Reproducing the paper

```bash
PYTHONPATH=src uv run python paper/experiment.py   # train + cache + figures
latexmk -pdf paper/main.tex                        # compile PDF
```

The script trains 5 architectures × 10 seeds across the 10 surviving
categories, caches the histories under `paper/_cache_*.pkl`, and writes every
figure (`paper/figures/`) and LaTeX table (`paper/tables/`) referenced by
`paper/main.tex`. Total runtime on a recent laptop CPU is under 30 minutes.

## Issues

Track issues at [github.com/quant-sci/graphnetz/issues](https://github.com/quant-sci/graphnet/issues).

## Citation

If GraphNetz is useful in your work, please cite the accompanying paper:

```bibtex
@article{dacosta2026graphnetz,
  title   = {GraphNetz: A Statistical-Reporting Layer for Graph Neural Network Benchmarks},
  author  = {da Costa, Kleyton and Modenesi, Bernardo},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## Contributing

Pull requests welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) first — the
short version is: every benchmark cell must carry a real held-out metric,
every change must thread through the multi-seed pipeline, and every PR must
be `ruff` clean.

```bash
uv run pytest
uv run ruff check
```

## License

MIT — see [`LICENCE.txt`](LICENCE.txt).

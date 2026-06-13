---
hide-toc: true
---

# Introduction

[![GitHub](https://img.shields.io/github/stars/Kleyt0n/graphnetz?style=flat-square&logo=github&logoColor=001233&labelColor=979dac&color=001233)](https://github.com/Kleyt0n/graphnetz)
[![Build](https://img.shields.io/badge/build-passing-001233?style=flat-square&labelColor=979dac)](https://github.com/Kleyt0n/graphnetz/actions)
[![Docs](https://img.shields.io/badge/passing-docs-001233?style=flat-square&labelColor=979dac)](https://graphnetz.readthedocs.io/en/latest/)
[![Python](https://img.shields.io/badge/python-3.10%2B-001233?style=flat-square&labelColor=979dac)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-001233?style=flat-square&labelColor=979dac)](https://github.com/Kleyt0n/graphnetz/blob/main/LICENCE.txt)
[![Paper](https://img.shields.io/badge/paper-PDF-001233?style=flat-square&labelColor=979dac)](https://arxiv.org/)

**A GNN benchmark whose default output is a statistical report, not a leaderboard.**

Whether you are proposing a new GNN architecture, testing a model on a new graph domain, or comparing existing methods across graph types, GraphNetz turns the usual “train, evaluate, table of accuracies” workflow into a reproducible statistical report. Instead of reporting point estimates alone, it provides confidence intervals for each result, paired model comparisons with multiple-testing correction, and rank-based summaries across datasets using critical-difference diagrams. The goal is not just to crown a leaderboard winner, but to give researchers a principled way to quantify uncertainty, compare methods fairly, and produce the exact evidence reviewers often ask for in graph-learning papers.

```{figure} _static/critical_difference.png
:alt: Demšar critical-difference diagram comparing four GNN architectures by mean rank.
:class: gn-cd only-light
:align: center

```

```{figure} _static/critical_difference_dark.png
:alt: Demšar critical-difference diagram comparing four GNN architectures by mean rank.
:class: gn-cd only-dark
:align: center

A Demšar critical-difference diagram . Models are ordered by mean Friedman
rank; the horizontal bar connects groups whose ranks are not significantly
different at the chosen $\alpha$ under the Nemenyi post-hoc.
```


## Install

```bash
pip install graphnetz
# or
uv add graphnetz
```

Requires Python ≥ 3.10, PyTorch ≥ 2.6, torch-geometric ≥ 2.6.

## Quick Start

```python
from graphnetz import GAT, GCN, GraphSAGE, run_benchmark

report = run_benchmark(
    "social",
    {"GCN": GCN, "GAT": GAT, "GraphSAGE": GraphSAGE},
    seeds=range(10),
    task_type="node_cls",
)

print(report.summary())          # per-(task, model) mean ± t-CI
print(report.pairwise())         # Holm-corrected paired t-tests
report.plot_critical_difference(alpha=0.05)
report.to_latex("results.tex")   # publication-ready table
```

→ Walk through this end-to-end in **[Getting started](getting-started.md)**.


## Why GraphNetz

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Honest comparisons by default
Per-cell Student's-*t* (or percentile-bootstrap) CIs, Holm-adjusted paired
*t*-tests within each task, Friedman ranks plus Nemenyi CD across tasks —
no extra bookkeeping.
:::

:::{grid-item-card} One call, every metric
`run_benchmark(category, models, seeds=...)` trains every compatible
*(task, model, seed)* triple and returns a
{py:class}`~graphnetz.benchmark.BenchmarkReport`.
:::

:::{grid-item-card} Publication-ready artefacts
`report.to_latex(...)`, `plot_forest()`, `plot_pairwise()`,
`plot_critical_difference()`
:::

:::{grid-item-card} Pluggable models
Decorator, class attribute, or inline tuple — your encoder runs through the
same statistical pipeline as the built-ins.
:::

::::

## At a glance

| | |
|---|---|
| **Tasks** | `node_cls` · `graph_cls` · `graph_reg` · `link_pred` |
| **Architectures** | GCN · GAT · GIN · GraphSAGE · GraphTransformer (DGI as a pre-training utility) |
| **Loaders** | 63 across 10 categories (combinatorial, biology, social, knowledge, infrastructure, finance, computing, vision, physics, security) |
| **Default report** | per-cell mean ± Student's-*t* CI · Holm-adjusted paired *t* · Demšar/Nemenyi CD |
| **Source** | [github.com/Kleyt0n/graphnetz](https://github.com/Kleyt0n/graphnetz) |

## Documentation

**Start here**
- [Getting started](getting-started.md) — install and run your first benchmark in five minutes.

**Concepts**
- [Dataset taxonomy](datasets.md) — the full *category × task* grid and how to pick a loader.
- [Models & adapters](models.md) — built-in encoders and three ways to plug in your own.
- [Benchmark protocol](benchmark.md) — the five-stage pipeline that turns raw histories into a publishable report.
- [Reading the report](report.md) — which view to use for which question.

**Reference**
- [API reference](api/index) — modules, classes, and functions.
- [Contributing](contributing.md) — add a loader, a model, or a new task.

```{toctree}
:maxdepth: 2
:hidden:
:caption: Start here

getting-started
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Concepts

datasets
models
benchmark
report
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Reference

api/index
contributing
```

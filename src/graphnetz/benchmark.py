"""Statistically robust benchmarks across a category for one or many models.

The dispatcher trains every compatible (model, task) pair across multiple
seeds and returns a :class:`BenchmarkReport` that exposes mean ± 95 % t-CI,
paired t-tests with Holm-Bonferroni correction, publication-ready LaTeX
tables, and Nature-styled plots.

Custom models are plugged in via the same three paths as before:

1. **Decorator / registry**::

       from graphnetz import register_model


       @register_model(kinds="node_cls")
       class MyGNN(torch.nn.Module):
           def __init__(self, in_channels, hidden_channels, out_channels): ...

2. **Class attribute**::

       class MyGNN(torch.nn.Module):
           task_kinds = {"node_cls"}

3. **Inline tuple** ``(cls, kinds)`` or ``(cls, kinds, factory)`` in the
   ``models`` mapping::

       run_benchmark("social", {"MyGNN": (MyGNN, "node_cls")})

The default factory calls ``cls(in_channels, hidden_channels, out_channels)``;
DGI-kind models receive ``(in_channels, hidden_channels)`` (the third argument
is dropped).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from graphnetz.datasets import (
    biology,
    combinatorial,
    computing,
    finance,
    infrastructure,
    knowledge,
    physics,
    security,
    social,
    vision,
)
from graphnetz.models import GAT, GCN, GIN, GraphSAGE, GraphTransformer
from graphnetz.plotting import NATURE_COLORS, plot_grouped_bars, save_figure, set_plot_style
from graphnetz.training import (
    train_graph_classification,
    train_graph_regression,
    train_link_prediction,
    train_node_classification,
)

_HAS_OGB = importlib.util.find_spec("ogb") is not None

# DGI is intentionally not a task kind: it is a self-supervised training
# objective whose "metric" is its own loss, so it cannot serve as a
# held-out evaluation. ``train_dgi`` and the ``DGIWrapper`` adapter remain
# available as utilities for users who want to pre-train an encoder
# unsupervised; the benchmark routes unlabelled graphs through
# ``link_pred`` instead (a real held-out edge split with an AUC metric).
TASK_KINDS: frozenset[str] = frozenset({"node_cls", "graph_cls", "graph_reg", "link_pred"})
_METRIC_KEYS: tuple[str, ...] = (
    "test_acc",
    "test_auc",
    "val_acc",
    "val_auc",
    "val_mae",
)
_LOWER_IS_BETTER: frozenset[str] = frozenset({"val_mae", "train_loss"})


# --------------------------------------------------------------------------- #
# Tasks and model specs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Task:
    """A single benchmark task: a dataset loader plus its training kind."""

    name: str
    kind: str
    # ``...`` admits seed-aware loaders ``f(root, *, seed=...)`` alongside
    # the basic ``f(root)`` shape — the dispatcher inspects the signature
    # and threads ``seed`` through when present.
    loader: Callable[..., Any]
    epochs: int = 30


@dataclass(frozen=True)
class ModelSpec:
    """How to instantiate a model and which task kinds it supports."""

    cls: type
    kinds: frozenset[str] = field(default_factory=frozenset)
    factory: Callable[..., torch.nn.Module] | None = None

    def build(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        *,
        kind: str = "node_cls",
    ) -> torch.nn.Module:
        if self.factory is not None:
            try:
                return self.factory(in_channels, hidden_channels, out_channels, kind=kind)
            except TypeError:
                return self.factory(in_channels, hidden_channels, out_channels)
        if kind == "dgi":
            return self.cls(in_channels, hidden_channels)
        return self.cls(in_channels, hidden_channels, out_channels)


_REGISTRY: dict[type, ModelSpec] = {}


def register_model(
    cls: type | None = None,
    *,
    kinds: str | Iterable[str],
    factory: Callable[..., torch.nn.Module] | None = None,
) -> Callable[[type], type] | type:
    """Register a model with the benchmark dispatcher.

    Usable as a decorator (``@register_model(kinds="node_cls")``) or as a
    plain function (``register_model(MyGNN, kinds={"graph_cls", "graph_reg"})``).
    """
    ks = frozenset({kinds} if isinstance(kinds, str) else kinds)
    unknown = ks - TASK_KINDS
    if unknown:
        msg = f"Unknown task kinds: {sorted(unknown)}; allowed: {sorted(TASK_KINDS)}"
        raise ValueError(msg)

    def _register(target: type) -> type:
        _REGISTRY[target] = ModelSpec(cls=target, kinds=ks, factory=factory)
        return target

    return _register(cls) if cls is not None else _register


def _multi_kind_factory(encoder_cls: type) -> Callable[..., torch.nn.Module]:
    """Adapt a node-level encoder to any of the four task kinds.

    For ``node_cls`` the encoder is built with the dataset's class count
    as ``out_channels`` and used directly. For ``graph_cls`` and
    ``graph_reg`` the encoder produces ``hidden_channels`` per node and a
    :class:`GraphLevelWrapper` adds global mean pooling and a head. For
    ``dgi`` the encoder is wrapped in :class:`DGIWrapper` so it plugs
    into the same training loop as :class:`graphnetz.models.DGI`.
    """
    from graphnetz.models._adapters import GraphLevelWrapper, LinkPredWrapper

    def factory(
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        *,
        kind: str = "node_cls",
    ) -> torch.nn.Module:
        if kind == "node_cls":
            return encoder_cls(in_channels, hidden_channels, out_channels)
        if kind in ("graph_cls", "graph_reg"):
            encoder = encoder_cls(in_channels, hidden_channels, hidden_channels)
            return GraphLevelWrapper(encoder, hidden_channels, out_channels)
        if kind == "link_pred":
            encoder = encoder_cls(in_channels, hidden_channels, hidden_channels)
            return LinkPredWrapper(encoder)
        msg = f"Unknown task kind: {kind}"
        raise ValueError(msg)

    return factory


# Pre-register built-ins. Node-level encoders are registered for every
# task kind via the multi-kind factory; GIN keeps its native graph-level
# pooling. ``DGI`` is intentionally not registered: it is exposed as a
# self-supervised training utility (``train_dgi`` + ``DGIWrapper``)
# rather than a benchmark-task model.
_ALL_KINDS = frozenset({"node_cls", "graph_cls", "graph_reg", "link_pred"})
register_model(GCN, kinds=_ALL_KINDS, factory=_multi_kind_factory(GCN))
register_model(GAT, kinds=_ALL_KINDS, factory=_multi_kind_factory(GAT))
register_model(GraphSAGE, kinds=_ALL_KINDS, factory=_multi_kind_factory(GraphSAGE))
register_model(GraphTransformer, kinds=_ALL_KINDS, factory=_multi_kind_factory(GraphTransformer))
register_model(GIN, kinds={"graph_cls", "graph_reg"})


def _spec_from(value: type | tuple[Any, ...] | ModelSpec) -> ModelSpec:
    """Resolve a ``models`` dict entry to a :class:`ModelSpec`."""
    if isinstance(value, ModelSpec):
        return value
    if isinstance(value, tuple):
        cls = value[0]
        kinds = value[1] if len(value) >= 2 else None
        factory = value[2] if len(value) >= 3 else None
        if kinds is None:
            base = _spec_from(cls)
            return ModelSpec(cls=base.cls, kinds=base.kinds, factory=factory or base.factory)
        ks = frozenset({kinds} if isinstance(kinds, str) else kinds)
        unknown = ks - TASK_KINDS
        if unknown:
            msg = f"Unknown task kinds: {sorted(unknown)}; allowed: {sorted(TASK_KINDS)}"
            raise ValueError(msg)
        return ModelSpec(cls=cls, kinds=ks, factory=factory)
    if value in _REGISTRY:
        return _REGISTRY[value]
    if hasattr(value, "task_kinds"):
        return ModelSpec(cls=value, kinds=frozenset(value.task_kinds))
    if hasattr(value, "task_kind"):
        return ModelSpec(cls=value, kinds=frozenset({value.task_kind}))
    return ModelSpec(cls=value, kinds=frozenset())


# --------------------------------------------------------------------------- #
# Curated benchmark tasks per category
# --------------------------------------------------------------------------- #


BENCHMARK_TASKS: dict[str, dict[str, list[Task]]] = {
    "combinatorial": {
        "link_pred": [
            Task(
                "random_tsp",
                "link_pred",
                lambda root, seed=0: combinatorial.random_tsp(root, num_graphs=1, num_nodes=200, k=4, seed=seed),
                epochs=80,
            ),
            Task(
                "random_coloring",
                "link_pred",
                lambda root, seed=0: combinatorial.random_coloring(
                    root, num_graphs=1, num_nodes=200, edge_prob=0.1, seed=seed
                ),
                epochs=80,
            ),
        ],
    },
    "biology": {
        "graph_cls": [
            Task("mutag", "graph_cls", biology.mutag, epochs=40),
            Task("proteins", "graph_cls", biology.proteins, epochs=20),
        ],
        "link_pred": [
            Task("celegans", "link_pred", biology.celegans, epochs=80),
        ],
    },
    "social": {
        "node_cls": [
            Task("cora", "node_cls", social.cora, epochs=100),
            Task("citeseer", "node_cls", social.citeseer, epochs=100),
            Task("pubmed", "node_cls", social.pubmed, epochs=100),
            Task("roman_empire", "node_cls", social.roman_empire, epochs=80),
            Task("minesweeper", "node_cls", social.minesweeper, epochs=80),
        ],
        "link_pred": [
            Task("cora_link_pred", "link_pred", social.cora, epochs=80),
            Task("citeseer_link_pred", "link_pred", social.citeseer, epochs=80),
        ],
    },
    "knowledge": {
        "link_pred": [
            Task("fb15k_237", "link_pred", knowledge.fb15k_237, epochs=20),
            Task("wordnet18rr", "link_pred", knowledge.wordnet18rr, epochs=20),
        ],
    },
    "infrastructure": {
        "link_pred": [
            Task("power_grid", "link_pred", infrastructure.power_grid, epochs=80),
            Task("euroroad", "link_pred", infrastructure.euroroad, epochs=80),
        ],
    },
    "finance": {
        "link_pred": [
            Task("product_space", "link_pred", finance.product_space, epochs=80),
            Task("board_directors", "link_pred", finance.board_directors, epochs=40),
        ],
    },
    "computing": {
        "link_pred": [
            Task("internet_as", "link_pred", lambda root: computing.internet_as(root), epochs=40),
            Task("topology", "link_pred", computing.topology, epochs=10),
        ],
    },
    "vision": {
        "graph_cls": [
            Task(
                "mnist_superpixels",
                "graph_cls",
                lambda root: vision.mnist_superpixels(root)[:1500],
                epochs=4,
            ),
        ],
    },
    "physics": {
        "graph_reg": [
            Task(
                "zinc",
                "graph_reg",
                lambda root: (
                    physics.zinc(root, subset=True, split="train"),
                    physics.zinc(root, subset=True, split="val"),
                ),
                epochs=10,
            ),
        ],
        "link_pred": [
            Task(
                "ising_lattice",
                "link_pred",
                lambda root, seed=0: physics.ising_lattice(root, num_graphs=1, side=20, seed=seed),
                epochs=60,
            ),
        ],
    },
    "security": {
        "link_pred": [
            Task("terrorists_911", "link_pred", security.terrorists_911, epochs=120),
        ],
    },
}

if _HAS_OGB:
    # OGB tasks live in the domain modules; we only register them as
    # benchmark tasks when the ``ogb`` extra is importable so the
    # curated catalogue stays runnable without it.
    BENCHMARK_TASKS["social"]["node_cls"].append(
        Task("ogbn_arxiv", "node_cls", social.ogbn_arxiv, epochs=50),
    )
    BENCHMARK_TASKS["social"]["link_pred"].append(
        Task("ogbl_collab", "link_pred", social.ogbl_collab, epochs=20),
    )
    BENCHMARK_TASKS["finance"].setdefault("node_cls", []).append(
        Task("ogbn_products", "node_cls", finance.ogbn_products, epochs=20),
    )
    BENCHMARK_TASKS["biology"]["graph_cls"].extend([
        Task("ogbg_molhiv", "graph_cls", biology.ogbg_molhiv, epochs=20),
        Task("ogbg_molpcba", "graph_cls", biology.ogbg_molpcba, epochs=20),
    ])


def iter_benchmark_tasks(
    category: str | None = None,
    kind: str | None = None,
) -> list[Task]:
    """Flatten ``BENCHMARK_TASKS`` to a list, optionally filtered by category/kind.

    Examples
    --------
    >>> [t.name for t in iter_benchmark_tasks(category="biology", kind="graph_cls")]
    ['mutag', 'proteins']
    """
    cats = [category] if category is not None else list(BENCHMARK_TASKS)
    out: list[Task] = []
    for c in cats:
        per_cat = BENCHMARK_TASKS.get(c, {})
        kinds = [kind] if kind is not None else list(per_cat)
        for k in kinds:
            out.extend(per_cat.get(k, []))
    return out


# --------------------------------------------------------------------------- #
# Custom-dataset helpers
# --------------------------------------------------------------------------- #


def task_from_dataset(
    name: str,
    kind: str,
    dataset: Any,
    *,
    epochs: int = 30,
) -> Task:
    """Wrap an already-loaded dataset as a :class:`Task`.

    The dataset must satisfy the conventions for ``kind``: a PyG dataset or
    any object exposing ``ds[0]`` plus the relevant attributes (``num_features``
    / ``num_classes`` / ``num_relations``). The benchmark dispatcher caches
    the dataset, so the same instance is reused across seeds without
    reloading.
    """
    if kind not in TASK_KINDS:
        msg = f"Unknown task kind {kind!r}; choices: {sorted(TASK_KINDS)}"
        raise ValueError(msg)
    return Task(name=name, kind=kind, loader=lambda _root: dataset, epochs=epochs)


def register_task(category: str, task: Task) -> None:
    """Register ``task`` under ``category`` in :data:`BENCHMARK_TASKS`.

    The task becomes visible to ``run_benchmark(category)`` and to
    :func:`iter_benchmark_tasks`. Use :func:`unregister_task` to remove it
    (e.g. in ``tearDown`` of a test).
    """
    if not isinstance(task, Task):
        msg = f"task must be a Task, got {type(task).__name__}"
        raise TypeError(msg)
    if task.kind not in TASK_KINDS:
        msg = f"Task {task.name!r} has unknown kind {task.kind!r}; choices: {sorted(TASK_KINDS)}"
        raise ValueError(msg)
    per_cat = BENCHMARK_TASKS.setdefault(category, {})
    per_kind = per_cat.setdefault(task.kind, [])
    if any(t.name == task.name for t in per_kind):
        msg = f"Task {task.name!r} already registered in category {category!r}/{task.kind!r}"
        raise ValueError(msg)
    per_kind.append(task)


def unregister_task(category: str, name: str) -> Task | None:
    """Remove a previously registered task; returns it, or ``None`` if absent."""
    per_cat = BENCHMARK_TASKS.get(category, {})
    for kind_tasks in per_cat.values():
        for i, t in enumerate(kind_tasks):
            if t.name == name:
                return kind_tasks.pop(i)
    return None


# --------------------------------------------------------------------------- #
# Statistical helpers
# --------------------------------------------------------------------------- #


def _ci_half_width(values: np.ndarray, ci: float = 0.95) -> float:
    """Half-width of a t-distribution confidence interval for the mean."""
    n = values.size
    if n < 2:
        return 0.0
    sem = stats.sem(values)
    return float(sem * stats.t.ppf((1 + ci) / 2, n - 1))


def _bootstrap_ci_half_width(
    values: np.ndarray,
    ci: float = 0.95,
    n_resamples: int = 10000,
    random_state: int = 0,
) -> float:
    """Half-width of a percentile-bootstrap CI for the mean.

    Robust for non-Gaussian metrics (e.g. Hits@K, MRR, AUC) where the
    Student's-t assumption is poor. Returns ``(hi - lo) / 2`` -- the
    half-width of a symmetric envelope with the same total width as the
    percentile interval, so callers reporting ``mean ± half`` recover
    the bootstrap interval's spread without inflating asymmetric tails.
    """
    arr = np.asarray(values, dtype=float).ravel()
    n = arr.size
    if n < 2:
        return 0.0
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float((hi - lo) / 2.0)


def _resolve_ci_half(
    values: np.ndarray,
    ci: float,
    method: str,
    n_resamples: int,
    random_state: int,
) -> float:
    if method == "t":
        return _ci_half_width(values, ci)
    if method == "bootstrap":
        return _bootstrap_ci_half_width(values, ci, n_resamples, random_state)
    msg = f"Unknown CI method: {method!r}; choices: 't', 'bootstrap'"
    raise ValueError(msg)


def _paired_pvalue(a: np.ndarray, b: np.ndarray, method: str) -> float:
    """p-value of a paired test between two seed-aligned metric arrays.

    ``method="t"`` is the paired Student's t-test (parametric). ``method=
    "wilcoxon"`` is the Wilcoxon signed-rank test on the paired
    differences -- recommended at small seed counts where the paired
    t-test's normality assumption is most fragile (Benavoli et al.,
    JMLR 2016).
    """
    if a.size < 2 or b.size < 2 or a.size != b.size:
        return float("nan")
    if method == "t":
        return float(stats.ttest_rel(a, b).pvalue)
    if method == "wilcoxon":
        diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
        # All-zero paired differences -> the signed-rank statistic has no
        # ranks to assign; return NaN so the row is reported as undefined
        # rather than as an artificial 1.0.
        if not np.any(diffs != 0):
            return float("nan")
        try:
            return float(stats.wilcoxon(diffs, zero_method="wilcox").pvalue)
        except ValueError:
            return float("nan")
    msg = f"Unknown pairwise method: {method!r}; choices: 't', 'wilcoxon'"
    raise ValueError(msg)


def _holm_correction(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (max-monotone).

    NaN inputs (e.g. tests that were undefined for that pair) are
    excluded from the rank table and propagated as NaN in the output;
    they are *not* counted toward the family size, so the remaining
    valid tests retain their proper power.
    """
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return p
    valid = ~np.isnan(p)
    n_valid = int(valid.sum())
    adjusted = np.full(n, np.nan, dtype=float)
    if n_valid == 0:
        return adjusted
    valid_idx = np.where(valid)[0]
    p_valid = p[valid_idx]
    order = np.argsort(p_valid)
    running = 0.0
    out_valid = np.empty(n_valid, dtype=float)
    for rank, idx in enumerate(order):
        adj = float(min(p_valid[idx] * (n_valid - rank), 1.0))
        running = max(running, adj)
        out_valid[idx] = running
    adjusted[valid_idx] = out_valid
    return adjusted


def _auto_metric_key(history: Mapping[str, Any]) -> str:
    for key in _METRIC_KEYS:
        if key in history:
            return key
    return next(iter(history))


def _final_metric(history: Mapping[str, list[float]]) -> tuple[str, float]:
    key = _auto_metric_key(history)
    return key, history[key][-1]


# --------------------------------------------------------------------------- #
# Task runner
# --------------------------------------------------------------------------- #


def _run_task(
    task: Task,
    ds: Any,
    spec: ModelSpec,
    hidden: int,
    epochs: int,
    verbose: bool = False,
) -> dict[str, list[float]]:
    if task.kind == "node_cls":
        data = ds[0]
        model = spec.build(ds.num_features, hidden, ds.num_classes, kind="node_cls")
        return train_node_classification(model, data, epochs=epochs, verbose=verbose)

    if task.kind == "graph_cls":
        shuffled = ds.shuffle()
        split = int(0.8 * len(shuffled))
        train_loader = DataLoader(shuffled[:split], batch_size=32, shuffle=True)
        val_loader = DataLoader(shuffled[split:], batch_size=32)
        model = spec.build(shuffled.num_features, hidden, shuffled.num_classes, kind="graph_cls")
        return train_graph_classification(model, train_loader, val_loader, epochs=epochs, verbose=verbose)

    if task.kind == "graph_reg":
        # Loader may return either a single dataset (used for both train and
        # held-out -- e.g. synthetic tasks with no canonical split) or a
        # ``(train_ds, val_ds)`` tuple (real benchmarks like ZINC).
        if isinstance(ds, tuple):
            train_ds, val_ds = ds
        else:
            train_ds = val_ds = ds
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64)
        inner = spec.build(hidden, hidden, 1, kind="graph_reg")

        class _AtomEmbed(torch.nn.Module):
            def __init__(self, num_atoms: int = 32) -> None:
                super().__init__()
                self.embed = torch.nn.Embedding(num_atoms, hidden)
                self.inner = inner

            def forward(self, batch: Any) -> torch.Tensor:
                # Embed only integer atom-type ids (e.g. ZINC); pass-through
                # any float feature matrix unchanged so we never silently
                # truncate continuous features via .long().
                if not batch.x.dtype.is_floating_point:
                    batch = batch.clone()
                    batch.x = self.embed(batch.x.view(-1).long())
                return self.inner(batch)

        return train_graph_regression(_AtomEmbed(), train_loader, val_loader, epochs=epochs, verbose=verbose)

    if task.kind == "link_pred":
        import math

        from torch_geometric.transforms import RandomLinkSplit
        from torch_geometric.utils import degree

        data = ds[0]

        def _fabricate_log_degree_features(d: Data, edge_index: torch.Tensor) -> Data:
            """Build a 3-D log-degree feature from `edge_index` only.

            Used as the fallback when a loader ships no node features. We
            keep the source of degree restricted to the caller-supplied
            `edge_index` so val/test edges never bleed into the feature
            matrix at training time.
            """
            n = int(d.num_nodes)
            deg = degree(edge_index[0], num_nodes=n, dtype=torch.float)
            log_deg = torch.log1p(deg) / math.log(max(n, 2))
            ones = torch.ones(n)
            out = d.clone()
            out.x = torch.stack([log_deg, log_deg.pow(2), ones], dim=1)
            return out

        # Relational link prediction (knowledge graphs with edge_type) is
        # detected on the *raw* data because PyG's RelLinkPredDataset
        # already restricts edge_index to training edges -- we only
        # fabricate features when missing, using train_edge_index.
        if hasattr(data, "edge_type") and hasattr(data, "train_edge_index"):
            from graphnetz.models._adapters import RelationalLinkPredWrapper
            from graphnetz.training import train_relational_link_prediction

            if getattr(data, "x", None) is None:
                data = _fabricate_log_degree_features(data, data.train_edge_index)

            num_relations = ds.num_relations if hasattr(ds, "num_relations") else int(data.edge_type.max()) + 1
            built = spec.build(data.num_features, hidden, hidden, kind="link_pred")
            # spec.build returns a LinkPredWrapper for kind="link_pred"; unwrap it
            # so RelationalLinkPredWrapper can drive the bare encoder directly
            # (otherwise its forward expects data.edge_label_index).
            from typing import cast

            encoder = cast(
                torch.nn.Module,
                built.encoder if hasattr(built, "encoder") else built,
            )
            model = RelationalLinkPredWrapper(encoder, hidden, num_relations)

            # Create separate Data objects for train/val/test splits
            train_split = Data(
                x=data.x, edge_index=data.train_edge_index, edge_type=data.train_edge_type, num_nodes=data.num_nodes
            )
            val_split = Data(
                x=data.x, edge_index=data.valid_edge_index, edge_type=data.valid_edge_type, num_nodes=data.num_nodes
            )
            test_split = Data(
                x=data.x, edge_index=data.test_edge_index, edge_type=data.test_edge_type, num_nodes=data.num_nodes
            )

            return train_relational_link_prediction(
                model,
                train_split,
                val_split,
                test_split,
                epochs=epochs,
                verbose=verbose,
            )

        # Detect graph direction from the data itself instead of forcing
        # ``is_undirected=True`` -- on a directed graph the latter silently
        # de-duplicates reciprocal edges and halves the supervision signal.
        is_undirected = not bool(data.is_directed())
        transform = RandomLinkSplit(
            num_val=0.05,
            num_test=0.10,
            is_undirected=is_undirected,
            add_negative_train_samples=True,
            neg_sampling_ratio=1.0,
        )
        train_data, val_data, test_data = transform(data)
        # Fabricate features *after* the split so val/test edges never
        # leak into the node features the encoder consumes.  Use only the
        # training message-passing edges (edge_index, not edge_label_index)
        # for the degree statistic.
        if getattr(train_data, "x", None) is None:
            train_data = _fabricate_log_degree_features(train_data, train_data.edge_index)
            val_data = val_data.clone()
            val_data.x = train_data.x
            test_data = test_data.clone()
            test_data.x = train_data.x
        # ``spec.build(kind="link_pred")`` returns a LinkPredWrapper, which
        # satisfies the ``_LinkPredLike`` protocol of the trainer; mypy
        # only sees the declared ``Module`` return so we narrow here.
        from typing import cast as _cast

        from graphnetz.training import _LinkPredLike

        lp_model = _cast(_LinkPredLike, spec.build(train_data.num_features, hidden, hidden, kind="link_pred"))
        return train_link_prediction(lp_model, train_data, val_data, test_data, epochs=epochs, verbose=verbose)

    msg = f"Unknown task kind: {task.kind}"
    raise ValueError(msg)


# --------------------------------------------------------------------------- #
# Benchmark report
# --------------------------------------------------------------------------- #


@dataclass
class BenchmarkReport:
    """Structured outcome of a multi-seed benchmark run.

    ``histories[task][model]`` is a list with one history dict per seed (in
    seed order). The report is also a read-only mapping ``task -> {model:
    history_seed_0}`` for backward compatibility with single-seed callers.
    """

    seeds: tuple[int, ...]
    histories: dict[str, dict[str, list[dict[str, list[float]]]]]
    config: dict[str, Any] = field(default_factory=dict)
    ci_method: str = "t"
    bootstrap_n: int = 10000
    bootstrap_seed: int = 0
    pairwise_method: str = "t"

    def _ci_half(
        self,
        values: np.ndarray,
        ci: float,
        method: str | None = None,
    ) -> float:
        return _resolve_ci_half(
            values,
            ci,
            method or self.ci_method,
            self.bootstrap_n,
            self.bootstrap_seed,
        )

    # ----- Pickle compatibility ---------------------------------------------

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore from pickle, backfilling fields added since serialisation.

        Older :class:`BenchmarkReport` pickles predate the ``ci_method`` /
        ``bootstrap_*`` / ``pairwise_method`` fields. ``__setstate__``
        ensures they load cleanly with sensible defaults so the experiment
        cache (``_cache_*.pkl``) survives library upgrades.
        """
        self.__dict__.update(state)
        self.__dict__.setdefault("ci_method", "t")
        self.__dict__.setdefault("bootstrap_n", 10000)
        self.__dict__.setdefault("bootstrap_seed", 0)
        self.__dict__.setdefault("pairwise_method", "t")
        self.__dict__.setdefault("config", {})

    # ----- Mapping protocol (backward compat with the legacy dict shape) -----

    def __iter__(self):
        return iter(self.histories)

    def __len__(self) -> int:
        return len(self.histories)

    def __getitem__(self, task: str) -> dict[str, dict[str, list[float]]]:
        per_task = self.histories[task]
        return {model: per_task[model][0] for model in per_task}

    def items(self):
        for task in self.histories:
            yield task, self[task]

    def keys(self):
        return self.histories.keys()

    def values(self):
        return [self[task] for task in self.histories]

    # ----- Statistics --------------------------------------------------------

    def final_metrics(self, key: str | None = None) -> dict[str, dict[str, list[float]]]:
        """Final metric value per (task, model, seed)."""
        out: dict[str, dict[str, list[float]]] = {}
        for task, per_task in self.histories.items():
            out[task] = {}
            for model, seed_histories in per_task.items():
                vals: list[float] = []
                for h in seed_histories:
                    k = key or _auto_metric_key(h)
                    vals.append(float(h[k][-1]))
                out[task][model] = vals
        return out

    def metric_name(self) -> str:
        for per_task in self.histories.values():
            for seed_histories in per_task.values():
                if seed_histories:
                    return _auto_metric_key(seed_histories[0])
        return "metric"

    def summary(self, ci: float = 0.95, method: str | None = None) -> pd.DataFrame:
        """Per-(task, model) mean, std, sem, CI half-width and bounds.

        ``method`` overrides ``self.ci_method`` for this call only; choose
        ``"t"`` for Student's-t intervals (default) or ``"bootstrap"`` for
        percentile-bootstrap intervals (better for non-Gaussian metrics
        such as Hits@K, MRR, or AUC).
        """
        rows = []
        for task, per_task in self.final_metrics().items():
            for model, values in per_task.items():
                arr = np.asarray(values, dtype=float)
                mean = float(arr.mean())
                std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
                sem = float(stats.sem(arr)) if arr.size > 1 else 0.0
                half = self._ci_half(arr, ci, method=method)
                rows.append(
                    {
                        "task": task,
                        "model": model,
                        "n_seeds": arr.size,
                        "mean": mean,
                        "std": std,
                        "sem": sem,
                        "ci_low": mean - half,
                        "ci_high": mean + half,
                    }
                )
        return pd.DataFrame(rows).set_index(["task", "model"]).sort_index()

    def pairwise(self, alpha: float = 0.05, method: str | None = None) -> pd.DataFrame:
        """Paired pairwise tests between models per task with Holm adjustment.

        ``method`` overrides ``self.pairwise_method`` for this call only:

        - ``"t"`` (default) -- paired Student's t-test on per-seed final metrics.
        - ``"wilcoxon"`` -- non-parametric Wilcoxon signed-rank test on the
          paired differences. Recommended at small seed counts where the
          paired t-test's normality assumption is most fragile; see
          Benavoli et al., *JMLR* 17(5):1-36, 2016.
        """
        finals = self.final_metrics()
        test = method or self.pairwise_method
        rows = []
        for task, per_task in finals.items():
            models = sorted(per_task)
            pairs: list[tuple[str, str, float, float]] = []
            ps: list[float] = []
            for i, model_a in enumerate(models):
                for model_b in models[i + 1 :]:
                    a = np.asarray(per_task[model_a], dtype=float)
                    b = np.asarray(per_task[model_b], dtype=float)
                    p = _paired_pvalue(a, b, test)
                    pairs.append((model_a, model_b, float(a.mean() - b.mean()), p))
                    ps.append(p)
            adj = _holm_correction(np.asarray(ps, dtype=float))
            for (model_a, model_b, diff, p_raw), p_holm in zip(pairs, adj, strict=False):
                rows.append(
                    {
                        "task": task,
                        "model_a": model_a,
                        "model_b": model_b,
                        "mean_diff": diff,
                        "p_raw": p_raw,
                        "p_holm": p_holm,
                        "significant": (not np.isnan(p_holm)) and p_holm < alpha,
                    }
                )
        return pd.DataFrame(rows)

    def friedman(self, alpha: float = 0.05) -> dict[str, float | int | bool]:
        r"""Friedman omnibus test on per-task ranks of seed-mean metrics.

        Returns a dict with the statistic ``chi2``, the asymptotic
        $\chi^2_{k-1}$ p-value, the rejection flag at ``alpha``, and the
        $(k, N)$ shape used. The Nemenyi post-hoc surfaced in
        :meth:`plot_critical_difference` should only be interpreted when
        ``rejected`` is true (Demšar, 2006).

        Only models present in every task are included; per-task ranks
        use the metric direction (lower-is-better for ``val_mae`` and
        ``train_loss``).
        """
        finals = self.final_metrics()
        if not finals:
            return {"chi2": float("nan"), "p_value": float("nan"), "k": 0, "n": 0, "rejected": False}
        common: set[str] = set.intersection(*[set(per.keys()) for per in finals.values()])
        if len(common) < 2 or len(finals) < 2:
            return {
                "chi2": float("nan"),
                "p_value": float("nan"),
                "k": len(common),
                "n": len(finals),
                "rejected": False,
            }
        models = sorted(common)
        tasks = sorted(finals)
        means = np.array([[float(np.mean(finals[t][m])) for m in models] for t in tasks])
        rows: list[np.ndarray] = []
        for i, task in enumerate(tasks):
            sample = next(iter(self.histories[task].values()))[0]
            sign = 1.0 if _auto_metric_key(sample) in _LOWER_IS_BETTER else -1.0
            rows.append(stats.rankdata(sign * means[i], method="average"))
        ranks = np.array(rows)
        k = len(models)
        n = len(tasks)
        avg = ranks.mean(axis=0)
        chi2 = (12.0 * n) / (k * (k + 1)) * (float(np.sum(avg**2)) - k * (k + 1) ** 2 / 4.0)
        p = float(stats.chi2.sf(chi2, df=k - 1))
        return {"chi2": float(chi2), "p_value": p, "k": k, "n": n, "rejected": bool(p < alpha)}

    # ----- Reporting helpers -------------------------------------------------

    def _best_per_task(self) -> dict[str, str]:
        finals = self.final_metrics()
        metric = self.metric_name()
        lower_is_better = metric in _LOWER_IS_BETTER
        best: dict[str, str] = {}
        for task, per_task in finals.items():
            scored = [(model, float(np.mean(values))) for model, values in per_task.items()]
            if lower_is_better:
                best[task] = min(scored, key=lambda x: x[1])[0]
            else:
                best[task] = max(scored, key=lambda x: x[1])[0]
        return best

    def to_latex(
        self,
        path: str | Path,
        *,
        ci: float = 0.95,
        bold_best: bool = True,
        pretty_tasks: Mapping[str, str] | None = None,
        caption: str | None = None,
        label: str | None = None,
        method: str | None = None,
    ) -> Path:
        """Booktabs LaTeX table of mean ± CI half-width with bold-best per task.

        ``method`` overrides ``self.ci_method`` (``"t"`` or ``"bootstrap"``).
        """
        finals = self.final_metrics()
        tasks = sorted(finals)
        models = sorted({m for per in finals.values() for m in per})
        best = self._best_per_task() if bold_best else {}
        pretty = dict(pretty_tasks or {})

        lines: list[str] = []
        if caption is not None or label is not None:
            lines.extend([r"\begin{table}[t]", r"  \centering"])
            if caption is not None:
                lines.append(rf"  \caption{{{caption}}}")
            if label is not None:
                lines.append(rf"  \label{{{label}}}")
        lines.append(r"\begin{tabular}{l" + "c" * len(tasks) + "}")
        lines.append(r"\toprule")
        header = "Model & " + " & ".join(pretty.get(t, t) for t in tasks) + r" \\"
        lines.append(header)
        lines.append(r"\midrule")
        for model in models:
            cells = []
            for task in tasks:
                values = np.asarray(finals[task].get(model, []), dtype=float)
                if values.size == 0:
                    cells.append("--")
                    continue
                mean = float(values.mean())
                half = self._ci_half(values, ci, method=method)
                if bold_best and best.get(task) == model:
                    cell = rf"$\mathbf{{{mean:.3f} \pm {half:.3f}}}$"
                else:
                    cell = rf"${mean:.3f} \pm {half:.3f}$"
                cells.append(cell)
            lines.append(f"{model} & " + " & ".join(cells) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        if caption is not None or label is not None:
            lines.append(r"\end{table}")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n")
        return out

    def pairwise_to_latex(
        self,
        path: str | Path,
        *,
        alpha: float = 0.05,
        caption: str | None = None,
        label: str | None = None,
        method: str | None = None,
    ) -> Path:
        """LaTeX booktabs table of pairwise Holm-adjusted p-values.

        ``method`` overrides ``self.pairwise_method`` (``"t"`` or
        ``"wilcoxon"``) for this call only.
        """
        df = self.pairwise(alpha=alpha, method=method)
        lines: list[str] = []
        if caption is not None or label is not None:
            lines.extend([r"\begin{table}[t]", r"  \centering"])
            if caption is not None:
                lines.append(rf"  \caption{{{caption}}}")
            if label is not None:
                lines.append(rf"  \label{{{label}}}")
        lines.append(r"\begin{tabular}{llcccl}")
        lines.append(r"\toprule")
        lines.append(r"Task & Comparison & $\Delta\mu$ & $p_{\text{raw}}$ & $p_{\text{Holm}}$ & Sig. \\")
        lines.append(r"\midrule")
        for _, row in df.iterrows():
            sig = r"\textbf{*}" if row["significant"] else ""
            p_raw = "n/a" if pd.isna(row["p_raw"]) else f"{row['p_raw']:.3g}"
            p_holm = "n/a" if pd.isna(row["p_holm"]) else f"{row['p_holm']:.3g}"
            lines.append(
                f"{row['task']} & {row['model_a']} vs.\\ {row['model_b']} & "
                f"${row['mean_diff']:+.3f}$ & {p_raw} & {p_holm} & {sig} \\\\"
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        if caption is not None or label is not None:
            lines.append(r"\end{table}")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n")
        return out

    # ----- Plotting ----------------------------------------------------------

    def plot(
        self,
        ax: plt.Axes | None = None,
        *,
        ci: float = 0.95,
        ylabel: str | None = None,
        title: str | None = None,
        annotate: bool = True,
        pretty_tasks: Mapping[str, str] | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Grouped bar chart of mean ± CI half-width across seeds."""
        finals = self.final_metrics()
        pretty = dict(pretty_tasks or {})
        values: dict[str, dict[str, float]] = {}
        errors: dict[str, dict[str, float]] = {}
        for task, per_task in finals.items():
            label = pretty.get(task, task)
            values[label] = {}
            errors[label] = {}
            for model, vals in per_task.items():
                arr = np.asarray(vals, dtype=float)
                values[label][model] = float(arr.mean())
                errors[label][model] = self._ci_half(arr, ci)
        return plot_grouped_bars(
            values,
            errors=errors,
            ax=ax,
            title=title,
            ylabel=ylabel or self.metric_name(),
            annotate=annotate,
        )

    def plot_forest(
        self,
        ax: plt.Axes | None = None,
        *,
        ci: float = 0.95,
        pretty_tasks: Mapping[str, str] | None = None,
        xlabel: str | None = None,
        height_per_task: float = 0.42,
        sort_within: bool = False,
        band: bool = True,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Forest plot, one row per task with models jittered within the row.

        Height scales with the number of *tasks* only -- adding more models
        widens the within-row jitter rather than adding new rows -- so the
        figure stays compact for many models.

        ``sort_within=True`` orders the jittered positions per-task so the
        best mean lands at the top of the row (helps spot leaders when there
        are many models).  Each model keeps a stable colour across tasks.

        ``band=True`` shades alternating task rows (banded reading aid).
        """
        from graphnetz.plotting import COLUMN_INCHES

        set_plot_style()
        finals = self.final_metrics()
        tasks = sorted(finals)
        models = sorted({m for per in finals.values() for m in per})
        pretty = dict(pretty_tasks or {})
        n_tasks = len(tasks)
        n_models = len(models)
        metric = self.metric_name()
        lower_is_better = metric in _LOWER_IS_BETTER

        if ax is None:
            height = max(1.6, height_per_task * n_tasks + 1.0)
            fig, ax = plt.subplots(figsize=(COLUMN_INCHES["single"] * 1.05, height))
        else:
            fig = ax.figure  # type: ignore[assignment]

        jitter_span = 0.7
        slot_positions = (
            np.linspace(-jitter_span / 2, jitter_span / 2, n_models) if n_models > 1 else np.zeros(n_models)
        )

        # Precompute per-task offsets (mapping model_index -> within-row offset).
        per_task_offset: dict[str, dict[str, float]] = {}
        for task in tasks:
            present = [m for m in models if m in finals[task]]
            if sort_within and len(present) > 1:
                means = np.array([float(np.mean(finals[task][m])) for m in present])
                order = np.argsort(means if lower_is_better else -means)
                ordered = [present[i] for i in order]
            else:
                ordered = present
            row_offsets = (
                np.linspace(-jitter_span / 2, jitter_span / 2, len(ordered))
                if len(ordered) > 1
                else np.zeros(len(ordered))
            )
            per_task_offset[task] = dict(zip(ordered, row_offsets, strict=False))

        if band:
            for i in range(n_tasks):
                if i % 2 == 0:
                    ax.axhspan(
                        i - 0.5,
                        i + 0.5,
                        facecolor="0.96",
                        edgecolor="none",
                        zorder=0,
                    )

        for j, model in enumerate(models):
            xs: list[float] = []
            ys: list[float] = []
            errs: list[float] = []
            for i, task in enumerate(tasks):
                if model not in finals[task]:
                    continue
                arr = np.asarray(finals[task][model], dtype=float)
                xs.append(float(arr.mean()))
                offset = per_task_offset[task].get(model, slot_positions[j])
                ys.append(i + offset)
                errs.append(self._ci_half(arr, ci))
            if xs:
                color = NATURE_COLORS[j % len(NATURE_COLORS)]
                ax.errorbar(
                    xs,
                    ys,
                    xerr=[errs, errs],
                    fmt="o",
                    color=color,
                    ecolor=color,
                    elinewidth=1.0,
                    capsize=2.0,
                    markersize=3.5,
                    label=model,
                    zorder=3,
                )

        for i in range(n_tasks - 1):
            ax.axhline(i + 0.5, color="0.85", linewidth=0.3, zorder=1)

        ax.set_yticks(range(n_tasks))
        ax.set_yticklabels([pretty.get(t, t) for t in tasks])
        ax.set_ylim(n_tasks - 0.5, -0.5)
        ax.set_xlabel(xlabel or metric)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, linewidth=0.3, alpha=0.4, zorder=1)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(n_models, 4),
            frameon=False,
            handlelength=1.2,
            handletextpad=0.4,
            columnspacing=1.0,
        )
        fig.tight_layout()
        return fig, ax

    def plot_pairwise(
        self,
        ax: plt.Axes | None = None,
        *,
        ci: float = 0.95,
        alpha: float = 0.05,
        pretty_tasks: Mapping[str, str] | None = None,
        layout: str = "matrix",
        max_cols: int = 3,
        method: str | None = None,
    ) -> tuple[plt.Figure, Any]:
        """Pairwise comparison plot, with two layouts that scale differently.

        ``layout="matrix"`` (default) -- one significance heatmap per task,
        with the lower triangle holding $-\\log_{10}(p_{\\text{Holm}})$ and the
        upper triangle holding the signed mean difference.  Scales to many
        models and many tasks (panels arranged in a grid with at most
        ``max_cols`` columns).

        ``layout="list"`` -- one row per pairwise comparison with CI whiskers
        and a significance marker.  Best for small numbers of comparisons.

        ``method`` overrides ``self.pairwise_method`` (``"t"`` or
        ``"wilcoxon"``) for this call only.
        """
        if layout == "list":
            return self._plot_pairwise_list(ax=ax, ci=ci, alpha=alpha, pretty_tasks=pretty_tasks, method=method)
        if layout == "matrix":
            return self._plot_pairwise_matrix(
                ci=ci, alpha=alpha, pretty_tasks=pretty_tasks, max_cols=max_cols, method=method
            )
        msg = f"Unknown pairwise layout: {layout!r}; choices: 'matrix', 'list'"
        raise ValueError(msg)

    def _plot_pairwise_matrix(
        self,
        *,
        ci: float = 0.95,
        alpha: float = 0.05,
        pretty_tasks: Mapping[str, str] | None = None,
        max_cols: int = 3,
        method: str | None = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        from matplotlib.colors import TwoSlopeNorm

        from graphnetz.plotting import COLUMN_INCHES

        set_plot_style()
        finals = self.final_metrics()
        df = self.pairwise(alpha=alpha, method=method)
        tasks = sorted(finals)
        pretty = dict(pretty_tasks or {})
        n_tasks = len(tasks)

        # Per-task model lists (intersection used for matrix axes).
        per_task_models = {t: sorted(finals[t]) for t in tasks}
        max_models = max((len(per_task_models[t]) for t in tasks), default=0)
        if max_models < 2:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "fewer than two models per task", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            return fig, np.array([[ax]])

        cols = max(1, min(max_cols, n_tasks))
        rows = (n_tasks + cols - 1) // cols
        fig_w = COLUMN_INCHES["double"] if cols > 1 else COLUMN_INCHES["single"]
        cell = max(0.42, 1.4 / max_models)
        fig_h = (cell * max_models + 1.4) * rows
        fig, axes_obj = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)

        diff_max = max(1e-9, df["mean_diff"].abs().max() if not df.empty else 1.0)
        norm = TwoSlopeNorm(vmin=-diff_max, vcenter=0, vmax=diff_max)

        for k, task in enumerate(tasks):
            r, c = divmod(k, cols)
            ax = axes_obj[r, c]
            models_t = per_task_models[task]
            n = len(models_t)
            mat = np.full((n, n), np.nan)  # lower: -log10(p), upper: mean diff
            sub = df[df["task"] == task] if not df.empty else df
            for _, row in sub.iterrows():
                ia = models_t.index(row["model_a"])
                ib = models_t.index(row["model_b"])
                if ia == ib:
                    continue
                lo, hi = (ia, ib) if ia < ib else (ib, ia)
                p_holm = row["p_holm"]
                if not np.isnan(p_holm):
                    mat[hi, lo] = -np.log10(max(p_holm, 1e-12))
                mat[lo, hi] = row["mean_diff"] if ia < ib else -row["mean_diff"]

            mask_lower = np.tri(n, n, -1, dtype=bool)
            mask_upper = mask_lower.T

            # Two passes: lower triangle (significance), upper triangle (effect).
            lower = np.where(mask_lower, mat, np.nan)
            upper = np.where(mask_upper, mat, np.nan)

            ax.imshow(lower, cmap="Greys", vmin=0.0, vmax=3.0, aspect="equal")
            ax.imshow(upper, cmap="RdBu_r", norm=norm, aspect="equal")

            # Annotate cells.
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if mask_lower[i, j]:
                        # significance cell: show p_holm
                        sub_match = sub[
                            ((sub["model_a"] == models_t[j]) & (sub["model_b"] == models_t[i]))
                            | ((sub["model_a"] == models_t[i]) & (sub["model_b"] == models_t[j]))
                        ]
                        if sub_match.empty:
                            continue
                        p = float(sub_match["p_holm"].iloc[0])
                        text = "n/a" if np.isnan(p) else f"{p:.2g}"
                        is_sig = (not np.isnan(p)) and p < alpha
                        if is_sig:
                            text += "*"
                        color = "white" if (not np.isnan(p) and -np.log10(max(p, 1e-12)) > 1.5) else "black"
                        weight = "bold" if is_sig else "normal"
                        ax.text(j, i, text, ha="center", va="center", fontsize=6, color=color, fontweight=weight)
                    elif mask_upper[i, j]:
                        d = mat[i, j]
                        if np.isnan(d):
                            continue
                        color = "white" if abs(d) > 0.6 * diff_max else "black"
                        ax.text(j, i, f"{d:+.2f}", ha="center", va="center", fontsize=6, color=color)

            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(models_t, rotation=30, ha="right")
            ax.set_yticklabels(models_t)
            ax.set_xticks([], minor=True)
            ax.set_yticks([], minor=True)
            ax.tick_params(which="both", length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(pretty.get(task, task))

        # Hide unused panels.
        for k in range(n_tasks, rows * cols):
            r, c = divmod(k, cols)
            axes_obj[r, c].axis("off")

        # Caption-style legend strip.
        fig.suptitle(
            r"lower: $-\log_{10}(p_{\mathrm{Holm}})$ (darker = more significant, $*$ = $p<\alpha$);"
            r"   upper: mean difference (row $-$ column, red $>0$, blue $<0$)",
            y=0.02,
            fontsize=7,
        )
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        return fig, axes_obj

    def _plot_pairwise_list(
        self,
        ax: plt.Axes | None = None,
        *,
        ci: float = 0.95,
        alpha: float = 0.05,
        pretty_tasks: Mapping[str, str] | None = None,
        method: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        from matplotlib.lines import Line2D

        from graphnetz.plotting import COLUMN_INCHES

        set_plot_style()
        finals = self.final_metrics()
        df = self.pairwise(alpha=alpha, method=method)
        if df.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "no pairwise comparisons", ha="center", va="center", transform=ax.transAxes)
            return fig, ax

        pretty = dict(pretty_tasks or {})
        rows: list[tuple[str, str, str, float, float, bool]] = []
        for _, row in df.iterrows():
            a = np.asarray(finals[row["task"]][row["model_a"]], dtype=float)
            b = np.asarray(finals[row["task"]][row["model_b"]], dtype=float)
            diff_per_seed = a - b
            mean = float(diff_per_seed.mean())
            half = self._ci_half(diff_per_seed, ci)
            rows.append(
                (
                    pretty.get(row["task"], row["task"]),
                    row["model_a"],
                    row["model_b"],
                    mean,
                    half,
                    bool(row["significant"]),
                )
            )

        if ax is None:
            fig, ax = plt.subplots(figsize=(COLUMN_INCHES["single"], 0.34 * len(rows) + 0.6))
        else:
            fig = ax.figure  # type: ignore[assignment]

        ytick_positions: list[float] = []
        ytick_labels: list[str] = []
        for i, (task_label, ma, mb, mean, half, sig) in enumerate(rows):
            color = NATURE_COLORS[0] if sig else NATURE_COLORS[3]
            ax.errorbar(
                mean,
                i,
                xerr=[[half], [half]],
                fmt="o" if sig else "s",
                color=color,
                ecolor=color,
                elinewidth=1.0,
                capsize=2.0,
                markersize=4.0 if sig else 3.0,
            )
            ytick_positions.append(i)
            ytick_labels.append(f"{task_label}: {ma} - {mb}")

        ax.axvline(0, color="0.4", linewidth=0.6, linestyle="--")
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(ytick_labels)
        ax.invert_yaxis()
        ax.set_xlabel(r"Mean difference (95% CI, paired)")
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, linewidth=0.3, alpha=0.4)

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color=NATURE_COLORS[0],
                linestyle="",
                markersize=4.0,
                label=rf"$p_{{\mathrm{{Holm}}}} < {alpha}$",
            ),
            Line2D([0], [0], marker="s", color=NATURE_COLORS[3], linestyle="", markersize=3.0, label="not significant"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            frameon=False,
            handlelength=1.2,
            handletextpad=0.4,
        )
        fig.tight_layout()
        return fig, ax

    def plot_critical_difference(
        self,
        *,
        alpha: float = 0.05,
        title: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        r"""Demšar critical-difference (CD) diagram.

        Computes mean ranks of every model across tasks and overlays the
        Nemenyi critical difference at level ``alpha``.  Models within
        ``CD`` of each other are joined by a thick horizontal "clique" bar
        (i.e., not significantly different).  This is the canonical
        scalable visualization for multi-method, multi-dataset benchmarks
        (Demšar, 2006).

        Only models present in *every* task are included.  Requires at
        least two tasks and at least two such models.
        """
        from scipy.stats import studentized_range

        from graphnetz.plotting import COLUMN_INCHES

        set_plot_style()
        finals = self.final_metrics()

        common: set[str] = set.intersection(*[set(per.keys()) for per in finals.values()]) if finals else set()
        if len(common) < 2 or len(finals) < 2:
            fig, ax = plt.subplots(figsize=(COLUMN_INCHES["single"], 1.6))
            ax.text(
                0.5,
                0.5,
                "CD diagram needs >= 2 tasks and >= 2 models common to all tasks",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )
            ax.axis("off")
            return fig, ax

        models = sorted(common)
        tasks = sorted(finals)
        means = np.array([[float(np.mean(finals[t][m])) for m in models] for t in tasks])
        # Direction (lower-is-better) is detected *per task* so the CD
        # diagram is correct on heterogeneous benchmarks where some tasks
        # use accuracy (higher better) and others use loss (lower better).
        rows: list[np.ndarray] = []
        for i, task in enumerate(tasks):
            sample = next(iter(self.histories[task].values()))[0]
            task_metric = _auto_metric_key(sample)
            sign = 1.0 if task_metric in _LOWER_IS_BETTER else -1.0
            rows.append(stats.rankdata(sign * means[i], method="average"))
        ranks = np.array(rows)
        avg_ranks = ranks.mean(axis=0)
        # Ranks are always lower-is-better by construction.

        k = len(models)
        n = len(tasks)
        # Friedman omnibus: only interpret Nemenyi after the global null is
        # rejected (Demšar, 2006). We compute it from the same rank table.
        avg_for_chi2 = ranks.mean(axis=0)
        chi2 = (12.0 * n) / (k * (k + 1)) * (float(np.sum(avg_for_chi2**2)) - k * (k + 1) ** 2 / 4.0)
        friedman_p = float(stats.chi2.sf(chi2, df=k - 1))
        friedman_rejected = friedman_p < alpha
        q = float(studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2))
        cd = q * float(np.sqrt(k * (k + 1) / (6 * n)))

        order = np.argsort(avg_ranks)
        sorted_models = [models[i] for i in order]
        sorted_ranks = avg_ranks[order]

        # Maximal cliques: contiguous runs in rank order whose span < CD.
        cliques_raw: list[tuple[int, int]] = []
        i = 0
        while i < k:
            j = i
            while j + 1 < k and sorted_ranks[j + 1] - sorted_ranks[i] < cd:
                j += 1
            if j > i:
                cliques_raw.append((i, j))
            i += 1
        cliques: list[tuple[int, int]] = []
        for a, b in sorted(set(cliques_raw)):
            if any(c <= a and b <= d for c, d in cliques):
                continue
            cliques = [(c, d) for c, d in cliques if not (a <= c and d <= b)]
            cliques.append((a, b))

        # Layout coordinates.
        fig_w = COLUMN_INCHES["double"]
        fig_h = max(2.2, 1.6 + 0.22 * k)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        rank_y = 0.0
        x_min, x_max = 1.0, float(k)
        ax.plot([x_min, x_max], [rank_y, rank_y], color="black", linewidth=0.8)
        for r in range(int(x_min), int(x_max) + 1):
            ax.plot([r, r], [rank_y, rank_y - 0.04], color="black", linewidth=0.6)
            ax.text(r, rank_y - 0.08, f"{r}", ha="center", va="top", fontsize=8)

        # Method leaders + side labels (left for top half, right for bottom half).
        half = (k + 1) // 2
        label_y_step = 0.16
        label_y_top = 0.32
        label_x_left = x_min - 0.5
        label_x_right = x_max + 0.5
        for idx, (model, r) in enumerate(zip(sorted_models, sorted_ranks, strict=False)):
            color = NATURE_COLORS[idx % len(NATURE_COLORS)]
            if idx < half:
                label_x = label_x_left
                ha = "right"
                ly = label_y_top + (half - idx - 1) * label_y_step
            else:
                label_x = label_x_right
                ha = "left"
                ly = label_y_top + (idx - half) * label_y_step
            ax.plot([r, r], [rank_y, ly], color="0.55", linewidth=0.5, zorder=1)
            ax.plot([r, label_x], [ly, ly], color="0.55", linewidth=0.5, zorder=1)
            ax.plot([r], [rank_y], marker="o", markersize=3.5, color=color, zorder=2)
            ax.text(
                label_x + (-0.05 if ha == "right" else 0.05),
                ly,
                f"{model} ({r:.2f})",
                va="center",
                ha=ha,
                fontsize=8,
                color=color,
            )

        # Clique bars below the rank axis (start below the tick labels).
        bar_y = rank_y - 0.16
        for a, b in cliques:
            ax.plot(
                [sorted_ranks[a] - 0.06, sorted_ranks[b] + 0.06],
                [bar_y, bar_y],
                color="black",
                linewidth=3.5,
                solid_capstyle="round",
                zorder=3,
            )
            bar_y -= 0.06

        # CD scale at the top.
        cd_y = label_y_top + max(half - 1, 0) * label_y_step + 0.22
        ax.plot([x_min, x_min + cd], [cd_y, cd_y], color="black", linewidth=1.0)
        ax.plot([x_min, x_min], [cd_y - 0.025, cd_y + 0.025], color="black", linewidth=1.0)
        ax.plot(
            [x_min + cd, x_min + cd],
            [cd_y - 0.025, cd_y + 0.025],
            color="black",
            linewidth=1.0,
        )
        ax.text(
            x_min + cd / 2,
            cd_y + 0.04,
            rf"CD = {cd:.3f} (Nemenyi, $\alpha={alpha}$, $k={k}$, $N={n}$)",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        friedman_color = "0.15" if friedman_rejected else "0.4"
        ax.text(
            x_min + cd / 2,
            cd_y + 0.18,
            rf"Friedman $\chi^2_{{{k - 1}}} = {chi2:.2f}$, $p = {friedman_p:.3g}$"
            + (" (reject)" if friedman_rejected else " (do not reject)"),
            ha="center",
            va="bottom",
            fontsize=7,
            color=friedman_color,
        )

        # Direction caption below all clique bars.
        caption_y = bar_y - 0.04
        ax.text(
            (x_min + x_max) / 2,
            caption_y,
            "Mean rank (lower rank = better)",
            ha="center",
            va="top",
            fontsize=8,
            color="0.3",
        )

        ax.set_xlim(label_x_left - 1.2, label_x_right + 1.2)
        ax.set_ylim(caption_y - 0.12, cd_y + 0.2)
        ax.axis("off")
        if title is not None:
            ax.set_title(title)
        fig.tight_layout()
        return fig, ax

    def plot_learning_curves(
        self,
        *,
        ci: float = 0.95,
        metric_key: str | None = None,
        pretty_tasks: Mapping[str, str] | None = None,
        ylabel: str = "Test accuracy",
        legend_loc: str = "lower right",
    ) -> tuple[plt.Figure, np.ndarray]:
        """Mean ± t-CI learning curves, one panel per task, sharing y-axis."""
        set_plot_style()
        from graphnetz.plotting import COLUMN_INCHES, panel_label

        tasks = list(self.histories)
        ncols = max(len(tasks), 1)
        width = COLUMN_INCHES["double"]
        height = width / 2.6
        fig, axes_obj = plt.subplots(1, ncols, figsize=(width, height), sharey=True)
        axes = np.atleast_1d(axes_obj)
        pretty = dict(pretty_tasks or {})
        for idx, task in enumerate(tasks):
            ax = axes[idx]
            per_task = self.histories[task]
            for j, model in enumerate(per_task):
                seed_histories = per_task[model]
                if not seed_histories:
                    continue
                key = metric_key or _auto_metric_key(seed_histories[0])
                arr = np.array([h[key] for h in seed_histories], dtype=float)
                mean = arr.mean(axis=0)
                n = arr.shape[0]
                if n > 1:
                    sem = arr.std(axis=0, ddof=1) / np.sqrt(n)
                    half = sem * stats.t.ppf((1 + ci) / 2, n - 1)
                else:
                    half = np.zeros_like(mean)
                epochs_axis = np.arange(1, mean.size + 1)
                color = NATURE_COLORS[j % len(NATURE_COLORS)]
                ax.plot(epochs_axis, mean, color=color, label=model, linewidth=1.2)
                ax.fill_between(epochs_axis, mean - half, mean + half, color=color, alpha=0.2, linewidth=0)
            ax.set_xlabel("Epoch")
            ax.set_title(pretty.get(task, task))
            ax.set_axisbelow(True)
            ax.yaxis.grid(True, linewidth=0.3, alpha=0.4)
            if idx == 0:
                ax.set_ylabel(ylabel)
                ax.legend(loc=legend_loc, borderaxespad=0.4)
            else:
                ax.tick_params(labelleft=False)
            panel_label(ax, "abcdefgh"[idx])
        fig.tight_layout()
        return fig, axes


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def _seed_all(seed: int) -> None:
    """Seed every RNG that benchmark training touches."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_seeds(
    seeds: int | Iterable[int] | None,
    seed: int | None,
) -> tuple[int, ...]:
    if seed is not None:
        return (int(seed),)
    if seeds is None:
        return (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    if isinstance(seeds, int):
        return (int(seeds),)
    return tuple(int(s) for s in seeds)


def run_benchmark(
    category: str | None = None,
    models: type | tuple[Any, ...] | ModelSpec | dict[str, type | tuple[Any, ...] | ModelSpec] | None = None,
    root: str = "data/benchmark",
    hidden_channels: int = 64,
    epochs: int | None = None,
    only: list[str] | None = None,
    kind: str | None = None,
    verbose: bool = True,
    seeds: int | Iterable[int] | None = None,
    seed: int | None = None,
    tasks: Iterable[Task] | None = None,
) -> BenchmarkReport:
    """Run a benchmark across one or more (model, task, seed) combinations.

    Two ways to choose tasks:

    1. **By category** (default) -- tasks come from
       :data:`BENCHMARK_TASKS` indexed as
       ``[category][task_kind] -> list[Task]``. Pass ``category="social"``
       (etc.) and optionally restrict with ``kind=`` and ``only=``.
    2. **Ad-hoc** -- pass ``tasks=[Task(...), ...]`` to bypass the registry
       entirely. Useful for benchmarking custom datasets without mutating
       global state. ``category`` then defaults to ``"custom"`` and is used
       only to namespace ``root/`` cache directories.

    The runner trains every compatible (model, task) pair across each
    value in ``seeds`` (default ``(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)``) and aggregates the per-seed
    histories into a :class:`BenchmarkReport`.
    """
    if models is None:
        msg = "run_benchmark requires `models` (a class, dict, or ModelSpec)"
        raise ValueError(msg)
    if kind is not None and kind not in TASK_KINDS:
        msg = f"Unknown task kind {kind!r}. Choices: {sorted(TASK_KINDS)}"
        raise ValueError(msg)
    if not isinstance(models, dict):
        spec = _spec_from(models)
        models = {spec.cls.__name__: spec}

    resolved = {name: _spec_from(value) for name, value in models.items()}
    seed_list = _normalize_seeds(seeds, seed)

    if tasks is not None:
        task_list = list(tasks)
        for t in task_list:
            if not isinstance(t, Task):
                msg = f"`tasks` must contain Task instances, got {type(t).__name__}"
                raise TypeError(msg)
            if t.kind not in TASK_KINDS:
                msg = f"Task {t.name!r} has unknown kind {t.kind!r}; choices: {sorted(TASK_KINDS)}"
                raise ValueError(msg)
        if kind is not None:
            task_list = [t for t in task_list if t.kind == kind]
        if category is None:
            category = "custom"
    else:
        if category is None:
            msg = "run_benchmark requires either `category` or `tasks=`"
            raise ValueError(msg)
        if category not in BENCHMARK_TASKS:
            msg = f"Unknown category {category!r}. Choices: {sorted(BENCHMARK_TASKS)}"
            raise KeyError(msg)
        task_list = iter_benchmark_tasks(category=category, kind=kind)
    if only is not None:
        task_list = [t for t in task_list if t.name in only]
    tasks = task_list  # the loop below treats this as the working list

    histories: dict[str, dict[str, list[dict[str, list[float]]]]] = {}
    total_combinations = sum(1 for spec in resolved.values() for task in tasks if task.kind in spec.kinds) * len(
        seed_list
    )
    overall_pbar = tqdm(
        total=total_combinations,
        desc="Benchmark",
        unit="run",
        disable=not verbose,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    import inspect

    for task in tasks:
        try:
            seed_aware = "seed" in inspect.signature(task.loader).parameters
        except (TypeError, ValueError):
            seed_aware = False
        ds_cache: Any = None  # for seed-agnostic loaders, load once
        histories[task.name] = {}
        for model_name, spec in resolved.items():
            if task.kind not in spec.kinds:
                continue
            histories[task.name][model_name] = []
            for s in seed_list:
                _seed_all(s)
                if seed_aware:
                    # Seed-aware loaders (e.g. synthetic combinatorial graphs)
                    # produce a fresh dataset per seed, so cross-seed variance
                    # captures data resampling rather than only model init.
                    ds = task.loader(f"{root}/{category}/{task.name}/seed{s}", seed=s)
                else:
                    if ds_cache is None:
                        ds_cache = task.loader(f"{root}/{category}/{task.name}")
                    ds = ds_cache
                history = _run_task(task, ds, spec, hidden_channels, epochs or task.epochs, verbose=verbose)
                histories[task.name][model_name].append(history)
                # Update overall progress with latest metric
                last_metrics = {k: v[-1] for k, v in history.items() if v}
                metric_str = " ".join(f"{k[:3]}={v:.3f}" for k, v in last_metrics.items())
                overall_pbar.set_postfix_str(f"{task.name}/{model_name}/s{s} | {metric_str}", refresh=False)
                overall_pbar.update(1)
    overall_pbar.close()

    config = {
        "category": category,
        "kind": kind,
        "hidden_channels": hidden_channels,
        "epochs": epochs,
        "only": only,
    }
    return BenchmarkReport(seeds=seed_list, histories=histories, config=config)


def plot_benchmark(
    results: BenchmarkReport | Mapping[str, Mapping[str, Mapping[str, list[float]]]],
    errors: Mapping[str, Mapping[str, float]] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    annotate: bool = True,
    ci: float = 0.95,
) -> tuple[plt.Figure, plt.Axes]:
    """Grouped bar chart with mean ± CI error bars.

    Accepts a :class:`BenchmarkReport` (preferred) or the legacy dict form for
    a single seed. ``errors`` overrides the default t-CI half-width.
    """
    if isinstance(results, BenchmarkReport):
        return results.plot(ax=ax, title=title, annotate=annotate, ci=ci)

    set_plot_style()
    values: dict[str, dict[str, float]] = {}
    metric_label: str | None = None
    for task_name, per_task in results.items():
        per_value: dict[str, float] = {}
        for model_name, history in per_task.items():
            metric, value = _final_metric(history)
            metric_label = metric_label or metric
            per_value[model_name] = value
        values[task_name] = per_value

    return plot_grouped_bars(
        values,
        errors=errors,
        ax=ax,
        title=title,
        ylabel=metric_label or "metric",
        annotate=annotate,
    )


__all__ = [
    "BENCHMARK_TASKS",
    "TASK_KINDS",
    "BenchmarkReport",
    "ModelSpec",
    "Task",
    "iter_benchmark_tasks",
    "plot_benchmark",
    "register_model",
    "register_task",
    "run_benchmark",
    "save_figure",
    "task_from_dataset",
    "unregister_task",
]

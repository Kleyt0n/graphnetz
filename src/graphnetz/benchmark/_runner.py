"""The benchmark driver: per-task training dispatch and the public runner."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from graphnetz.benchmark._report import BenchmarkReport
from graphnetz.benchmark._specs import TASK_TYPES, ModelSpec, Task, _spec_from
from graphnetz.benchmark._stats import _final_metric
from graphnetz.benchmark._tasks import BENCHMARK_TASKS, iter_benchmark_tasks
from graphnetz.plotting import plot_grouped_bars, set_plot_style
from graphnetz.training import (
    train_graph_classification,
    train_graph_regression,
    train_link_prediction,
    train_node_classification,
)

# --------------------------------------------------------------------------- #
# Task runner
# --------------------------------------------------------------------------- #


def _run_task(
    task_type: Task,
    ds: Any,
    spec: ModelSpec,
    hidden: int,
    epochs: int,
    verbose: bool = False,
    device: torch.device | str | None = "auto",
) -> dict[str, list[float]]:
    if task_type.task_type == "node_cls":
        data = ds[0]
        model = spec.build(ds.num_features, hidden, ds.num_classes, task_type="node_cls")
        return train_node_classification(model, data, epochs=epochs, verbose=verbose, device=device)

    if task_type.task_type == "graph_cls":
        shuffled = ds.shuffle()
        split = int(0.8 * len(shuffled))
        train_loader = DataLoader(shuffled[:split], batch_size=32, shuffle=True)
        val_loader = DataLoader(shuffled[split:], batch_size=32)
        model = spec.build(shuffled.num_features, hidden, shuffled.num_classes, task_type="graph_cls")
        return train_graph_classification(
            model, train_loader, val_loader, epochs=epochs, verbose=verbose, device=device
        )

    if task_type.task_type == "graph_reg":
        # Loader may return either a single dataset (used for both train and
        # held-out -- e.g. synthetic tasks with no canonical split) or a
        # ``(train_ds, val_ds)`` tuple (real benchmarks like ZINC).
        if isinstance(ds, tuple):
            train_ds, val_ds = ds
        else:
            train_ds = val_ds = ds
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64)
        inner = spec.build(hidden, hidden, 1, task_type="graph_reg")

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

        return train_graph_regression(
            _AtomEmbed(), train_loader, val_loader, epochs=epochs, verbose=verbose, device=device
        )

    if task_type.task_type == "link_pred":
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
            built = spec.build(data.num_features, hidden, hidden, task_type="link_pred")
            # spec.build returns a LinkPredWrapper for task_type="link_pred"; unwrap it
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
                device=device,
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
        # ``spec.build(task_type="link_pred")`` returns a LinkPredWrapper, which
        # satisfies the ``_LinkPredLike`` protocol of the trainer; mypy
        # only sees the declared ``Module`` return so we narrow here.
        from typing import cast as _cast

        from graphnetz.training import _LinkPredLike

        lp_model = _cast(_LinkPredLike, spec.build(train_data.num_features, hidden, hidden, task_type="link_pred"))
        return train_link_prediction(
            lp_model, train_data, val_data, test_data, epochs=epochs, verbose=verbose, device=device
        )

    msg = f"Unknown task task_type: {task_type.task_type}"
    raise ValueError(msg)


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
    verbose: bool = True,
    seeds: int | Iterable[int] | None = None,
    seed: int | None = None,
    task_type: str | None = None,
    tasks: Iterable[Task] | None = None,
    device: torch.device | str | None = "auto",
) -> BenchmarkReport:
    """Run a benchmark across one or more (model, task, seed) combinations.

    Two ways to choose tasks:

    1. **By category** (default) -- tasks come from
       :data:`BENCHMARK_TASKS` indexed as
       ``[category][task_type] -> list[Task]``. Pass ``category="social"``
       (etc.) and optionally restrict with ``task_type`` and ``only=``.
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
    if task_type is not None and task_type not in TASK_TYPES:
        msg = f"Unknown task type {task_type!r}. Choices: {sorted(TASK_TYPES)}"
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
            if t.task_type not in TASK_TYPES:
                msg = f"Task {t.name!r} has unknown task type {t.task_type!r}; choices: {sorted(TASK_TYPES)}"
                raise ValueError(msg)
        if task_type is not None:
            task_list = [t for t in task_list if t.task_type == task_type]
        if category is None:
            category = "custom"
    else:
        if category is None:
            msg = "run_benchmark requires either `category` or `tasks=`"
            raise ValueError(msg)
        if category not in BENCHMARK_TASKS:
            msg = f"Unknown category {category!r}. Choices: {sorted(BENCHMARK_TASKS)}"
            raise KeyError(msg)
        task_list = iter_benchmark_tasks(category=category, task_type=task_type)
    if only is not None:
        task_list = [t for t in task_list if t.name in only]
    tasks = task_list  # the loop below treats this as the working list

    histories: dict[str, dict[str, list[dict[str, list[float]]]]] = {}
    total_combinations = sum(
        1 for spec in resolved.values() for task in tasks if task.task_type in spec.task_type
    ) * len(seed_list)
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
            if task.task_type not in spec.task_type:
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
                history = _run_task(
                    task, ds, spec, hidden_channels, epochs or task.epochs, verbose=verbose, device=device
                )
                histories[task.name][model_name].append(history)
                # Update overall progress with latest metric
                last_metrics = {k: v[-1] for k, v in history.items() if v}
                metric_str = " ".join(f"{k[:3]}={v:.3f}" for k, v in last_metrics.items())
                overall_pbar.set_postfix_str(f"{task.name}/{model_name}/s{s} | {metric_str}", refresh=False)
                overall_pbar.update(1)
    overall_pbar.close()

    from graphnetz.training import _resolve_device

    config = {
        "category": category,
        "task": task,
        "hidden_channels": hidden_channels,
        "epochs": epochs,
        "only": only,
        "device": str(_resolve_device(device)),
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

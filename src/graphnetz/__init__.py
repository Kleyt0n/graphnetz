"""GraphNet: a database and benchmark for graph learning."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from graphnetz.__about__ import __version__

# Map public attribute -> module that owns it. Loaded lazily via PEP 562 so
# ``import graphnetz; graphnetz.__version__`` doesn't import matplotlib/tqdm.
_LAZY: dict[str, str] = {
    # benchmark
    "BENCHMARK_TASKS": "graphnetz.benchmark",
    "BenchmarkReport": "graphnetz.benchmark",
    "ModelSpec": "graphnetz.benchmark",
    "Task": "graphnetz.benchmark",
    "plot_benchmark": "graphnetz.benchmark",
    "register_model": "graphnetz.benchmark",
    "register_task": "graphnetz.benchmark",
    "run_benchmark": "graphnetz.benchmark",
    "task_from_dataset": "graphnetz.benchmark",
    "unregister_task": "graphnetz.benchmark",
    # datasets
    "CATEGORIES": "graphnetz.datasets",
    "Netz": "graphnetz.datasets",
    "download_all_networks_netz": "graphnetz.datasets",
    "list_datasets": "graphnetz.datasets",
    # models
    "DGI": "graphnetz.models",
    "GAT": "graphnetz.models",
    "GCN": "graphnetz.models",
    "GIN": "graphnetz.models",
    "GraphSAGE": "graphnetz.models",
    "GraphTransformer": "graphnetz.models",
    # plotting
    "figure": "graphnetz.plotting",
    "panel_label": "graphnetz.plotting",
    "plot_grouped_bars": "graphnetz.plotting",
    "plot_history": "graphnetz.plotting",
    "save_figure": "graphnetz.plotting",
    "set_plot_style": "graphnetz.plotting",
    # training
    "train_dgi": "graphnetz.training",
    "train_graph_classification": "graphnetz.training",
    "train_graph_regression": "graphnetz.training",
    "train_link_prediction": "graphnetz.training",
    "train_node_classification": "graphnetz.training",
    "train_node_degree_regression": "graphnetz.training",
    "train_relational_link_prediction": "graphnetz.training",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module = importlib.import_module(_LAZY[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    msg = f"module 'graphnetz' has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted({*_LAZY, "__version__"})


if TYPE_CHECKING:
    # Re-export for static analysers; not executed at runtime.
    from graphnetz.benchmark import (
        BENCHMARK_TASKS,
        BenchmarkReport,
        ModelSpec,
        Task,
        plot_benchmark,
        register_model,
        register_task,
        run_benchmark,
        task_from_dataset,
        unregister_task,
    )
    from graphnetz.datasets import CATEGORIES, Netz, download_all_networks_netz, list_datasets
    from graphnetz.models import DGI, GAT, GCN, GIN, GraphSAGE, GraphTransformer
    from graphnetz.plotting import (
        figure,
        panel_label,
        plot_grouped_bars,
        plot_history,
        save_figure,
        set_plot_style,
    )
    from graphnetz.training import (
        train_dgi,
        train_graph_classification,
        train_graph_regression,
        train_link_prediction,
        train_node_classification,
        train_node_degree_regression,
        train_relational_link_prediction,
    )

__all__ = [
    "BENCHMARK_TASKS",
    "CATEGORIES",
    "DGI",
    "GAT",
    "GCN",
    "GIN",
    "BenchmarkReport",
    "GraphSAGE",
    "GraphTransformer",
    "ModelSpec",
    "Netz",
    "Task",
    "__version__",
    "download_all_networks_netz",
    "figure",
    "list_datasets",
    "panel_label",
    "plot_benchmark",
    "plot_grouped_bars",
    "plot_history",
    "register_model",
    "register_task",
    "run_benchmark",
    "save_figure",
    "set_plot_style",
    "task_from_dataset",
    "train_dgi",
    "train_graph_classification",
    "train_graph_regression",
    "train_link_prediction",
    "train_node_classification",
    "train_node_degree_regression",
    "train_relational_link_prediction",
    "unregister_task",
]

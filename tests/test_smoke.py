"""Import-surface smoke tests. Run with ``uv run pytest``."""

from __future__ import annotations

import pytest


def test_top_level_imports() -> None:
    import graphnetz

    assert graphnetz.__version__
    for name in (
        "GCN",
        "GAT",
        "GIN",
        "GraphSAGE",
        "GraphTransformer",
        "DGI",
        "Netz",
        "run_benchmark",
        "plot_benchmark",
        "plot_history",
        "set_plot_style",
        "list_datasets",
    ):
        assert hasattr(graphnetz, name), name


def test_dataset_registry() -> None:
    import importlib.util

    from graphnetz import list_datasets

    ld = list_datasets()
    required = {
        "biology",
        "combinatorial",
        "computing",
        "finance",
        "infrastructure",
        "knowledge",
        "physics",
        "security",
        "social",
        "vision",
    }
    assert required.issubset(set(ld))
    # ``ogb`` is only registered when the optional extra is installed.
    assert ("ogb" in ld) == (importlib.util.find_spec("ogb") is not None)
    # Every category should expose at least one loader.
    assert all(ld.values())


def test_benchmark_registry() -> None:
    from graphnetz.benchmark import BENCHMARK_TASKS

    assert set(BENCHMARK_TASKS) >= {"social", "biology", "combinatorial", "infrastructure"}
    for per_cat in BENCHMARK_TASKS.values():
        for task_list in per_cat.values():
            for task in task_list:
                assert task.kind in {"node_cls", "graph_cls", "graph_reg", "link_pred"}
                assert callable(task.loader)


def test_ogb_import_error() -> None:
    """OGB loaders raise a helpful error when the extra is missing."""
    import importlib.util

    ogb_spec = importlib.util.find_spec("ogb")
    if ogb_spec is not None:
        pytest.skip("ogb extra is installed")

    from graphnetz.datasets.ogb import ogbn_arxiv

    with pytest.raises(ImportError, match="requires the 'ogb' extra"):
        ogbn_arxiv("data/ogb")

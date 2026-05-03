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

    from graphnetz.datasets.social import ogbn_arxiv

    with pytest.raises(ImportError, match="requires the 'ogb' extra"):
        ogbn_arxiv("data/ogb")


def test_resolve_device_auto_picks_available() -> None:
    """``device='auto'`` resolves to CUDA, then MPS, then CPU."""
    import torch

    from graphnetz.training import _resolve_device

    expected: str
    if torch.cuda.is_available():
        expected = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        expected = "mps"
    else:
        expected = "cpu"
    assert _resolve_device("auto").type == expected
    assert _resolve_device(None).type == expected
    # Explicit overrides pass through unchanged.
    assert _resolve_device("cpu").type == "cpu"
    assert _resolve_device(torch.device("cpu")).type == "cpu"


def test_train_node_classification_auto_device() -> None:
    """Trainer auto-moves model + data; final accuracy is on the resolved device."""
    import torch

    from graphnetz import GCN, train_node_classification
    from graphnetz.datasets.social import cora

    ds = cora("data/cora")
    data = ds[0]
    # Leave inputs on CPU; the trainer must move them itself.
    model = GCN(ds.num_features, 16, ds.num_classes)
    history = train_node_classification(model, data, epochs=2, verbose=False)
    assert len(history["train_loss"]) == 2
    # Model parameters end up on the resolved device.
    from graphnetz.training import _resolve_device

    expected_device = _resolve_device("auto")
    assert next(model.parameters()).device.type == expected_device.type

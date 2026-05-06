"""Tests for the custom-dataset surface (ad-hoc Tasks, registration helpers)."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import KarateClub

# --------------------------------------------------------------------------- #
# Ad-hoc dataset wrapper
# --------------------------------------------------------------------------- #


class _MiniNodeDataset:
    """Minimum viable PyG-shaped dataset for node classification.

    Three Karate-Club-sized graphs is enough to exercise the dispatcher
    without a network download.
    """

    def __init__(self) -> None:
        karate = KarateClub()
        self._data = karate[0]
        # Ensure splits exist so train_node_classification has something to do.
        n = self._data.num_nodes
        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        test_mask = torch.zeros(n, dtype=torch.bool)
        train_mask[: n // 2] = True
        val_mask[n // 2 : 3 * n // 4] = True
        test_mask[3 * n // 4 :] = True
        self._data.train_mask = train_mask
        self._data.val_mask = val_mask
        self._data.test_mask = test_mask
        self.num_features = int(self._data.num_features)
        self.num_classes = int(self._data.y.max().item()) + 1

    def __getitem__(self, idx: int) -> Data:
        if idx != 0:
            raise IndexError(idx)
        return self._data

    def __len__(self) -> int:
        return 1


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_task_from_dataset_runs_through_run_benchmark() -> None:
    from graphnetz import GCN, run_benchmark, task_from_dataset

    ds = _MiniNodeDataset()
    task = task_from_dataset("karate_mini", "node_cls", ds, epochs=2)

    report = run_benchmark(
        models={"GCN": GCN},
        tasks=[task],
        seeds=(0, 1),
        verbose=False,
    )

    assert "karate_mini" in report.histories
    assert "GCN" in report.histories["karate_mini"]
    seed_histories = report.histories["karate_mini"]["GCN"]
    assert len(seed_histories) == 2
    assert "test_acc" in seed_histories[0]
    summary = report.summary()
    assert summary.loc[("karate_mini", "GCN"), "n_seeds"] == 2


def test_run_benchmark_validates_task() -> None:
    from graphnetz import GCN, run_benchmark
    from graphnetz.benchmark import Task

    bad = Task(name="bogus", task_type="not_a_task_type", loader=lambda _root: None, epochs=1)
    with pytest.raises(ValueError, match="unknown task type"):
        run_benchmark(models={"GCN": GCN}, tasks=[bad], seeds=(0,), verbose=False)


def test_run_benchmark_requires_either_category_or_tasks() -> None:
    from graphnetz import GCN, run_benchmark

    with pytest.raises(ValueError, match="either `category` or `tasks="):
        run_benchmark(models={"GCN": GCN}, seeds=(0,), verbose=False)


def test_run_benchmark_requires_models() -> None:
    from graphnetz import run_benchmark, task_from_dataset

    task = task_from_dataset("x", "node_cls", _MiniNodeDataset(), epochs=1)
    with pytest.raises(ValueError, match="requires `models`"):
        run_benchmark(tasks=[task], seeds=(0,), verbose=False)


def test_register_task_and_unregister_round_trip() -> None:
    from graphnetz import register_task, task_from_dataset, unregister_task
    from graphnetz.benchmark import BENCHMARK_TASKS, iter_benchmark_tasks

    task = task_from_dataset("karate_mini_reg", "node_cls", _MiniNodeDataset(), epochs=1)
    register_task("custom_test_category", task)
    try:
        assert "custom_test_category" in BENCHMARK_TASKS
        names = [t.name for t in iter_benchmark_tasks(category="custom_test_category")]
        assert names == ["karate_mini_reg"]

        # Re-registering the same name must error.
        with pytest.raises(ValueError, match="already registered"):
            register_task("custom_test_category", task)
    finally:
        removed = unregister_task("custom_test_category", "karate_mini_reg")
        # Clean up the empty category itself so we don't leak state.
        if "custom_test_category" in BENCHMARK_TASKS:
            BENCHMARK_TASKS.pop("custom_test_category", None)
    assert removed is task


def test_register_task_then_run_benchmark_by_category() -> None:
    from graphnetz import GCN, register_task, run_benchmark, task_from_dataset, unregister_task
    from graphnetz.benchmark import BENCHMARK_TASKS

    task = task_from_dataset("karate_via_registry", "node_cls", _MiniNodeDataset(), epochs=2)
    register_task("custom_test_category", task)
    try:
        report = run_benchmark(
            "custom_test_category",
            {"GCN": GCN},
            seeds=(0,),
            verbose=False,
        )
        assert "karate_via_registry" in report.histories
    finally:
        unregister_task("custom_test_category", "karate_via_registry")
        BENCHMARK_TASKS.pop("custom_test_category", None)


def test_seed_aware_loader_receives_seed() -> None:
    """Loaders that declare a `seed` parameter get the benchmark seed passed in."""
    from graphnetz import GCN, run_benchmark
    from graphnetz.benchmark import Task

    seeds_seen: list[int] = []
    base_ds = _MiniNodeDataset()

    def loader(_root: str, *, seed: int) -> _MiniNodeDataset:
        seeds_seen.append(seed)
        return base_ds

    task = Task(name="seed_aware", task_type="node_cls", loader=loader, epochs=1)
    run_benchmark(
        models={"GCN": GCN},
        tasks=[task],
        seeds=(7, 11),
        verbose=False,
    )
    assert seeds_seen == [7, 11]


def test_iter_benchmark_tasks_excludes_custom_after_unregister() -> None:
    from graphnetz import register_task, task_from_dataset, unregister_task
    from graphnetz.benchmark import BENCHMARK_TASKS, iter_benchmark_tasks

    name = "iter_check_dataset"
    task = task_from_dataset(name, "node_cls", _MiniNodeDataset(), epochs=1)
    register_task("custom_test_category", task)
    unregister_task("custom_test_category", name)
    BENCHMARK_TASKS.pop("custom_test_category", None)

    assert all(t.name != name for t in iter_benchmark_tasks())

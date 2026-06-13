"""Curated benchmark task taxonomy and task registration helpers."""

from __future__ import annotations

import importlib.util
from typing import Any

from graphnetz.benchmark._specs import TASK_TYPES, Task
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

_HAS_OGB = importlib.util.find_spec("ogb") is not None

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
"""Curated benchmark taxonomy: ``category -> task_type -> [Task, ...]``."""

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
    BENCHMARK_TASKS["biology"]["graph_cls"].extend(
        [
            Task("ogbg_molhiv", "graph_cls", biology.ogbg_molhiv, epochs=20),
            Task("ogbg_molpcba", "graph_cls", biology.ogbg_molpcba, epochs=20),
        ]
    )


def iter_benchmark_tasks(
    category: str | None = None,
    task_type: str | None = None,
) -> list[Task]:
    """Flatten ``BENCHMARK_TASKS`` to a list, optionally filtered by category/task.

    Examples
    --------
    >>> [
    ...     t.name
    ...     for t in iter_benchmark_tasks(category="biology", task_type="graph_cls")
    ... ]
    ['mutag', 'proteins']
    """
    cats = [category] if category is not None else list(BENCHMARK_TASKS)
    out: list[Task] = []
    for c in cats:
        per_cat = BENCHMARK_TASKS.get(c, {})
        tasks = [task_type] if task_type is not None else list(per_cat)
        for k in tasks:
            out.extend(per_cat.get(k, []))
    return out


# --------------------------------------------------------------------------- #
# Custom-dataset helpers
# --------------------------------------------------------------------------- #


def task_from_dataset(
    name: str,
    task_type: str,
    dataset: Any,
    *,
    epochs: int = 30,
) -> Task:
    """Wrap an already-loaded dataset as a :class:`Task`.

    The dataset must satisfy the conventions for ``task``: a PyG dataset or
    any object exposing ``ds[0]`` plus the relevant attributes (``num_features``
    / ``num_classes`` / ``num_relations``). The benchmark dispatcher caches
    the dataset, so the same instance is reused across seeds without
    reloading.
    """
    if task_type not in TASK_TYPES:
        msg = f"Unknown task {task_type!r}; choices: {sorted(TASK_TYPES)}"
        raise ValueError(msg)
    return Task(name=name, task_type=task_type, loader=lambda _root: dataset, epochs=epochs)


def register_task(category: str, task_type: Task) -> None:
    """Register ``task`` under ``category`` in :data:`BENCHMARK_TASKS`.

    The task becomes visible to ``run_benchmark(category)`` and to
    :func:`iter_benchmark_tasks`. Use :func:`unregister_task` to remove it
    (e.g. in ``tearDown`` of a test).
    """
    if not isinstance(task_type, Task):
        msg = f"task must be a Task, got {type(task_type).__name__}"
        raise TypeError(msg)
    if task_type.task_type not in TASK_TYPES:
        msg = f"Task {task_type.name!r} has unknown task {task_type.task_type!r}; choices: {sorted(TASK_TYPES)}"
        raise ValueError(msg)
    per_cat = BENCHMARK_TASKS.setdefault(category, {})
    per = per_cat.setdefault(task_type.task_type, [])
    if any(t.name == task_type.name for t in per):
        msg = f"Task {task_type.name!r} already registered in category {category!r}/{task_type.task_type!r}"
        raise ValueError(msg)
    per.append(task_type)


def unregister_task(category: str, name: str) -> Task | None:
    """Remove a previously registered task; returns it, or ``None`` if absent."""
    per_cat = BENCHMARK_TASKS.get(category, {})
    for task_tasks in per_cat.values():
        for i, t in enumerate(task_tasks):
            if t.name == name:
                return task_tasks.pop(i)
    return None

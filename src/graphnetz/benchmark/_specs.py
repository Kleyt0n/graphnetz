"""Task/model specifications and the model registry."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import torch

from graphnetz.models import GAT, GCN, GIN, GraphSAGE, GraphTransformer

# DGI is intentionally not a task task_type: it is a self-supervised training
# objective whose "metric" is its own loss, so it cannot serve as a
# held-out evaluation. ``train_dgi`` and the ``DGIWrapper`` adapter remain
# available as utilities for users who want to pre-train an encoder
# unsupervised; the benchmark routes unlabelled graphs through
# ``link_pred`` instead (a real held-out edge split with an AUC metric).
TASK_TYPES: frozenset[str] = frozenset({"node_cls", "graph_cls", "graph_reg", "link_pred"})


# --------------------------------------------------------------------------- #
# Tasks and model specs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Task:
    """A single benchmark task_type: a dataset loader plus its training task."""

    name: str
    task_type: str
    # ``...`` admits seed-aware loaders ``f(root, *, seed=...)`` alongside
    # the basic ``f(root)`` shape — the dispatcher inspects the signature
    # and threads ``seed`` through when present.
    loader: Callable[..., Any]
    epochs: int = 30


@dataclass(frozen=True)
class ModelSpec:
    """How to instantiate a model and which task tasks it supports."""

    cls: type
    task_type: frozenset[str] = field(default_factory=frozenset)
    factory: Callable[..., torch.nn.Module] | None = None

    def build(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        *,
        task_type: str = "node_cls",
    ) -> torch.nn.Module:
        if self.factory is not None:
            try:
                return self.factory(in_channels, hidden_channels, out_channels, task_type=task_type)
            except TypeError:
                return self.factory(in_channels, hidden_channels, out_channels)
        if task_type == "dgi":
            return self.cls(in_channels, hidden_channels)
        return self.cls(in_channels, hidden_channels, out_channels)


_REGISTRY: dict[type, ModelSpec] = {}


def register_model(
    cls: type | None = None,
    *,
    task_type: str | Iterable[str],
    factory: Callable[..., torch.nn.Module] | None = None,
) -> Callable[[type], type] | type:
    """Register a model with the benchmark dispatcher.

    Usable as a decorator (``@register_model(task_type="node_cls")``) or as a
    plain function (``register_model(MyGNN, task_type={"graph_cls", "graph_reg"})``).
    """
    tasks = frozenset({task_type} if isinstance(task_type, str) else task_type)
    unknown = tasks - TASK_TYPES
    if unknown:
        msg = f"Unknown task {sorted(unknown)}; allowed: {sorted(TASK_TYPES)}"
        raise ValueError(msg)

    def _register(target: type) -> type:
        _REGISTRY[target] = ModelSpec(cls=target, task_type=tasks, factory=factory)
        return target

    return _register(cls) if cls is not None else _register


def _multi_task_factory(encoder_cls: type) -> Callable[..., torch.nn.Module]:
    """Adapt a node-level encoder to any of the four tasks.

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
        task_type: str = "node_cls",
    ) -> torch.nn.Module:
        if task_type == "node_cls":
            return encoder_cls(in_channels, hidden_channels, out_channels)
        if task_type in ("graph_cls", "graph_reg"):
            encoder = encoder_cls(in_channels, hidden_channels, hidden_channels)
            return GraphLevelWrapper(encoder, hidden_channels, out_channels)
        if task_type == "link_pred":
            encoder = encoder_cls(in_channels, hidden_channels, hidden_channels)
            return LinkPredWrapper(encoder)
        msg = f"Unknown task task_type: {task_type!r}; choices: {sorted(TASK_TYPES)}"
        raise ValueError(msg)

    return factory


# Pre-register built-ins. Node-level encoders are registered for every
# task task via the multi-task factory; GIN keeps its native graph-level
# pooling. ``DGI`` is intentionally not registered: it is exposed as a
# self-supervised training utility (``train_dgi`` + ``DGIWrapper``)
# rather than a benchmark-task model.
_ALL_TASKS = frozenset({"node_cls", "graph_cls", "graph_reg", "link_pred"})
register_model(GCN, task_type=_ALL_TASKS, factory=_multi_task_factory(GCN))
register_model(GAT, task_type=_ALL_TASKS, factory=_multi_task_factory(GAT))
register_model(GraphSAGE, task_type=_ALL_TASKS, factory=_multi_task_factory(GraphSAGE))
register_model(GraphTransformer, task_type=_ALL_TASKS, factory=_multi_task_factory(GraphTransformer))
register_model(GIN, task_type={"graph_cls", "graph_reg"})


def _spec_from(value: type | tuple[Any, ...] | ModelSpec) -> ModelSpec:
    """Resolve a ``models`` dict entry to a :class:`ModelSpec`."""
    if isinstance(value, ModelSpec):
        return value
    if isinstance(value, tuple):
        cls = value[0]
        tasks = value[1] if len(value) >= 2 else None
        factory = value[2] if len(value) >= 3 else None
        if tasks is None:
            base = _spec_from(cls)
            return ModelSpec(cls=base.cls, task_type=base.task_type, factory=factory or base.factory)
        ks = frozenset({tasks} if isinstance(tasks, str) else tasks)
        unknown = ks - TASK_TYPES
        if unknown:
            msg = f"Unknown task task_type {sorted(unknown)}; allowed: {sorted(TASK_TYPES)}"
            raise ValueError(msg)
        return ModelSpec(cls=cls, task_type=ks, factory=factory)
    if value in _REGISTRY:
        return _REGISTRY[value]
    if hasattr(value, "task_types"):
        return ModelSpec(cls=value, task_type=frozenset(value.task_types))
    if hasattr(value, "task"):
        return ModelSpec(cls=value, task_type=frozenset({value.task}))
    return ModelSpec(cls=value, task_type=frozenset())

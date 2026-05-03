"""Shared loading helpers for Open Graph Benchmark (OGB) datasets.

The public entry points live in the domain modules (``social``,
``biology``, ``finance``); this module only houses the wire-format
adapters that turn an OGB dataset into a PyG-shaped object. All
helpers raise a friendly :class:`ImportError` when ``ogb`` is missing,
pointing the user at ``pip install graphnetz[ogb]``.
"""

from __future__ import annotations

from typing import Any

from torch_geometric.data import Data

_OGB_INSTALL_HINT = "Install with:  pip install graphnetz[ogb]"


def load_ogb_node(name: str, root: str) -> Data:
    """Return a PyG ``Data`` for an OGB node-property dataset."""
    try:
        from ogb.nodeproppred import NodePropPredDataset
    except ImportError as exc:
        msg = f"'{name}' requires the 'ogb' extra. {_OGB_INSTALL_HINT}"
        raise ImportError(msg) from exc

    import torch

    ds = NodePropPredDataset(name=name, root=root)
    graph, label = ds[0]
    split = ds.get_idx_split()

    edge_index = torch.from_numpy(graph["edge_index"]).long()
    x = torch.from_numpy(graph["node_feat"]).float()
    y = torch.from_numpy(label).long().view(-1)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = torch.zeros(y.size(0), dtype=torch.bool)
    data.val_mask = torch.zeros(y.size(0), dtype=torch.bool)
    data.test_mask = torch.zeros(y.size(0), dtype=torch.bool)
    data.train_mask[split["train"]] = True
    data.val_mask[split["valid"]] = True
    data.test_mask[split["test"]] = True

    data.num_features = x.size(1)
    data.num_classes = int(y.max()) + 1
    return data


def load_ogb_graph(name: str, root: str) -> Any:
    """Return the raw OGB ``GraphPropPredDataset`` for graph-prop tasks."""
    try:
        from ogb.graphproppred import GraphPropPredDataset
    except ImportError as exc:
        msg = f"'{name}' requires the 'ogb' extra. {_OGB_INSTALL_HINT}"
        raise ImportError(msg) from exc

    return GraphPropPredDataset(name=name, root=root)


def load_ogb_link(name: str, root: str) -> Data:
    """Return a single-graph PyG ``Data`` for an OGB link-prop dataset.

    The benchmark runner re-splits via ``RandomLinkSplit``; OGB's
    official train/valid/test edge split is not consumed here. For
    protocol-faithful evaluation, fall back to
    ``ogb.linkproppred.LinkPropPredDataset`` directly.
    """
    try:
        from ogb.linkproppred import LinkPropPredDataset
    except ImportError as exc:
        msg = f"'{name}' requires the 'ogb' extra. {_OGB_INSTALL_HINT}"
        raise ImportError(msg) from exc

    import torch

    ds = LinkPropPredDataset(name=name, root=root)
    graph = ds[0]

    edge_index = torch.from_numpy(graph["edge_index"]).long()
    num_nodes = int(graph["num_nodes"])
    data = Data(edge_index=edge_index, num_nodes=num_nodes)

    node_feat = graph.get("node_feat")
    if node_feat is not None:
        data.x = torch.from_numpy(node_feat).float()
        data.num_features = data.x.size(1)
    return data

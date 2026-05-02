"""Open Graph Benchmark (OGB) dataset loaders.

These loaders are only available when the ``ogb`` extra is installed::

    pip install graphnetz[ogb]

Coverage
--------
- Node classification: ``ogbn-arxiv`` (citation network, ~169 K nodes).
- Graph classification: ``ogbg-molhiv`` (molecular property prediction,
  ~41 K graphs, requires the ``chem`` extra for RDKit featurisation).
"""

from __future__ import annotations

from typing import Any

from torch_geometric.data import Data


def _load_ogb_node(name: str, root: str) -> Any:
    try:
        from ogb.nodeproppred import NodePropPredDataset
    except ImportError as exc:
        msg = f"'{name}' requires the 'ogb' extra. Install with:  pip install graphnetz[ogb]"
        raise ImportError(msg) from exc

    ds = NodePropPredDataset(name=name, root=root)
    graph, label = ds[0]
    split = ds.get_idx_split()

    import torch

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

    # Attach metadata expected by the benchmark dispatcher
    data.num_features = x.size(1)
    data.num_classes = int(y.max()) + 1
    return data


def ogbn_arxiv(root: str) -> Data:
    """OGB node-property prediction: arXiv citation network.

    Returns a single :class:`~torch_geometric.data.Data` object with
    ``train_mask``, ``val_mask``, and ``test_mask`` attributes.
    """
    return _load_ogb_node("ogbn-arxiv", root)


def _load_ogb_graph(name: str, root: str) -> Any:
    try:
        from ogb.graphproppred import GraphPropPredDataset
    except ImportError as exc:
        msg = f"'{name}' requires the 'ogb' extra. Install with:  pip install graphnetz[ogb]"
        raise ImportError(msg) from exc

    ds = GraphPropPredDataset(name=name, root=root)
    return ds


def ogbg_molhiv(root: str) -> Any:
    """OGB graph-property prediction: molecular HIV inhibition (MolHIV).

    Returns a :class:`~ogb.graphproppred.GraphPropPredDataset` instance that
    is compatible with PyG's :class:`~torch_geometric.loader.DataLoader`.
    """
    return _load_ogb_graph("ogbg-molhiv", root)


__all__ = ["ogbg_molhiv", "ogbn_arxiv"]

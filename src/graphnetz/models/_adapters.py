"""Task-kind adapters that turn any node-level encoder into a graph-level
classifier/regressor or a Deep Graph Infomax model.

This is the glue that lets GCN, GAT, GraphSAGE, and the Graph Transformer
plug into every benchmark task kind in the library, not just node
classification.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import DeepGraphInfomax, global_mean_pool


def _corruption(x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard DGI corruption: row-permute the node features."""
    return x[torch.randperm(x.size(0))], edge_index


class GraphLevelWrapper(nn.Module):
    """Wrap a node-level encoder for graph-level prediction.

    The encoder is expected to map a PyG ``Data`` batch to per-node features
    of shape ``[N, hidden_channels]``. The wrapper adds a global mean pool
    over the batch index and a linear classification/regression head.
    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(hidden_channels, out_channels)

    def forward(self, data: Data) -> torch.Tensor:
        z = self.encoder(data)
        pooled = global_mean_pool(z, data.batch)
        return self.head(pooled)


class _DataAdapter(nn.Module):
    """Adapt a ``forward(data)`` encoder to a ``forward(x, edge_index)`` API.

    PyG's :class:`DeepGraphInfomax` calls the encoder with positional
    ``(x, edge_index)`` but every model in the library accepts a ``Data``
    object. This shim builds a temporary ``Data`` for the inner call.
    """

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(Data(x=x, edge_index=edge_index))


class DGIWrapper(nn.Module):
    """Wrap any node-level encoder as a Deep Graph Infomax model.

    Mirrors the :class:`graphnetz.models.DGI` interface (``forward(data)``
    returning the ``(pos_z, neg_z, summary)`` triple, plus a ``loss(...)``
    helper) so the benchmark trainer does not need to special-case it.
    """

    def __init__(self, encoder: nn.Module, hidden_channels: int) -> None:
        super().__init__()
        self.model = DeepGraphInfomax(
            hidden_channels=hidden_channels,
            encoder=_DataAdapter(encoder),
            summary=lambda z, *_: torch.sigmoid(z.mean(dim=0)),
            corruption=_corruption,
        )

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(data.x, data.edge_index)

    def loss(
        self,
        pos_z: torch.Tensor,
        neg_z: torch.Tensor,
        summary: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.loss(pos_z, neg_z, summary)


class LinkPredWrapper(nn.Module):
    """Wrap any node-level encoder as a link predictor with a dot-product decoder.

    The wrapper exposes ``encode(data)`` returning per-node embeddings of shape
    ``[N, hidden_channels]`` and ``decode(z, edge_label_index)`` returning a
    ``[E]`` tensor of edge logits.
    """

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def encode(self, data: Data) -> torch.Tensor:
        return self.encoder(data)

    @staticmethod
    def decode(z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def forward(self, data: Data) -> torch.Tensor:
        z = self.encode(data)
        return self.decode(z, data.edge_label_index)


class RelationalLinkPredWrapper(nn.Module):
    """Wrap any node-level encoder as a relational link predictor (DistMult).

    Learns a relation embedding matrix and scores triples via
    ``(z[h] * r * z[t]).sum()`` (element-wise product, DistMult).
    """

    def __init__(self, encoder: nn.Module, hidden_channels: int, num_relations: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.relation_emb = nn.Embedding(num_relations, hidden_channels)

    def encode(self, data: Data) -> torch.Tensor:
        return self.encoder(data)

    def decode(self, z: torch.Tensor, edge_label_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        h = z[edge_label_index[0]]
        t = z[edge_label_index[1]]
        r = self.relation_emb(edge_type)
        return (h * r * t).sum(dim=-1)

    def forward(self, data: Data) -> torch.Tensor:
        z = self.encode(data)
        return self.decode(z, data.edge_label_index, data.edge_type)


__all__ = ["DGIWrapper", "GraphLevelWrapper", "LinkPredWrapper", "RelationalLinkPredWrapper"]

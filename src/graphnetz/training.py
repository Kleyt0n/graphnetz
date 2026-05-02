"""Lightweight training loops shared by the example notebooks.

Each function returns a plain ``dict`` of per-epoch metrics, ready to feed into
:func:`graphnetz.plotting.plot_history`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from tqdm.auto import tqdm


@runtime_checkable
class _LinkPredLike(Protocol):
    """Encoder + dot-product decoder API expected by ``train_link_prediction``."""

    def encode(self, data: Data) -> torch.Tensor: ...
    def decode(self, z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor: ...
    def train(self, mode: bool = ...) -> _LinkPredLike: ...
    def eval(self) -> _LinkPredLike: ...
    def parameters(self): ...  # type: ignore[no-untyped-def]


@runtime_checkable
class _RelationalLinkPredLike(Protocol):
    """Encoder + relational decoder API expected by the relational LP loop."""

    def encode(self, data: Data) -> torch.Tensor: ...
    def decode(self, z: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor: ...
    def train(self, mode: bool = ...) -> _RelationalLinkPredLike: ...
    def eval(self) -> _RelationalLinkPredLike: ...
    def parameters(self): ...  # type: ignore[no-untyped-def]


@runtime_checkable
class _DGILike(Protocol):
    """``forward(data) -> (pos_z, neg_z, summary)`` plus a ``loss`` helper."""

    def __call__(self, data: Data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def loss(self, pos_z: torch.Tensor, neg_z: torch.Tensor, summary: torch.Tensor) -> torch.Tensor: ...
    def train(self, mode: bool = ...) -> _DGILike: ...
    def parameters(self): ...  # type: ignore[no-untyped-def]


def _select_split_mask(mask: torch.Tensor) -> torch.Tensor:
    """Reduce PyG split masks to a flat 1-D bool tensor.

    PyG's ``HeterophilousGraphDataset`` and ``WikiCS`` ship masks of shape
    ``[num_nodes, num_splits]``. Pick the first split column so the
    standard trainers work without modification; users who want a
    different split can pre-select it before calling ``run_benchmark``.
    """
    if mask.dim() == 1:
        return mask
    return mask[:, 0]


def train_node_classification(
    model: torch.nn.Module,
    data: Data,
    epochs: int = 100,
    lr: float = 1e-2,
    weight_decay: float = 5e-4,
    verbose: bool = False,
) -> dict[str, list[float]]:
    """Train a node classifier with Planetoid-style train/val/test masks."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: dict[str, list[float]] = {"train_loss": [], "val_acc": [], "test_acc": []}
    train_mask = _select_split_mask(data.train_mask)
    val_mask = _select_split_mask(data.val_mask)
    test_mask = _select_split_mask(data.test_mask)
    iterator = tqdm(range(epochs), desc="Epochs", leave=False, disable=not verbose)
    for _ in iterator:
        model.train()
        opt.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(data).argmax(dim=1)
            val_acc = (pred[val_mask] == data.y[val_mask]).float().mean().item()
            test_acc = (pred[test_mask] == data.y[test_mask]).float().mean().item()
        history["train_loss"].append(loss.item())
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)
        if verbose:
            iterator.set_postfix(
                loss=f"{loss.item():.4f}",
                val=f"{val_acc:.4f}",
                test=f"{test_acc:.4f}",
            )
    return history


def train_graph_classification(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 1e-3,
    verbose: bool = False,
) -> dict[str, list[float]]:
    """Train a graph-level classifier.

    Handles single-label and multi-label datasets transparently:
    when ``batch.y`` is shaped ``[B, C]`` with float dtype (e.g.\
    LRGB ``Peptides-func``, OGB molhiv variants), the loss switches to
    binary cross-entropy with logits and the reported metric is the
    average correctly-classified label fraction.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: dict[str, list[float]] = {"train_loss": [], "val_acc": []}

    # Sniff one batch to decide single- vs. multi-label.
    sniff_batch = next(iter(train_loader))
    if sniff_batch.y is None:
        msg = "train_graph_classification requires graph-level `y` on each batch"
        raise ValueError(msg)
    multi_label = sniff_batch.y.dim() == 2 and sniff_batch.y.size(1) > 1

    iterator = tqdm(range(epochs), desc="Epochs", leave=False, disable=not verbose)
    for _ in iterator:
        model.train()
        total = 0.0
        n = 0
        for batch in train_loader:
            opt.zero_grad()
            out = model(batch)
            if multi_label:
                loss = F.binary_cross_entropy_with_logits(out, batch.y.float())
            else:
                loss = F.cross_entropy(out, batch.y.view(-1))
            loss.backward()
            opt.step()
            total += loss.item() * batch.num_graphs
            n += batch.num_graphs

        model.eval()
        correct = 0.0
        m = 0
        with torch.no_grad():
            for batch in val_loader:
                if multi_label:
                    pred = (model(batch) > 0).float()
                    correct += (pred == batch.y.float()).float().mean().item() * batch.num_graphs
                else:
                    pred = model(batch).argmax(dim=1)
                    correct += (pred == batch.y.view(-1)).sum().item()
                m += batch.num_graphs
        train_loss = total / max(n, 1)
        val_acc = correct / max(m, 1)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        if verbose:
            iterator.set_postfix(loss=f"{train_loss:.4f}", val=f"{val_acc:.4f}")
    return history


def train_graph_regression(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 1e-3,
    verbose: bool = False,
) -> dict[str, list[float]]:
    """Train a graph-level regressor (MSE loss, MAE on val)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: dict[str, list[float]] = {"train_loss": [], "val_mae": []}
    iterator = tqdm(range(epochs), desc="Epochs", leave=False, disable=not verbose)
    for _ in iterator:
        model.train()
        total = 0.0
        n = 0
        for batch in train_loader:
            opt.zero_grad()
            out = model(batch).view(-1)
            loss = F.mse_loss(out, batch.y.float().view(-1))
            loss.backward()
            opt.step()
            total += loss.item() * batch.num_graphs
            n += batch.num_graphs

        model.eval()
        mae = 0.0
        m = 0
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch).view(-1)
                mae += (out - batch.y.float().view(-1)).abs().sum().item()
                m += batch.num_graphs
        train_loss = total / max(n, 1)
        val_mae = mae / max(m, 1)
        history["train_loss"].append(train_loss)
        history["val_mae"].append(val_mae)
        if verbose:
            iterator.set_postfix(loss=f"{train_loss:.4f}", mae=f"{val_mae:.4f}")
    return history


def train_node_degree_regression(
    model: torch.nn.Module,
    data: Data,
    epochs: int = 100,
    lr: float = 1e-2,
    verbose: bool = False,
) -> dict[str, list[float]]:
    """Self-supervised node-level regression: predict log node degree."""
    target = torch.log1p(degree(data.edge_index[0], num_nodes=data.num_nodes).float())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: dict[str, list[float]] = {"train_loss": [], "val_mae": []}
    iterator = tqdm(range(epochs), desc="Epochs", leave=False, disable=not verbose)
    for _ in iterator:
        model.train()
        opt.zero_grad()
        out = model(data).view(-1)
        loss = F.mse_loss(out, target)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            eval_out = model(data).view(-1)
            mae = (eval_out - target).abs().mean().item()
        history["train_loss"].append(loss.item())
        history["val_mae"].append(mae)
        if verbose:
            iterator.set_postfix(loss=f"{loss.item():.4f}", mae=f"{mae:.4f}")
    return history


def train_dgi(
    model: _DGILike,
    data: Data,
    epochs: int = 100,
    lr: float = 1e-3,
    verbose: bool = False,
) -> dict[str, list[float]]:
    """Train a Deep Graph Infomax model (unsupervised)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: dict[str, list[float]] = {"dgi_loss": []}
    iterator = tqdm(range(epochs), desc="Epochs", leave=False, disable=not verbose)
    for _ in iterator:
        model.train()
        opt.zero_grad()
        pos_z, neg_z, summary = model(data)
        loss = model.loss(pos_z, neg_z, summary)
        loss.backward()
        opt.step()
        history["dgi_loss"].append(loss.item())
        if verbose:
            iterator.set_postfix(loss=f"{loss.item():.4f}")
    return history


def train_link_prediction(
    model: _LinkPredLike,
    train_data: Data,
    val_data: Data,
    test_data: Data,
    epochs: int = 100,
    lr: float = 1e-2,
    verbose: bool = False,
) -> dict[str, list[float]]:
    """Train a link predictor with binary cross-entropy on RandomLinkSplit.

    The model is expected to expose ``encode(data)`` returning per-node
    embeddings and ``decode(z, edge_label_index)`` returning per-edge scores
    (see :class:`graphnetz.models._adapters.LinkPredWrapper`).
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: dict[str, list[float]] = {"train_loss": [], "val_auc": [], "test_auc": []}
    iterator = tqdm(range(epochs), desc="Epochs", leave=False, disable=not verbose)
    for _ in iterator:
        model.train()
        opt.zero_grad()
        z = model.encode(train_data)
        logits = model.decode(z, train_data.edge_label_index)
        loss = F.binary_cross_entropy_with_logits(logits, train_data.edge_label.float())
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            z_eval = model.encode(train_data)
            val_logits = model.decode(z_eval, val_data.edge_label_index).cpu()
            test_logits = model.decode(z_eval, test_data.edge_label_index).cpu()
            val_auc = _safe_auc(val_data.edge_label.cpu(), val_logits)
            test_auc = _safe_auc(test_data.edge_label.cpu(), test_logits)
        history["train_loss"].append(float(loss.item()))
        history["val_auc"].append(val_auc)
        history["test_auc"].append(test_auc)
        if verbose:
            iterator.set_postfix(
                loss=f"{loss.item():.4f}",
                val=f"{val_auc:.4f}",
                test=f"{test_auc:.4f}",
            )
    return history


def _sample_negative_triples(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    num_nodes: int,
    num_negatives: int = 1,
    positive_set: set[tuple[int, int, int]] | None = None,
    max_resamples: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample negative triples by corrupting the tail entity.

    When ``positive_set`` is provided, rejection-resamples corruptions that
    coincide with an existing ``(h, r, t)`` triple ("filtered" setting).
    After ``max_resamples`` retries any survivors are kept as-is so the
    routine remains O(N) on dense graphs.
    """
    n = edge_index.size(1)
    device = edge_index.device
    heads = edge_index[0].repeat(num_negatives)
    rels = edge_type.repeat(num_negatives)
    neg_tails = torch.randint(0, num_nodes, (n * num_negatives,), device=device)

    if positive_set is not None and len(positive_set) > 0:
        heads_cpu = heads.cpu().tolist()
        rels_cpu = rels.cpu().tolist()
        tails_cpu = neg_tails.cpu().tolist()
        for _ in range(max_resamples):
            collisions = [i for i, t in enumerate(tails_cpu) if (heads_cpu[i], rels_cpu[i], t) in positive_set]
            if not collisions:
                break
            fresh = torch.randint(0, num_nodes, (len(collisions),)).tolist()
            for j, i in enumerate(collisions):
                tails_cpu[i] = fresh[j]
        neg_tails = torch.as_tensor(tails_cpu, dtype=neg_tails.dtype, device=device)

    neg_edge_index = torch.stack([heads, neg_tails])
    return neg_edge_index, rels


def _safe_auc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    """``roc_auc_score`` that returns NaN on degenerate (single-class) splits."""
    try:
        return float(roc_auc_score(y_true.numpy(), y_score.numpy()))
    except ValueError:
        return float("nan")


def _ensure_node_features(data: Data, hidden: int = 128) -> Data:
    """Fabricate node features if they are missing."""
    if getattr(data, "x", None) is not None:
        return data
    data = data.clone()
    data.x = torch.eye(data.num_nodes, hidden)
    return data


def _relational_eval_auc(
    model: _RelationalLinkPredLike,
    z: torch.Tensor,
    split_data: Data,
    num_nodes: int,
) -> float:
    """AUC over a split's positive triples and per-call random tail corruptions.

    Returns NaN when the AUC is undefined (e.g. an empty split or a
    degenerate case with only one class — happens on tiny KGs).
    """
    n = split_data.edge_index.size(1)
    if n == 0:
        return float("nan")
    device = split_data.edge_index.device
    pos_logits = model.decode(z, split_data.edge_index, split_data.edge_type).cpu()
    neg_tails = torch.randint(0, num_nodes, (n,), device=device)
    neg_index = torch.stack([split_data.edge_index[0], neg_tails])
    neg_logits = model.decode(z, neg_index, split_data.edge_type).cpu()
    y_true = torch.cat([torch.ones(n), torch.zeros(n)]).numpy()
    y_score = torch.cat([pos_logits, neg_logits]).numpy()
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def train_relational_link_prediction(
    model: _RelationalLinkPredLike,
    train_data: Data,
    val_data: Data,
    test_data: Data,
    epochs: int = 100,
    lr: float = 1e-2,
    verbose: bool = False,
) -> dict[str, list[float]]:
    """Train a relational link predictor (DistMult) on knowledge graph triples.

    The model is expected to expose ``encode(data)`` returning per-node
    embeddings and ``decode(z, edge_index, edge_type)`` returning per-edge
    scores (see :class:`graphnetz.models._adapters.RelationalLinkPredWrapper`).
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: dict[str, list[float]] = {"train_loss": [], "val_auc": [], "test_auc": []}

    # Ensure node features exist
    train_data = _ensure_node_features(train_data)
    val_data = _ensure_node_features(val_data)
    test_data = _ensure_node_features(test_data)

    pos_edge_index = train_data.edge_index
    pos_edge_type = train_data.edge_type
    pos_labels = torch.ones(pos_edge_index.size(1), device=pos_edge_index.device)
    neg_labels = torch.zeros(pos_edge_index.size(1), device=pos_edge_index.device)
    train_labels = torch.cat([pos_labels, neg_labels])
    positive_set = {
        (int(h), int(r), int(t))
        for h, r, t in zip(pos_edge_index[0].tolist(), pos_edge_type.tolist(), pos_edge_index[1].tolist(), strict=False)
    }

    iterator = tqdm(range(epochs), desc="Epochs", leave=False, disable=not verbose)
    for _ in iterator:
        model.train()
        # Resample negatives every epoch so the model sees fresh corruptions
        # (otherwise it overfits to a fixed negative set). Filter out
        # corruptions that collide with real training positives.
        neg_edge_index, neg_edge_type = _sample_negative_triples(
            pos_edge_index, pos_edge_type, train_data.num_nodes, positive_set=positive_set
        )
        train_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        train_edge_type = torch.cat([pos_edge_type, neg_edge_type])

        opt.zero_grad()
        z = model.encode(train_data)
        logits = model.decode(z, train_edge_index, train_edge_type)
        loss = F.binary_cross_entropy_with_logits(logits, train_labels)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            z_eval = model.encode(train_data)
            val_auc = _relational_eval_auc(model, z_eval, val_data, train_data.num_nodes)
            test_auc = _relational_eval_auc(model, z_eval, test_data, train_data.num_nodes)
        history["train_loss"].append(float(loss.item()))
        history["val_auc"].append(val_auc)
        history["test_auc"].append(test_auc)
        if verbose:
            iterator.set_postfix(
                loss=f"{loss.item():.4f}",
                val=f"{val_auc:.4f}",
                test=f"{test_auc:.4f}",
            )
    return history


__all__ = [
    "train_dgi",
    "train_graph_classification",
    "train_graph_regression",
    "train_link_prediction",
    "train_node_classification",
    "train_node_degree_regression",
    "train_relational_link_prediction",
]

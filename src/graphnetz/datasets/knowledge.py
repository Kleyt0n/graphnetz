"""Knowledge graph and language datasets.

Wraps PyG knowledge-graph benchmarks for relational link prediction.
"""

from torch_geometric.data import Data
from torch_geometric.datasets import RelLinkPredDataset, WordNet18RR

from graphnetz.datasets._netz import Netz


def fb15k_237(root: str) -> RelLinkPredDataset:
    """FB15k-237 relational link prediction benchmark."""
    return RelLinkPredDataset(root=root, name="FB15k-237")


class _WordNet18RRRel:
    """WN18-RR reshaped to match :class:`RelLinkPredDataset`'s interface.

    PyG's :class:`WordNet18RR` exposes edge-level ``train_mask`` / ``val_mask`` /
    ``test_mask`` over a single ``edge_index``; the benchmark dispatcher's
    relational path expects ``train_edge_index`` / ``valid_edge_index`` /
    ``test_edge_index`` (plus matching ``*_edge_type``) and ``num_relations``,
    as on :class:`RelLinkPredDataset`. This wrapper performs that conversion.
    """

    def __init__(self, base: WordNet18RR) -> None:
        d = base[0]
        ei, et = d.edge_index, d.edge_type
        train_ei, train_et = ei[:, d.train_mask], et[d.train_mask]
        valid_ei, valid_et = ei[:, d.val_mask], et[d.val_mask]
        test_ei, test_et = ei[:, d.test_mask], et[d.test_mask]
        self._data = Data(
            edge_index=train_ei,
            edge_type=train_et,
            train_edge_index=train_ei,
            train_edge_type=train_et,
            valid_edge_index=valid_ei,
            valid_edge_type=valid_et,
            test_edge_index=test_ei,
            test_edge_type=test_et,
            num_nodes=int(d.num_nodes),
        )
        self.num_relations = int(et.max()) + 1
        self.num_features = 0

    def __getitem__(self, idx: int) -> Data:
        if idx != 0:
            raise IndexError(idx)
        return self._data

    def __len__(self) -> int:
        return 1


def wordnet18rr(root: str) -> _WordNet18RRRel:
    """WordNet18-RR relational link prediction benchmark."""
    return _WordNet18RRRel(WordNet18RR(root=root))


def wordnet_netz(root: str) -> Netz:
    """WordNet semantic graph (Netzschleuder)."""
    return Netz(root=root, dataset_name="wordnet", network_name="wordnet")


__all__ = ["fb15k_237", "wordnet18rr", "wordnet_netz"]

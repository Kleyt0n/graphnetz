"""Physics and chemistry datasets.

Coverage:
- Molecules: PyG ``QM9``, ``ZINC``.
- Spin systems / lattices: synthetic 2D Ising lattice graphs (:class:`IsingLattice`).

Feynman diagrams, reaction networks, and large crystal-structure databases lack
canonical PyG-format datasets and are intentionally omitted.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import QM9, ZINC
from torch_geometric.utils import grid


def qm9(root: str) -> QM9:
    """QM9 quantum-chemistry benchmark (134k small molecules)."""
    return QM9(root=root)


def zinc(root: str, subset: bool = True, split: str = "train") -> ZINC:
    """ZINC molecular regression benchmark."""
    return ZINC(root=root, subset=subset, split=split)


class IsingLattice(InMemoryDataset):
    """Synthetic 2D Ising lattice ensemble.

    Each graph is an ``L x L`` square lattice with periodic-free boundaries;
    node features are Bernoulli spins drawn at temperature ``temperature``
    (Glauber-style independent sampling -- not a thermalised configuration but a
    cheap proxy useful for representation-learning benchmarks).
    """

    def __init__(
        self,
        root: str,
        num_graphs: int = 64,
        side: int = 10,
        temperature: float = 2.27,
        seed: int = 0,
    ) -> None:
        self.num_graphs = num_graphs
        self.side = side
        self.temperature = temperature
        self.seed = seed
        super().__init__(root)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> str:
        return f"ising_{self.num_graphs}_{self.side}_{self.temperature}_{self.seed}.pt"

    def process(self) -> None:
        gen = torch.Generator().manual_seed(self.seed)
        edge_index, _ = grid(self.side, self.side)
        n = self.side * self.side
        bias = torch.tanh(torch.tensor(1.0 / max(self.temperature, 1e-6))) / 2 + 0.5
        data_list: list[Data] = []
        for _ in range(self.num_graphs):
            spins = (torch.rand((n, 1), generator=gen) < bias).float() * 2 - 1
            data_list.append(Data(x=spins, edge_index=edge_index))
        self.save(data_list, self.processed_paths[0])


def ising_lattice(root: str, **kwargs: int | float) -> IsingLattice:
    return IsingLattice(root=root, **kwargs)  # type: ignore[arg-type]


__all__ = ["IsingLattice", "ising_lattice", "qm9", "zinc"]

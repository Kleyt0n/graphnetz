"""Combinatorial-optimization graph datasets.

All instances are synthetic; canonical PyG benchmarks do not cover this
category. The library provides:

- :class:`RandomTSP` / :func:`random_tsp` — Euclidean TSP (k-NN over 2D points).
- :class:`RandomVRP` / :func:`random_vrp` — Capacitated VRP (multi-depot k-NN).
- :class:`RandomMaxFlow` / :func:`random_maxflow` — random capacitated networks
  with a single source/sink, suitable for max-flow / min-cut benchmarks.
- :class:`RandomBipartiteMatching` / :func:`random_bipartite_matching` —
  bipartite assignment instances with random weights.
- :class:`RandomColoring` / :func:`random_coloring` — random Erdos-Renyi graphs
  for graph-coloring / max-cut experiments.
- :class:`RandomMaxCut` / :func:`random_maxcut` — alias of RandomColoring.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import erdos_renyi_graph, to_undirected


def _euclidean_knn_edges(pos: torch.Tensor, k: int) -> torch.Tensor:
    n = pos.size(0)
    dist = torch.cdist(pos, pos)
    knn = dist.topk(k + 1, largest=False).indices[:, 1:]
    src = torch.arange(n).repeat_interleave(k)
    dst = knn.reshape(-1)
    return to_undirected(torch.stack([src, dst], dim=0))


class RandomTSP(InMemoryDataset):
    """Euclidean TSP instances as k-NN graphs over 2D points."""

    def __init__(self, root: str, num_graphs: int = 128, num_nodes: int = 50, k: int = 5, seed: int = 0) -> None:
        self.num_graphs, self.num_nodes, self.k, self.seed = num_graphs, num_nodes, k, seed
        super().__init__(root)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> str:
        return f"tsp_{self.num_graphs}_{self.num_nodes}_{self.k}_{self.seed}.pt"

    def process(self) -> None:
        gen = torch.Generator().manual_seed(self.seed)
        data_list: list[Data] = []
        for _ in range(self.num_graphs):
            pos = torch.rand((self.num_nodes, 2), generator=gen)
            data_list.append(Data(x=pos, edge_index=_euclidean_knn_edges(pos, self.k), pos=pos))
        self.save(data_list, self.processed_paths[0])


class RandomVRP(InMemoryDataset):
    """Capacitated VRP instances: customers + multiple depots, k-NN connectivity.

    Node features are ``[x, y, demand, is_depot]``. Demands are zero for depots
    and uniform on ``(0, 1]`` for customers.
    """

    def __init__(
        self,
        root: str,
        num_graphs: int = 128,
        num_customers: int = 40,
        num_depots: int = 3,
        k: int = 6,
        seed: int = 0,
    ) -> None:
        self.num_graphs = num_graphs
        self.num_customers = num_customers
        self.num_depots = num_depots
        self.k = k
        self.seed = seed
        super().__init__(root)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> str:
        return f"vrp_{self.num_graphs}_{self.num_customers}_{self.num_depots}_{self.k}_{self.seed}.pt"

    def process(self) -> None:
        gen = torch.Generator().manual_seed(self.seed)
        n = self.num_customers + self.num_depots
        data_list: list[Data] = []
        for _ in range(self.num_graphs):
            pos = torch.rand((n, 2), generator=gen)
            demand = torch.rand((n, 1), generator=gen)
            is_depot = torch.zeros((n, 1))
            is_depot[: self.num_depots] = 1.0
            demand[: self.num_depots] = 0.0
            x = torch.cat([pos, demand, is_depot], dim=-1)
            data_list.append(Data(x=x, edge_index=_euclidean_knn_edges(pos, self.k), pos=pos))
        self.save(data_list, self.processed_paths[0])


class RandomMaxFlow(InMemoryDataset):
    """Random capacitated networks with a marked source/sink for max-flow tasks.

    Nodes 0 and ``num_nodes - 1`` are designated source and sink. Edges carry
    a positive capacity stored in ``edge_attr``.
    """

    def __init__(
        self,
        root: str,
        num_graphs: int = 128,
        num_nodes: int = 30,
        edge_prob: float = 0.2,
        seed: int = 0,
    ) -> None:
        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self.edge_prob = edge_prob
        self.seed = seed
        super().__init__(root)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> str:
        return f"maxflow_{self.num_graphs}_{self.num_nodes}_{self.edge_prob}_{self.seed}.pt"

    def process(self) -> None:
        # Use a *local* generator via a saved/restored RNG fork so we don't
        # mutate the caller's torch RNG state. ``erdos_renyi_graph`` reads
        # the global RNG, so we have to bracket the call with fork_rng.
        gen = torch.Generator().manual_seed(self.seed)
        data_list: list[Data] = []
        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            for _ in range(self.num_graphs):
                edge_index = erdos_renyi_graph(self.num_nodes, self.edge_prob, directed=True)
                capacity = torch.rand((edge_index.size(1), 1), generator=gen) * 9 + 1  # in [1, 10]
                x = torch.zeros((self.num_nodes, 2))
                x[0, 0] = 1.0  # source flag
                x[-1, 1] = 1.0  # sink flag
                data_list.append(Data(x=x, edge_index=edge_index, edge_attr=capacity))
        self.save(data_list, self.processed_paths[0])


class RandomBipartiteMatching(InMemoryDataset):
    """Random bipartite assignment instances with weighted edges.

    Two sides of size ``size`` are connected by a Bernoulli mask of probability
    ``edge_prob``; edge weights are uniform on ``(0, 1]`` and stored in
    ``edge_attr``. Node features mark each node's side.
    """

    def __init__(
        self,
        root: str,
        num_graphs: int = 128,
        size: int = 25,
        edge_prob: float = 0.4,
        seed: int = 0,
    ) -> None:
        self.num_graphs = num_graphs
        self.size = size
        self.edge_prob = edge_prob
        self.seed = seed
        super().__init__(root)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> str:
        return f"matching_{self.num_graphs}_{self.size}_{self.edge_prob}_{self.seed}.pt"

    def process(self) -> None:
        gen = torch.Generator().manual_seed(self.seed)
        data_list: list[Data] = []
        for _ in range(self.num_graphs):
            mask = torch.rand((self.size, self.size), generator=gen) < self.edge_prob
            left, right = mask.nonzero(as_tuple=True)
            edge_index = torch.stack([left, right + self.size], dim=0)
            edge_index = to_undirected(edge_index)
            weights = torch.rand(edge_index.size(1), 1, generator=gen)
            x = torch.zeros((2 * self.size, 2))
            x[: self.size, 0] = 1.0
            x[self.size :, 1] = 1.0
            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=weights))
        self.save(data_list, self.processed_paths[0])


class RandomColoring(InMemoryDataset):
    """Random Erdos-Renyi graphs for graph-coloring and max-cut benchmarks."""

    def __init__(
        self,
        root: str,
        num_graphs: int = 128,
        num_nodes: int = 40,
        edge_prob: float = 0.15,
        seed: int = 0,
    ) -> None:
        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self.edge_prob = edge_prob
        self.seed = seed
        super().__init__(root)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> str:
        return f"coloring_{self.num_graphs}_{self.num_nodes}_{self.edge_prob}_{self.seed}.pt"

    def process(self) -> None:
        # Bracket ``erdos_renyi_graph`` (which reads the global torch RNG)
        # in ``fork_rng`` so the caller's RNG state is restored after this
        # builder runs.
        data_list: list[Data] = []
        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            for _ in range(self.num_graphs):
                edge_index = erdos_renyi_graph(self.num_nodes, self.edge_prob, directed=False)
                x = torch.ones((self.num_nodes, 1))
                data_list.append(Data(x=x, edge_index=edge_index))
        self.save(data_list, self.processed_paths[0])


# Backward-compatible alias: max-cut and coloring share this generator.
RandomMaxCut = RandomColoring


def random_tsp(root: str, **kwargs: int | float) -> RandomTSP:
    return RandomTSP(root=root, **kwargs)  # type: ignore[arg-type]


def random_vrp(root: str, **kwargs: int | float) -> RandomVRP:
    return RandomVRP(root=root, **kwargs)  # type: ignore[arg-type]


def random_maxflow(root: str, **kwargs: int | float) -> RandomMaxFlow:
    return RandomMaxFlow(root=root, **kwargs)  # type: ignore[arg-type]


def random_bipartite_matching(root: str, **kwargs: int | float) -> RandomBipartiteMatching:
    return RandomBipartiteMatching(root=root, **kwargs)  # type: ignore[arg-type]


def random_coloring(root: str, **kwargs: int | float) -> RandomColoring:
    return RandomColoring(root=root, **kwargs)  # type: ignore[arg-type]


def random_maxcut(root: str, **kwargs: int | float) -> RandomColoring:
    return RandomColoring(root=root, **kwargs)  # type: ignore[arg-type]


__all__ = [
    "RandomBipartiteMatching",
    "RandomColoring",
    "RandomMaxCut",
    "RandomMaxFlow",
    "RandomTSP",
    "RandomVRP",
    "random_bipartite_matching",
    "random_coloring",
    "random_maxcut",
    "random_maxflow",
    "random_tsp",
    "random_vrp",
]

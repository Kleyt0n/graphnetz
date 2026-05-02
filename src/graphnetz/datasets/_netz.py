import math
import os
import os.path as osp
import zipfile
from collections.abc import Callable

import networkx as nx
import pandas as pd
import requests  # type: ignore[import-untyped]
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import degree, from_networkx

NETZ_API = "https://networks.skewed.de/api/net"
NETZ_FILES = "https://networks.skewed.de/net"

# Above this many nodes, an N x N identity feature matrix would explode memory
# (~5 GB at 35k nodes, ~11 TB at 1.7M nodes), so we fall back to a compact
# structural feature (log-degree + constant bias).
_EYE_FEATURE_LIMIT = 8192


class Netz(InMemoryDataset):
    """Netzschleuder network dataset.

    Parameters
    ----------
    root : str
        Root directory where the dataset should be saved.
    dataset_name : str
        Name of the dataset from the Netzschleuder repository.
    network_name : str
        Name of the network within the dataset.
    transform, pre_transform : callable, optional
        Standard PyG transform hooks.

    Examples
    --------
    >>> dataset = Netz(
    ...     root="data", dataset_name="urban_streets", network_name="brasilia"
    ... )
    >>> dataset[0]
    Data(...)

    References
    ----------
    Tiago P. Peixoto. (2020). The Netzschleuder network catalogue and repository.
    https://networks.skewed.de/
    """

    def __init__(
        self,
        root: str,
        dataset_name: str = "urban_streets",
        network_name: str = "brasilia",
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        multigraph: bool = False,
    ) -> None:
        self.dataset_name = dataset_name
        self.network_name = network_name
        self.network_url = f"{NETZ_FILES}/{dataset_name}/files/{network_name}.csv.zip"
        # ``multigraph=True`` preserves parallel edges and self-loops (needed for
        # multiplex / transit / airline networks); the default collapses them
        # via :class:`networkx.Graph` for compatibility with existing PyG flows.
        self.multigraph = multigraph
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.dataset_name, self.network_name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.dataset_name, self.network_name, "processed")

    @property
    def raw_file_names(self) -> list[str]:
        return ["data.csv.zip"]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self) -> None:
        os.makedirs(self.raw_dir, exist_ok=True)
        raw_path = osp.join(self.raw_dir, self.raw_file_names[0])
        # Stream to disk so multi-GB Netzschleuder networks (e.g. as_skitter)
        # don't have to fit in RAM.
        with requests.get(self.network_url, timeout=60, stream=True) as response:
            response.raise_for_status()
            with open(raw_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)

    def process(self) -> None:
        import io

        raw_path = osp.join(self.raw_dir, self.raw_file_names[0])
        with zipfile.ZipFile(raw_path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)

        edges_file = osp.join(self.raw_dir, "edges.csv")
        normalized = self._normalize_header_in_memory(edges_file)

        edges = pd.read_csv(io.StringIO(normalized))
        create_using = nx.MultiGraph if self.multigraph else nx.Graph
        graph = nx.from_pandas_edgelist(edges, source="source", target="target", create_using=create_using)

        data = from_networkx(graph)
        data.x = self._default_features(data.num_nodes, data.edge_index)

        if self.pre_transform is not None:
            data = self.pre_transform(data)
        self.save([data], self.processed_paths[0])

    @staticmethod
    def _default_features(num_nodes: int, edge_index: torch.Tensor) -> torch.Tensor:
        if num_nodes <= _EYE_FEATURE_LIMIT:
            return torch.eye(num_nodes)
        deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
        log_deg = torch.log1p(deg) / math.log(max(num_nodes, 2))
        ones = torch.ones(num_nodes)
        return torch.stack([log_deg, log_deg.pow(2), ones], dim=1)

    @staticmethod
    def _normalize_header_in_memory(edges_file: str) -> str:
        """Return the edges CSV with its first line stripped of leading ``#`` /
        whitespace, without mutating the on-disk raw file (PyG semantics)."""
        with open(edges_file) as f:
            lines = f.readlines()
        if lines:
            lines[0] = lines[0].lstrip("#").replace(" ", "").strip() + "\n"
        return "".join(lines)

    def __repr__(self) -> str:
        return f"{self.dataset_name}({self.network_name})"


def download_all_networks_netz(
    root: str,
    dataset_name: str,
    transform: Callable | None = None,
    pre_transform: Callable | None = None,
) -> None:
    """Download and process every network in a Netzschleuder dataset."""
    response = requests.get(f"{NETZ_API}/{dataset_name}", timeout=60)
    response.raise_for_status()
    for network_name in response.json()["nets"]:
        Netz(root, dataset_name, network_name, transform, pre_transform)

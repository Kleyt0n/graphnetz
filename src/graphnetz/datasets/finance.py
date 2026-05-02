"""Finance and economics networks.

Coverage:

- Trade: ``product_space`` (economic complexity).
- Ownership / corporate control: ``board_directors`` (Norwegian boards).
- Innovation: ``us_patents`` citation network.
- Transactions / fraud / AML: PyG ``EllipticBitcoinDataset`` (illicit-wallet
  detection on Bitcoin transactions).

Inter-bank exposure datasets are typically confidential and have no canonical
public benchmark.
"""

from torch_geometric.datasets import EllipticBitcoinDataset

from graphnetz.datasets._netz import Netz


def product_space(root: str) -> Netz:
    """Product space of international trade (economic complexity)."""
    return Netz(root=root, dataset_name="product_space", network_name="product_space")


def board_directors(root: str, network_name: str = "net1m_2002-05-01") -> Netz:
    """Norwegian boards of directors interlock network (snapshot)."""
    return Netz(root=root, dataset_name="board_directors", network_name=network_name)


def us_patents(root: str) -> Netz:
    """US patents citation network."""
    return Netz(root=root, dataset_name="us_patents", network_name="us_patents")


def elliptic_bitcoin(root: str) -> EllipticBitcoinDataset:
    """Elliptic Bitcoin transactions dataset for illicit-wallet detection."""
    return EllipticBitcoinDataset(root=root)


__all__ = ["board_directors", "elliptic_bitcoin", "product_space", "us_patents"]

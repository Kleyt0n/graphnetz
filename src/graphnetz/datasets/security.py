"""Security-related graph datasets.

Coverage:

- Terrorism association networks (Krebs 9/11; Madrid 2004 train bombings) via
  Netzschleuder.
- Malware function call graphs: PyG ``MalNetTiny`` (5 malware families).

Generic attack graphs and threat-intelligence/IoC graphs lack canonical public
benchmarks and are intentionally omitted.
"""

from torch_geometric.datasets import MalNetTiny

from graphnetz.datasets._netz import Netz


def terrorists_911(root: str) -> Netz:
    """Krebs 9/11 terrorist association network."""
    return Netz(root=root, dataset_name="terrorists_911", network_name="terrorists_911")


def train_terrorists(root: str) -> Netz:
    """Madrid 2004 train bombing terrorist network."""
    return Netz(root=root, dataset_name="train_terrorists", network_name="train_terrorists")


def malnet_tiny(root: str, split: str = "train") -> MalNetTiny:
    """MalNet-Tiny: 5 malware family function-call graphs."""
    return MalNetTiny(root=root, split=split)


__all__ = ["malnet_tiny", "terrorists_911", "train_terrorists"]

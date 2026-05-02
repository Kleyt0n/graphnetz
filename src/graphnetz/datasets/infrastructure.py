"""Infrastructure and physical-system networks (Netzschleuder)."""

from graphnetz.datasets._netz import Netz


def power_grid(root: str) -> Netz:
    """US Western power grid."""
    return Netz(root=root, dataset_name="power", network_name="power")


def euroroad(root: str) -> Netz:
    """European road network."""
    return Netz(root=root, dataset_name="euroroad", network_name="euroroad")


def us_roads(root: str, network_name: str = "DC") -> Netz:
    """US road network for a given state (e.g. ``DC``, ``CA``)."""
    return Netz(root=root, dataset_name="us_roads", network_name=network_name)


def eu_airlines(root: str) -> Netz:
    """European airline route multiplex."""
    return Netz(root=root, dataset_name="eu_airlines", network_name="eu_airlines")


def london_transport(root: str) -> Netz:
    """London transport multiplex (rail + bus + underground)."""
    return Netz(root=root, dataset_name="london_transport", network_name="london_transport")


def urban_streets(root: str, network_name: str = "brasilia") -> Netz:
    """Urban street network for a given city (e.g. ``brasilia``, ``manhattan``)."""
    return Netz(root=root, dataset_name="urban_streets", network_name=network_name)


__all__ = [
    "eu_airlines",
    "euroroad",
    "london_transport",
    "power_grid",
    "urban_streets",
    "us_roads",
]

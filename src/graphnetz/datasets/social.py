"""Social and information network datasets.

Coverage:

- Social: ``karate``, ``facebook_friends`` (Netzschleuder).
- Citation/collaboration: Planetoid (Cora, CiteSeer, PubMed) + Netz ``dblp_coauthor``.
- Web/hyperlink: PyG ``WikiCS``.
- Heterophilic node classification: PyG ``HeterophilousGraphDataset``
  (Roman-empire, Amazon-ratings, Minesweeper, Tolokers, Questions) — the
  current SOTA stress test for GNNs whose accuracy collapses outside the
  homophilic Planetoid setting (Platonov et al., NeurIPS 2023).
- Communication: Netz ``dnc`` (Democratic National Committee email leak).
- Recommendation: PyG ``MovieLens100K``.
- Open Graph Benchmark (optional ``ogb`` extra): ``ogbn_arxiv`` (arXiv
  citation network for node classification), ``ogbl_collab``
  (collaboration network for link prediction).
"""

from torch_geometric.data import Data
from torch_geometric.datasets import (
    HeterophilousGraphDataset,
    MovieLens100K,
    Planetoid,
    WikiCS,
)

from graphnetz.datasets._netz import Netz
from graphnetz.datasets._ogb import load_ogb_link, load_ogb_node


def cora(root: str) -> Planetoid:
    """Cora citation network (2708 nodes, 7 classes)."""
    return Planetoid(root=root, name="Cora")


def citeseer(root: str) -> Planetoid:
    """CiteSeer citation network (3327 nodes, 6 classes)."""
    return Planetoid(root=root, name="CiteSeer")


def pubmed(root: str) -> Planetoid:
    """PubMed citation network (19717 nodes, 3 classes)."""
    return Planetoid(root=root, name="PubMed")


def karate(root: str) -> Netz:
    """Zachary's karate club (the canonical small social network)."""
    return Netz(root=root, dataset_name="karate", network_name="karate")


def facebook_friends(root: str) -> Netz:
    """Facebook ego friendship network (Netzschleuder)."""
    return Netz(root=root, dataset_name="facebook_friends", network_name="facebook_friends")


def dblp_coauthor(root: str) -> Netz:
    """DBLP co-authorship network (Netzschleuder)."""
    return Netz(root=root, dataset_name="dblp_coauthor", network_name="dblp_coauthor")


def wikics(root: str) -> WikiCS:
    """Wikipedia computer-science article hyperlink graph."""
    return WikiCS(root=root)


def dnc_emails(root: str) -> Netz:
    """DNC email communication network (Netzschleuder)."""
    return Netz(root=root, dataset_name="dnc", network_name="dnc")


def movielens100k(root: str) -> MovieLens100K:
    """MovieLens 100K user-item bipartite ratings graph."""
    return MovieLens100K(root=root)


def roman_empire(root: str) -> HeterophilousGraphDataset:
    """Roman-empire heterophilic node-classification benchmark."""
    return HeterophilousGraphDataset(root=root, name="Roman-empire")


def amazon_ratings(root: str) -> HeterophilousGraphDataset:
    """Amazon-ratings heterophilic node-classification benchmark."""
    return HeterophilousGraphDataset(root=root, name="Amazon-ratings")


def minesweeper(root: str) -> HeterophilousGraphDataset:
    """Minesweeper heterophilic node-classification benchmark."""
    return HeterophilousGraphDataset(root=root, name="Minesweeper")


def tolokers(root: str) -> HeterophilousGraphDataset:
    """Tolokers heterophilic node-classification benchmark."""
    return HeterophilousGraphDataset(root=root, name="Tolokers")


def questions(root: str) -> HeterophilousGraphDataset:
    """Questions heterophilic node-classification benchmark."""
    return HeterophilousGraphDataset(root=root, name="Questions")


def ogbn_arxiv(root: str) -> Data:
    """OGB arXiv citation network (~169 K nodes, 40 subject classes)."""
    return load_ogb_node("ogbn-arxiv", root)


def ogbl_collab(root: str) -> Data:
    """OGB collaboration network (~235 K author nodes, 128-d features).

    Returns a single PyG ``Data`` graph; the benchmark runner re-splits
    via ``RandomLinkSplit`` rather than using OGB's official edge split.
    """
    return load_ogb_link("ogbl-collab", root)


__all__ = [
    "amazon_ratings",
    "citeseer",
    "cora",
    "dblp_coauthor",
    "dnc_emails",
    "facebook_friends",
    "karate",
    "minesweeper",
    "movielens100k",
    "ogbl_collab",
    "ogbn_arxiv",
    "pubmed",
    "questions",
    "roman_empire",
    "tolokers",
    "wikics",
]

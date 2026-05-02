"""Health and biology datasets.

Coverage:

- Molecular: PyG TUDataset (MUTAG, PROTEINS, ENZYMES).
- Long-range peptides: PyG ``LRGBDataset`` (Peptides-func graph
  classification, Peptides-struct graph regression — Dwivedi et al., NeurIPS
  2022 long-range graph benchmark).
- Protein-protein interaction: PyG ``PPI`` (inductive multi-graph).
- Metabolic: Netzschleuder ``celegans_metabolic``.
- Brain connectomes: Netzschleuder ``budapest_connectome``.
- Epidemiology: Netzschleuder ``sp_hospital`` and ``sp_high_school`` contact graphs.

Patient-disease-treatment knowledge graphs have no canonical free dataset and
are intentionally omitted.
"""

from torch_geometric.datasets import PPI, LRGBDataset, TUDataset

from graphnetz.datasets._netz import Netz


def mutag(root: str) -> TUDataset:
    """Mutagenicity: 188 molecules, binary class."""
    return TUDataset(root=root, name="MUTAG")


def proteins(root: str) -> TUDataset:
    """Proteins: 1113 graphs, binary class."""
    return TUDataset(root=root, name="PROTEINS")


def enzymes(root: str) -> TUDataset:
    """Enzymes: 600 graphs, 6 classes."""
    return TUDataset(root=root, name="ENZYMES")


def ppi(root: str, split: str = "train") -> PPI:
    """Protein-protein interaction (inductive node multi-label classification)."""
    return PPI(root=root, split=split)


def celegans(root: str) -> Netz:
    """C. elegans metabolic network (Netzschleuder)."""
    return Netz(root=root, dataset_name="celegans_metabolic", network_name="celegans_metabolic")


def budapest_connectome(root: str, network_name: str = "100m_avg") -> Netz:
    """Budapest reference connectome (mean connectivity across 100 subjects)."""
    return Netz(root=root, dataset_name="budapest_connectome", network_name=network_name)


def hospital_contacts(root: str) -> Netz:
    """Sociopatterns hospital ward face-to-face contact network."""
    return Netz(root=root, dataset_name="sp_hospital", network_name="sp_hospital")


def high_school_contacts(root: str) -> Netz:
    """Sociopatterns high-school contact network."""
    return Netz(root=root, dataset_name="sp_high_school", network_name="sp_high_school")


def peptides_func(root: str, split: str = "train") -> LRGBDataset:
    """Peptides-func: long-range graph classification (10-way multilabel)."""
    return LRGBDataset(root=root, name="Peptides-func", split=split)


def peptides_struct(root: str, split: str = "train") -> LRGBDataset:
    """Peptides-struct: long-range graph regression (11 structural targets)."""
    return LRGBDataset(root=root, name="Peptides-struct", split=split)


__all__ = [
    "budapest_connectome",
    "celegans",
    "enzymes",
    "high_school_contacts",
    "hospital_contacts",
    "mutag",
    "peptides_func",
    "peptides_struct",
    "ppi",
    "proteins",
]

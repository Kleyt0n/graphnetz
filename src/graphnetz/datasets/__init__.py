"""Dataset registry, organized by research category and task.

Each category exposes thin loader functions that return a ``torch_geometric``
dataset (PyG built-in or :class:`~graphnetz.datasets.Netz`).  The
:data:`LOADER_REGISTRY` table maps every loader to its category and the task it can serve, mirroring the structure used by
:data:`graphnetz.benchmark.BENCHMARK_TASKS` (``[category][task_type] -> [...]``).

Categories
----------
- ``combinatorial``: synthetic TSP, VRP, max-cut, max-flow, matching, coloring.
- ``biology``: MUTAG, PROTEINS, ENZYMES, PPI, Peptides (LRGB), C. elegans,
  connectomes, contact networks.
- ``social``: Cora, CiteSeer, PubMed, WikiCS, Heterophilous (Roman-empire,
  Amazon-ratings, Minesweeper, Tolokers, Questions), Karate, ego networks.
- ``knowledge``: FB15k-237, WordNet18-RR, Netz WordNet.
- ``infrastructure``: power grids, road and air networks.
- ``finance``: product space, board interlocks, patents, Elliptic Bitcoin.
- ``computing``: AS topology, Skitter, BGP route views.
- ``vision``: MNIST/CIFAR10 superpixel graphs, ModelNet, ShapeNet.
- ``physics``: QM9, ZINC, Ising lattice.
- ``security``: terrorist association networks, MalNet-Tiny.

Tasks
----------
``node_cls``, ``graph_cls``, ``graph_reg``, ``link_pred``. A loader may
serve more than one task type (e.g. ``cora`` is used for both ``node_cls`` and
``link_pred``). Deep Graph Infomax is *not* a task: it is a
self-supervised training objective whose metric is its own loss, so the
benchmark routes unlabelled graphs through ``link_pred`` (a real held-out
edge split with an AUC metric) instead. ``train_dgi`` and the
``DGIWrapper`` adapter remain available as utilities for users who want
unsupervised pre-training on top of any encoder.
"""

import importlib.util

from graphnetz.datasets import (
    biology,
    combinatorial,
    computing,
    finance,
    infrastructure,
    knowledge,
    physics,
    security,
    social,
    vision,
)
from graphnetz.datasets._netz import Netz, download_all_networks_netz

# OGB loaders live directly inside the domain modules (``social.ogbn_arxiv``,
# ``biology.ogbg_molhiv``, etc.). They raise a clear ``ImportError`` at
# call time when the optional ``ogb`` package is missing; the registries
# below only reference them when ``ogb`` is importable so the curated
# benchmark stays runnable without the extra installed.
_HAS_OGB = importlib.util.find_spec("ogb") is not None

CATEGORIES = {
    "combinatorial": combinatorial,
    "biology": biology,
    "social": social,
    "knowledge": knowledge,
    "infrastructure": infrastructure,
    "finance": finance,
    "computing": computing,
    "vision": vision,
    "physics": physics,
    "security": security,
}
"""Mapping of category name to the dataset module that implements it."""


# Source-of-truth taxonomy: category -> task -> [(loader_name, callable)].
# The benchmark dispatcher curates a subset of these; users can also load any
# loader directly via the category module.
LOADER_REGISTRY: dict[str, dict[str, list[tuple[str, object]]]] = {
    "combinatorial": {
        # Synthetic combinatorial generators ship no graph-level ``y``,
        # so they cannot serve graph_cls / graph_reg without a labelling
        # convention. They enter the benchmark exclusively through
        # link_pred (a real held-out edge split with an AUC metric).
        "link_pred": [
            ("random_bipartite_matching", combinatorial.random_bipartite_matching),
            ("random_tsp", combinatorial.random_tsp),
            ("random_vrp", combinatorial.random_vrp),
            ("random_coloring", combinatorial.random_coloring),
            ("random_maxcut", combinatorial.random_maxcut),
            ("random_maxflow", combinatorial.random_maxflow),
        ],
    },
    "biology": {
        "graph_cls": [
            ("mutag", biology.mutag),
            ("proteins", biology.proteins),
            ("enzymes", biology.enzymes),
            ("peptides_func", biology.peptides_func),
        ],
        "graph_reg": [("peptides_struct", biology.peptides_struct)],
        # PPI is a multi-graph inductive dataset and does not fit the
        # single-``Data`` + ``train_mask`` shape that
        # ``train_node_classification`` expects, so it enters the
        # benchmark through ``link_pred`` -- ``RandomLinkSplit`` on the
        # first graph yields a real held-out-edge AUC, matching the
        # protocol the framework uses for the other unlabelled biology
        # graphs (celegans, connectome, contact networks).
        "link_pred": [
            ("celegans", biology.celegans),
            ("budapest_connectome", biology.budapest_connectome),
            ("hospital_contacts", biology.hospital_contacts),
            ("high_school_contacts", biology.high_school_contacts),
            ("ppi", biology.ppi),
        ],
    },
    "social": {
        "node_cls": [
            ("cora", social.cora),
            ("citeseer", social.citeseer),
            ("pubmed", social.pubmed),
            ("wikics", social.wikics),
            ("roman_empire", social.roman_empire),
            ("amazon_ratings", social.amazon_ratings),
            ("minesweeper", social.minesweeper),
            ("tolokers", social.tolokers),
            ("questions", social.questions),
        ],
        "link_pred": [
            ("cora", social.cora),
            ("citeseer", social.citeseer),
            ("pubmed", social.pubmed),
            ("movielens100k", social.movielens100k),
            ("karate", social.karate),
            ("facebook_friends", social.facebook_friends),
            ("dblp_coauthor", social.dblp_coauthor),
            ("dnc_emails", social.dnc_emails),
        ],
    },
    "knowledge": {
        "link_pred": [
            ("fb15k_237", knowledge.fb15k_237),
            ("wordnet18rr", knowledge.wordnet18rr),
            ("wordnet_netz", knowledge.wordnet_netz),
        ],
    },
    "infrastructure": {
        "link_pred": [
            ("power_grid", infrastructure.power_grid),
            ("euroroad", infrastructure.euroroad),
            ("us_roads", infrastructure.us_roads),
            ("eu_airlines", infrastructure.eu_airlines),
            ("london_transport", infrastructure.london_transport),
            ("urban_streets", infrastructure.urban_streets),
        ],
    },
    "finance": {
        "node_cls": [("elliptic_bitcoin", finance.elliptic_bitcoin)],
        "link_pred": [
            ("product_space", finance.product_space),
            ("board_directors", finance.board_directors),
            ("us_patents", finance.us_patents),
        ],
    },
    "computing": {
        "link_pred": [
            ("internet_as", computing.internet_as),
            ("topology", computing.topology),
            ("as_skitter", computing.as_skitter),
            ("route_views", computing.route_views),
        ],
    },
    "vision": {
        "graph_cls": [
            ("mnist_superpixels", vision.mnist_superpixels),
            ("cifar10_superpixels", vision.cifar10_superpixels),
            ("modelnet10", vision.modelnet10),
            ("modelnet40", vision.modelnet40),
        ],
        "node_cls": [("shapenet", vision.shapenet)],
    },
    "physics": {
        "graph_reg": [
            ("qm9", physics.qm9),
            ("zinc", physics.zinc),
        ],
        "link_pred": [("ising_lattice", physics.ising_lattice)],
    },
    "security": {
        "graph_cls": [("malnet_tiny", security.malnet_tiny)],
        "link_pred": [
            ("terrorists_911", security.terrorists_911),
            ("train_terrorists", security.train_terrorists),
        ],
    },
}

if _HAS_OGB:
    # OGB loaders live in the domain modules; we only fold them into the
    # registry when the ``ogb`` extra is importable so default benchmark
    # runs stay green without it. Mapping rationale:
    #   - ogbn_arxiv     -> social   (citation network)
    #   - ogbn_products  -> finance  (e-commerce co-purchase, alongside
    #                                  product_space)
    #   - ogbg_molhiv    -> biology  (molecular bioactivity)
    #   - ogbg_molpcba   -> biology
    #   - ogbl_collab    -> social   (collaboration / coauthorship)
    LOADER_REGISTRY["social"]["node_cls"].append(("ogbn_arxiv", social.ogbn_arxiv))
    LOADER_REGISTRY["social"]["link_pred"].append(("ogbl_collab", social.ogbl_collab))
    LOADER_REGISTRY["finance"]["node_cls"].append(("ogbn_products", finance.ogbn_products))
    LOADER_REGISTRY["biology"]["graph_cls"].append(("ogbg_molhiv", biology.ogbg_molhiv))
    LOADER_REGISTRY["biology"]["graph_cls"].append(("ogbg_molpcba", biology.ogbg_molpcba))


def list_datasets(
    category: str | None = None,
    task: str | None = None,
) -> dict[str, dict[str, list[str]]]:
    """Return loader names organized by category and task.

    Output shape: ``{category: {task_type: [loader_name, ...]}}``. Pass
    ``category`` and/or ``task`` to restrict the view.
    """
    cats = [category] if category is not None else list(LOADER_REGISTRY)
    out: dict[str, dict[str, list[str]]] = {}
    for c in cats:
        per_cat = LOADER_REGISTRY.get(c, {})
        tasks = [task] if task is not None else list(per_cat)
        out[c] = {k: [name for name, _ in per_cat.get(k, [])] for k in tasks if k in per_cat}
    return out


__all__ = [
    "CATEGORIES",
    "LOADER_REGISTRY",
    "Netz",
    "biology",
    "combinatorial",
    "computing",
    "download_all_networks_netz",
    "finance",
    "infrastructure",
    "knowledge",
    "list_datasets",
    "physics",
    "security",
    "social",
    "vision",
]

# Dataset taxonomy

GraphNetz organises **58 loaders across 10 scientific categories**, each
declaring the task kinds it can serve. The taxonomy is the single source of
truth: the curated benchmark, the per-category notebooks, and the documented
loader names all derive from
{py:data}`graphnetz.datasets.LOADER_REGISTRY`.

```python
from graphnetz.datasets import LOADER_REGISTRY, list_datasets

list_datasets(category="biology")
# {'biology': {'graph_cls': [...], 'graph_reg': [...], 'node_cls': [...], 'link_pred': [...]}}
```

## Task kinds

Every cell in the benchmark — `(category, task_kind, dataset, model, seed)` —
maps to one of four task families. The default metric for each is what the
report headlines:

| Symbol | Kind | Default metric | Adapter |
|---|---|---|---|
| `node_cls` | Node classification | test accuracy | encoder used directly |
| `graph_cls` | Graph classification | val accuracy | mean-pool + linear head |
| `graph_reg` | Graph regression | val MAE | mean-pool + linear head |
| `link_pred` | Link prediction | test ROC-AUC | dot-product (homogeneous) or DistMult (relational) decoder |

```{note}
Unlabelled graphs (Netzschleuder networks, synthetic combinatorial
instances, the Ising lattice) enter the benchmark through `link_pred` on a
held-out edge split, so every cell carries a real held-out metric — there
is no self-supervised pretext loss in the headline report.
```

## Categories

| Category | # | Task kinds | Loaders |
|---|---:|---|---|
| **Combinatorial** | 6 | GC · GR · LP | random TSP, VRP, max-flow, bipartite matching, coloring, max-cut |
| **Biology** | 10 | GC · GR · NC · LP | MUTAG, PROTEINS, ENZYMES, Peptides-func/struct, PPI, *C. elegans*, Budapest connectome, hospital / high-school contacts |
| **Social** | 14 | NC · LP | Cora, CiteSeer, PubMed, WikiCS, Roman-empire, Amazon-ratings, Minesweeper, Tolokers, Questions, MovieLens-100k, Karate, Facebook friends, DBLP coauthor, DNC emails |
| **Knowledge** | 3 | LP | FB15k-237, WordNet18-RR, WordNet (Netzschleuder) |
| **Infrastructure** | 6 | LP | power grid, EuroRoad, US roads, EU airlines, London transport, urban streets |
| **Finance** | 4 | NC · LP | Elliptic Bitcoin, product space, board of directors, US patents |
| **Computing** | 4 | LP | Internet AS, Internet topology, AS-Skitter, route views |
| **Vision** | 5 | GC · NC | MNIST/CIFAR-10 superpixels, ModelNet10/40, ShapeNet |
| **Physics** | 3 | GR · LP | QM9, ZINC, Ising lattice |
| **Security** | 3 | GC · LP | MalNet-Tiny, 9/11 terrorists, train terrorists |

## Loading individual datasets

Each category exposes thin loader functions returning a PyG dataset:

```python
from graphnetz.datasets.social import cora, roman_empire
from graphnetz.datasets.biology import mutag, peptides_func
from graphnetz.datasets.computing import internet_as

ds_cora = cora("data/cora")                       # node_cls + link_pred
ds_rom  = roman_empire("data/roman_empire")       # heterophilic node_cls
ds_mut  = mutag("data/mutag")                     # graph_cls
ds_pep  = peptides_func("data/peptides_func")     # LRGB graph_cls
ds_inet = internet_as("data/internet_as")         # link_pred
```

The first call downloads + processes into the directory you pass; subsequent
calls hit the on-disk cache.

## Arbitrary Netzschleuder networks

The `Netz` loader fetches any network from the [Netzschleuder
catalogue](https://networks.skewed.de/) on demand and converts it to the PyG
format used by the rest of the library:

```python
from graphnetz import Netz

ds = Netz(root="data", dataset_name="urban_streets", network_name="brasilia")
data = ds[0]   # PyG Data object

# Multiplex / transit / airline networks need parallel-edge support:
ds_air = Netz(
    root="data",
    dataset_name="eu_airlines",
    network_name="eu_airlines",
    multigraph=True,
)
```

## Choosing a dataset

| If you want… | Try |
|---|---|
| A small node-classification sanity check | `social.cora`, `social.citeseer` |
| Heterophilic node classification | `social.roman_empire`, `social.minesweeper` |
| Long-range graph classification | `biology.peptides_func` (LRGB) |
| Molecular regression | `physics.zinc`, `physics.qm9` |
| Knowledge-graph link prediction | `knowledge.fb15k_237`, `knowledge.wordnet18rr` |
| A real-world infrastructure network | `infrastructure.power_grid`, `infrastructure.euroroad` |
| Synthetic, deterministic, fast | `combinatorial.random_coloring`, `combinatorial.random_tsp` |

## Adding a new loader

See [Contributing → Adding a dataset loader](contributing.md#adding-a-dataset-loader).
The short version:

1. Write a thin loader function under the right category module.
2. Register it in `LOADER_REGISTRY` for each task kind it supports.
3. Optionally add a `Task(...)` to `BENCHMARK_TASKS` for the curated run.
4. Add a one-line smoke test in `tests/test_smoke.py`.

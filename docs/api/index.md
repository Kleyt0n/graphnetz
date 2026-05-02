# API reference

Auto-generated reference for the public surface of GraphNetz. The reference
is organised by module so deep links stay stable as the package grows; the
top-level convenience imports below cover everything you'll typically need.

```{toctree}
:maxdepth: 1

datasets
models
benchmark
training
plotting
```

## Top-level convenience imports

Importable directly from `graphnetz`:

```{eval-rst}
.. currentmodule:: graphnetz

.. rubric:: Models

.. autosummary::
   :nosignatures:

   GCN
   GAT
   GIN
   GraphSAGE
   GraphTransformer
   DGI

.. rubric:: Datasets

.. autosummary::
   :nosignatures:

   Netz

.. rubric:: Benchmark

.. autosummary::
   :nosignatures:

   run_benchmark
   register_model
   BenchmarkReport
   ModelSpec
   plot_benchmark

.. rubric:: Training utilities

.. autosummary::
   :nosignatures:

   train_node_classification
   train_graph_classification
   train_graph_regression
   train_link_prediction
   train_dgi
   train_node_degree_regression

.. rubric:: Plotting

.. autosummary::
   :nosignatures:

   plot_history
   set_plot_style
```

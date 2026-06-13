"""Statistically robust benchmarks across a category for one or many models.

The dispatcher trains every compatible (model, task) pair across multiple
seeds and returns a :class:`BenchmarkReport` that exposes mean ± 95 % t-CI,
paired t-tests with Holm-Bonferroni correction, publication-ready LaTeX
tables, and plots.

Custom models are plugged in via the same three paths as before:

1. **Decorator / registry**::

       from graphnetz import register_model


       @register_model(task_type="node_cls")
       class MyGNN(torch.nn.Module):
           def __init__(self, in_channels, hidden_channels, out_channels): ...

2. **Class attribute**::

       class MyGNN(torch.nn.Module):
           task_types = {"node_cls"}

3. **Inline tuple** ``(cls, tasks)`` or ``(cls, tasks, factory)`` in the
   ``models`` mapping::

       run_benchmark("social", {"MyGNN": (MyGNN, "node_cls")})

The default factory calls ``cls(in_channels, hidden_channels, out_channels)``;
DGI-task models receive ``(in_channels, hidden_channels)`` (the third argument
is dropped).
"""

from graphnetz.benchmark._report import BenchmarkReport
from graphnetz.benchmark._runner import (
    _run_task as _run_task,  # documented extension point (docs/contributing.md)
)
from graphnetz.benchmark._runner import (
    plot_benchmark,
    run_benchmark,
)
from graphnetz.benchmark._specs import (
    TASK_TYPES,
    ModelSpec,
    Task,
    register_model,
)
from graphnetz.benchmark._specs import (
    _multi_task_factory as _multi_task_factory,  # documented (docs/models.md)
)
from graphnetz.benchmark._tasks import (
    BENCHMARK_TASKS,
    iter_benchmark_tasks,
    register_task,
    task_from_dataset,
    unregister_task,
)
from graphnetz.plotting import save_figure

__all__ = [
    "BENCHMARK_TASKS",
    "TASK_TYPES",
    "BenchmarkReport",
    "ModelSpec",
    "Task",
    "iter_benchmark_tasks",
    "plot_benchmark",
    "register_model",
    "register_task",
    "run_benchmark",
    "save_figure",
    "task_from_dataset",
    "unregister_task",
]

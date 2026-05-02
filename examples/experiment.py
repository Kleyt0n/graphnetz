"""Quick multi-seed benchmark across a few categories.

Trains a handful of architectures on each requested category, caches the
per-category :class:`BenchmarkReport`, and writes a forest plot + Demšar CD
diagram + LaTeX summary/pairwise tables.

Usage::

    python examples/experiment.py                       # defaults
    python examples/experiment.py --categories social biology
    python examples/experiment.py --seeds 5 --epochs 50 --force

Outputs land in ``examples/_artifacts/`` by default::

    _artifacts/
        cache/   <category>.pkl              -- pickled BenchmarkReport
        figures/ <category>_cd.pdf           -- per-category CD diagram (also .png)
                 <category>_forest.pdf
                 cd_overall.pdf              -- CD across every (cat, task)
        tables/  <category>_summary.tex
                 <category>_pairwise_t.tex          -- Holm-corrected paired-t
                 <category>_pairwise_wilcoxon.tex   -- Holm-corrected Wilcoxon
"""

from __future__ import annotations

import argparse
import pickle
from collections.abc import Sequence
from pathlib import Path

from graphnetz import (
    GAT,
    GCN,
    BenchmarkReport,
    GraphSAGE,
    GraphTransformer,
    run_benchmark,
    save_figure,
    set_plot_style,
)
from graphnetz.benchmark import iter_benchmark_tasks

DEFAULT_CATEGORIES = ("social", "biology", "infrastructure")
DEFAULT_MODELS = {
    "GCN": GCN,
    "GAT": GAT,
    "GraphSAGE": GraphSAGE,
    "GraphTransformer": GraphTransformer,
}


def _root(path: str | None) -> Path:
    return Path(path) if path else Path(__file__).resolve().parent / "_artifacts"


def _run_one(
    category: str,
    seeds: tuple[int, ...],
    epochs: int | None,
    cache_dir: Path,
    data_root: str,
    force: bool,
) -> BenchmarkReport:
    cache_path = cache_dir / f"{category}.pkl"
    if cache_path.exists() and not force:
        print(f"[{category}] cache hit -> {cache_path}")
        with cache_path.open("rb") as f:
            return pickle.load(f)
    print(f"[{category}] training {len(DEFAULT_MODELS)} models x {len(seeds)} seeds")
    report = run_benchmark(
        category,
        DEFAULT_MODELS,
        root=data_root,
        seeds=seeds,
        epochs=epochs,
        verbose=False,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(report, f)
    return report


def _emit(report: BenchmarkReport, name: str, fig_dir: Path, tab_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    fig_cd, _ = report.plot_critical_difference(title=name)
    save_figure(fig_cd, fig_dir / f"{name}_cd")
    fig_forest, _ = report.plot_forest()
    save_figure(fig_forest, fig_dir / f"{name}_forest")
    report.to_latex(tab_dir / f"{name}_summary.tex", caption=f"{name}: mean test metric ± 95% t-CI.")
    # Pairwise tables under both tests: paired-t (parametric) and Wilcoxon
    # signed-rank (non-parametric, recommended at small S; Benavoli et al. 2016).
    report.pairwise_to_latex(
        tab_dir / f"{name}_pairwise_t.tex",
        caption=f"{name}: Holm-corrected paired-$t$ tests.",
    )
    report.pairwise_to_latex(
        tab_dir / f"{name}_pairwise_wilcoxon.tex",
        method="wilcoxon",
        caption=f"{name}: Holm-corrected Wilcoxon signed-rank tests.",
    )


def _merge_overall(reports: dict[str, BenchmarkReport]) -> BenchmarkReport:
    """Stitch per-category reports into one with task names prefixed by category."""
    histories: dict[str, dict[str, list]] = {}
    seeds: tuple[int, ...] = ()
    for cat, rep in reports.items():
        seeds = rep.seeds
        for task, per_task in rep.histories.items():
            histories[f"{cat}/{task}"] = per_task
    return BenchmarkReport(seeds=seeds, histories=histories, config={"merged_from": list(reports)})


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    parser.add_argument("--seeds", type=int, default=10, help="number of seeds (0..N-1)")
    parser.add_argument("--epochs", type=int, default=None, help="override per-task epoch budget")
    parser.add_argument("--data-root", default="data/benchmark")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true", help="ignore the cache and retrain")
    args = parser.parse_args(argv)

    seeds = tuple(range(args.seeds))
    out = _root(args.output_dir)
    cache_dir = out / "cache"
    fig_dir = out / "figures"
    tab_dir = out / "tables"

    set_plot_style()
    reports: dict[str, BenchmarkReport] = {}
    for category in args.categories:
        if not iter_benchmark_tasks(category=category):
            print(f"[{category}] skipped: no curated tasks")
            continue
        rep = _run_one(category, seeds, args.epochs, cache_dir, args.data_root, args.force)
        _emit(rep, category, fig_dir, tab_dir)
        reports[category] = rep
        print(rep.summary())

    if len(reports) >= 2:
        overall = _merge_overall(reports)
        fig, _ = overall.plot_critical_difference(title="Across all categories")
        save_figure(fig, fig_dir / "cd_overall")
        overall.to_latex(tab_dir / "overall_summary.tex", caption="Cross-category mean ± 95% t-CI.")
        print(f"[overall] CD diagram across {len(overall.histories)} (category, task) cells")

    print(f"\nArtifacts written to {out}")


if __name__ == "__main__":
    main()

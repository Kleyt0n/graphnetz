"""Statistical helpers shared by the report and the runner."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import stats

_METRIC_KEYS: tuple[str, ...] = (
    "test_acc",
    "test_auc",
    "val_acc",
    "val_auc",
    "val_mae",
)
_LOWER_IS_BETTER: frozenset[str] = frozenset({"val_mae", "train_loss"})

# --------------------------------------------------------------------------- #
# Statistical helpers
# --------------------------------------------------------------------------- #


def _ci_half_width(values: np.ndarray, ci: float = 0.95) -> float:
    """Half-width of a t-distribution confidence interval for the mean."""
    n = values.size
    if n < 2:
        return 0.0
    sem = stats.sem(values)
    return float(sem * stats.t.ppf((1 + ci) / 2, n - 1))


def _bootstrap_ci_half_width(
    values: np.ndarray,
    ci: float = 0.95,
    n_resamples: int = 10000,
    random_state: int = 0,
) -> float:
    """Half-width of a percentile-bootstrap CI for the mean.

    Robust for non-Gaussian metrics (e.g. Hits@K, MRR, AUC) where the
    Student's-t assumption is poor. Returns ``(hi - lo) / 2`` -- the
    half-width of a symmetric envelope with the same total width as the
    percentile interval, so callers reporting ``mean ± half`` recover
    the bootstrap interval's spread without inflating asymmetric tails.
    """
    arr = np.asarray(values, dtype=float).ravel()
    n = arr.size
    if n < 2:
        return 0.0
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float((hi - lo) / 2.0)


def _resolve_ci_half(
    values: np.ndarray,
    ci: float,
    method: str,
    n_resamples: int,
    random_state: int,
) -> float:
    if method == "t":
        return _ci_half_width(values, ci)
    if method == "bootstrap":
        return _bootstrap_ci_half_width(values, ci, n_resamples, random_state)
    msg = f"Unknown CI method: {method!r}; choices: 't', 'bootstrap'"
    raise ValueError(msg)


def _paired_pvalue(a: np.ndarray, b: np.ndarray, method: str) -> float:
    """p-value of a paired test between two seed-aligned metric arrays.

    ``method="t"`` is the paired Student's t-test (parametric). ``method=
    "wilcoxon"`` is the Wilcoxon signed-rank test on the paired
    differences -- recommended at small seed counts where the paired
    t-test's normality assumption is most fragile (Benavoli et al.,
    JMLR 2016).
    """
    if a.size < 2 or b.size < 2 or a.size != b.size:
        return float("nan")
    if method == "t":
        return float(stats.ttest_rel(a, b).pvalue)
    if method == "wilcoxon":
        diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
        # All-zero paired differences -> the signed-rank statistic has no
        # ranks to assign; return NaN so the row is reported as undefined
        # rather than as an artificial 1.0.
        if not np.any(diffs != 0):
            return float("nan")
        try:
            return float(stats.wilcoxon(diffs, zero_method="wilcox").pvalue)
        except ValueError:
            return float("nan")
    msg = f"Unknown pairwise method: {method!r}; choices: 't', 'wilcoxon'"
    raise ValueError(msg)


def _holm_correction(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (max-monotone).

    NaN inputs (e.g. tests that were undefined for that pair) are
    excluded from the rank table and propagated as NaN in the output;
    they are *not* counted toward the family size, so the remaining
    valid tests retain their proper power.
    """
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return p
    valid = ~np.isnan(p)
    n_valid = int(valid.sum())
    adjusted = np.full(n, np.nan, dtype=float)
    if n_valid == 0:
        return adjusted
    valid_idx = np.where(valid)[0]
    p_valid = p[valid_idx]
    order = np.argsort(p_valid)
    running = 0.0
    out_valid = np.empty(n_valid, dtype=float)
    for rank, idx in enumerate(order):
        adj = float(min(p_valid[idx] * (n_valid - rank), 1.0))
        running = max(running, adj)
        out_valid[idx] = running
    adjusted[valid_idx] = out_valid
    return adjusted


def _auto_metric_key(history: Mapping[str, Any]) -> str:
    for key in _METRIC_KEYS:
        if key in history:
            return key
    return next(iter(history))


def _final_metric(history: Mapping[str, list[float]]) -> tuple[str, float]:
    key = _auto_metric_key(history)
    return key, history[key][-1]

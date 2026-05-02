"""Tests for the paired-test surface (Student's t and Wilcoxon signed-rank)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest


def _report_with_two_models() -> BenchmarkReport:  # noqa: F821
    """Construct a synthetic 2-model, 2-task report with known seed values.

    Model A is consistently better than Model B on both tasks, so the
    pairwise tests should produce small (significant-leaning) p-values.
    """
    from graphnetz.benchmark import BenchmarkReport

    histories = {
        "task1": {
            "A": [
                {"test_acc": [0.5, 0.55, 0.6, 0.65, 0.70]},
                {"test_acc": [0.55, 0.60, 0.65, 0.70, 0.74]},
                {"test_acc": [0.52, 0.57, 0.63, 0.68, 0.72]},
                {"test_acc": [0.54, 0.59, 0.64, 0.69, 0.73]},
                {"test_acc": [0.53, 0.58, 0.62, 0.66, 0.71]},
            ],
            "B": [
                {"test_acc": [0.40, 0.45, 0.50, 0.55, 0.60]},
                {"test_acc": [0.42, 0.47, 0.52, 0.57, 0.62]},
                {"test_acc": [0.41, 0.46, 0.51, 0.56, 0.61]},
                {"test_acc": [0.43, 0.48, 0.53, 0.58, 0.63]},
                {"test_acc": [0.44, 0.49, 0.54, 0.59, 0.64]},
            ],
        },
        "task2": {
            "A": [
                {"test_acc": [0.62, 0.66, 0.69, 0.71, 0.74]},
                {"test_acc": [0.63, 0.67, 0.70, 0.72, 0.75]},
                {"test_acc": [0.64, 0.68, 0.71, 0.73, 0.76]},
                {"test_acc": [0.65, 0.69, 0.72, 0.74, 0.77]},
                {"test_acc": [0.66, 0.70, 0.73, 0.75, 0.78]},
            ],
            "B": [
                {"test_acc": [0.55, 0.59, 0.62, 0.64, 0.67]},
                {"test_acc": [0.56, 0.60, 0.63, 0.65, 0.68]},
                {"test_acc": [0.57, 0.61, 0.64, 0.66, 0.69]},
                {"test_acc": [0.58, 0.62, 0.65, 0.67, 0.70]},
                {"test_acc": [0.59, 0.63, 0.66, 0.68, 0.71]},
            ],
        },
    }
    return BenchmarkReport(seeds=(0, 1, 2, 3, 4), histories=histories)


def test_pairwise_default_is_paired_t() -> None:
    report = _report_with_two_models()
    df = report.pairwise()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"task", "model_a", "model_b", "mean_diff", "p_raw", "p_holm", "significant"}
    # Two tasks x one comparison each.
    assert len(df) == 2
    # A is consistently better than B; mean_diff should be positive on both rows.
    assert (df["mean_diff"] > 0).all()


def test_pairwise_wilcoxon_method_runs() -> None:
    report = _report_with_two_models()
    df = report.pairwise(method="wilcoxon")
    assert len(df) == 2
    # All p-values must be in [0, 1] and not NaN with 5 paired samples.
    assert df["p_raw"].between(0, 1).all()
    assert df["p_holm"].between(0, 1).all()


def test_pairwise_wilcoxon_differs_from_t_at_small_n() -> None:
    """Wilcoxon and paired-t should not produce identical p-values on the
    same paired-difference vector except by coincidence; we only assert that
    *at least one* row differs."""
    report = _report_with_two_models()
    p_t = report.pairwise(method="t")["p_raw"].to_numpy()
    p_w = report.pairwise(method="wilcoxon")["p_raw"].to_numpy()
    assert not np.allclose(p_t, p_w, atol=1e-9)


def test_pairwise_wilcoxon_handles_all_zero_diff() -> None:
    """When two models have identical per-seed metrics, paired diffs are
    all zero; Wilcoxon's signed-rank statistic is undefined and we report
    NaN rather than an artificial 1.0."""
    from graphnetz.benchmark import BenchmarkReport

    same = [{"test_acc": [0.5]} for _ in range(3)]
    histories = {"t1": {"A": same, "B": same}}
    report = BenchmarkReport(seeds=(0, 1, 2), histories=histories)
    df = report.pairwise(method="wilcoxon")
    assert len(df) == 1
    assert math.isnan(df["p_raw"].iloc[0])
    assert not df["significant"].iloc[0]


def test_pairwise_method_field_is_sticky() -> None:
    report = _report_with_two_models()
    p_default = report.pairwise()["p_raw"].to_numpy()
    report.pairwise_method = "wilcoxon"
    p_sticky = report.pairwise()["p_raw"].to_numpy()
    assert not np.allclose(p_default, p_sticky, atol=1e-9)
    # And the per-call override beats the field setting.
    p_override = report.pairwise(method="t")["p_raw"].to_numpy()
    assert np.allclose(p_override, p_default, atol=1e-12)


def test_pairwise_unknown_method_raises() -> None:
    report = _report_with_two_models()
    with pytest.raises(ValueError, match="Unknown pairwise method"):
        report.pairwise(method="permutation")


def test_pairwise_to_latex_threads_method() -> None:
    import tempfile
    from pathlib import Path

    report = _report_with_two_models()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "pairwise.tex"
        out = report.pairwise_to_latex(path, method="wilcoxon")
        assert out.exists()
        text = out.read_text()
        # Sanity: file is a non-empty booktabs table mentioning both tasks.
        assert "\\toprule" in text
        assert "task1" in text
        assert "task2" in text

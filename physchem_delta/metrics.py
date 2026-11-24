"""Evaluation metrics and statistical tests using true dataset ground truth."""

from __future__ import annotations

import numpy as np
from scipy import stats


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def compute_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size < 3 or np.std(y_pred) < 1e-12:
        return float("nan")
    rho, _ = stats.spearmanr(y_true, y_pred)
    return float(rho)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cliff_mol: np.ndarray,
) -> dict[str, float]:
    """Compute global, cliff-subset, and non-cliff-subset metrics on ground-truth labels."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    cliff_mask = np.asarray(cliff_mol, dtype=int) == 1
    non_cliff_mask = ~cliff_mask

    return {
        "rmse": compute_rmse(y_true, y_pred),
        "cliff_rmse": compute_rmse(y_true[cliff_mask], y_pred[cliff_mask]),
        "non_cliff_rmse": compute_rmse(y_true[non_cliff_mask], y_pred[non_cliff_mask]),
        "mae": compute_mae(y_true, y_pred),
        "cliff_mae": compute_mae(y_true[cliff_mask], y_pred[cliff_mask]),
        "r2": compute_r2(y_true, y_pred),
        "spearman": compute_spearman(y_true, y_pred),
        "n_test": int(y_true.size),
        "n_cliff": int(np.sum(cliff_mask)),
        "n_non_cliff": int(np.sum(non_cliff_mask)),
    }


def bootstrap_rmse_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (point_rmse, ci_low_95, ci_high_95)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = y_true.size
    sq_err = (y_true - y_pred) ** 2
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_rmses = np.sqrt(np.mean(sq_err[idx], axis=1))
    return (
        float(np.sqrt(np.mean(sq_err))),
        float(np.percentile(boot_rmses, 2.5)),
        float(np.percentile(boot_rmses, 97.5)),
    )


def paired_wilcoxon(abs_err_a: np.ndarray, abs_err_b: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value comparing per-sample absolute errors."""
    abs_err_a = np.asarray(abs_err_a, dtype=np.float64)
    abs_err_b = np.asarray(abs_err_b, dtype=np.float64)
    diff = abs_err_a - abs_err_b
    if np.all(np.abs(diff) < 1e-12):
        return 1.0
    try:
        _, pval = stats.wilcoxon(abs_err_a, abs_err_b, alternative="two-sided")
        return float(pval)
    except Exception:
        return float("nan")


def similarity_bin_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    s_max: np.ndarray,
    cliff_mol: np.ndarray,
    bins: tuple[float, ...] = (0.0, 0.5, 0.7, 0.85, 1.01),
) -> list[dict[str, float | str | int]]:
    """Evaluate error across max-training-similarity bins for OOD gate diagnostics."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    s_max = np.asarray(s_max, dtype=np.float64)
    cliff_mask = np.asarray(cliff_mol, dtype=int) == 1

    rows: list[dict[str, float | str | int]] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (s_max >= lo) & (s_max < hi)
        if np.sum(mask) == 0:
            continue
        cmask = mask & cliff_mask
        rows.append(
            {
                "bin": f"[{lo:.2f}, {min(hi, 1.0):.2f})",
                "n": int(np.sum(mask)),
                "n_cliff": int(np.sum(cmask)),
                "rmse": compute_rmse(y_true[mask], y_pred[mask]),
                "cliff_rmse": compute_rmse(y_true[cmask], y_pred[cmask]) if np.sum(cmask) > 0 else float("nan"),
                "mae": compute_mae(y_true[mask], y_pred[mask]),
            }
        )
    return rows

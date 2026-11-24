"""Wave-based 30-target MoleculeACE benchmark and negative-result audit for G2_Prev."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from physchem_delta.data import load_target_dataset
from physchem_delta.eval_plan_a import (
    LFER_CHANNELS,
    SEEDS,
    _fit_2param_gate,
    _fit_phys_gate_g2,
    _fit_rbf_svr,
    _lfer_atom_matrix,
    _predict_delta_backbone,
    _train_delta_backbone,
)
from physchem_delta.featurize import extract_or_load_target_features
from physchem_delta.metrics import compute_rmse, evaluate_predictions, paired_wilcoxon
from physchem_delta.models import get_device

WAVE_1_TARGETS: tuple[str, ...] = (
    "CHEMBL204_Ki",
    "CHEMBL2147_Ki",
    "CHEMBL237_Ki",
    "CHEMBL1862_Ki",
    "CHEMBL1871_Ki",
    "CHEMBL218_EC50",
    "CHEMBL228_Ki",
    "CHEMBL235_EC50",
    "CHEMBL244_Ki",
    "CHEMBL2835_Ki",
)

WAVE_2_TARGETS: tuple[str, ...] = (
    "CHEMBL2034_Ki",
    "CHEMBL2047_EC50",
    "CHEMBL214_Ki",
    "CHEMBL219_Ki",
    "CHEMBL231_Ki",
    "CHEMBL233_Ki",
    "CHEMBL234_Ki",
    "CHEMBL236_Ki",
    "CHEMBL237_EC50",
    "CHEMBL238_Ki",
)

WAVE_3_TARGETS: tuple[str, ...] = (
    "CHEMBL239_EC50",
    "CHEMBL262_Ki",
    "CHEMBL264_Ki",
    "CHEMBL287_Ki",
    "CHEMBL2971_Ki",
    "CHEMBL3979_EC50",
    "CHEMBL4005_Ki",
    "CHEMBL4203_Ki",
    "CHEMBL4616_EC50",
    "CHEMBL4792_Ki",
)

ALL_30_TARGETS: tuple[str, ...] = WAVE_1_TARGETS + WAVE_2_TARGETS + WAVE_3_TARGETS

MODELS_ORDER: tuple[str, ...] = (
    "B1c_RBF_SVM",
    "A2_2D_Delta",
    "L2_eSpan_LFER_Delta",
    "G0_RBFSVM_Gate_A2",
    "G1_RBFSVM_Gate_L2",
    "G2_RBFSVM_PhysGate_L2",
)


def _out_root() -> Path:
    d = ROOT_DIR / "refine-logs" / "benchmark30_results"
    (d / "per_target").mkdir(parents=True, exist_ok=True)
    return d


def _fig_dir() -> Path:
    d = ROOT_DIR / "refine-logs" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_stat(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr, ddof=0))


def _evaluate_single_target(target: str, n_workers: int, force: bool) -> dict[str, Any]:
    out_dir = _out_root() / "per_target"
    json_path = out_dir / f"{target}.json"
    npz_path = out_dir / f"{target}_preds.npz"

    if json_path.exists() and not force:
        cached = json.loads(json_path.read_text(encoding="utf-8"))
        if all(m in cached.get("metrics", {}) for m in MODELS_ORDER):
            print(f"[CACHE] {target}: loaded existing checkpoint from {json_path.name}", flush=True)
            return cached

    t0 = time.time()
    device = get_device()
    ds = load_target_dataset(target)
    feats = extract_or_load_target_features(
        target, ds.smiles, prefer_aimnet=True, num_confs=5, n_workers=n_workers
    )
    lfer_atoms = _lfer_atom_matrix(feats)

    y_te = ds.y[ds.test_idx]
    c_te = ds.cliff_mol[ds.test_idx]
    cmask = c_te == 1

    assay_type = "EC50" if target.endswith("_EC50") else "Ki"
    aimnet_fb = int(feats.meta.get("aimnet_fallback_count", 0))
    embed_fb = int(feats.meta.get("embed_fallback_count", 0))
    chemistry_profile = {
        "target": target,
        "assay_type": assay_type,
        "n_total": int(len(ds.smiles)),
        "n_train": int(ds.train_idx.size),
        "n_test": int(ds.test_idx.size),
        "n_test_cliff": int(np.sum(cmask)),
        "cliff_frac": float(np.mean(cmask)),
        "mean_mw": float(np.mean(feats.desc2d[:, 0])),
        "mean_logp": float(np.mean(feats.desc2d[:, 1])),
        "mean_tpsa": float(np.mean(feats.desc2d[:, 2])),
        "mean_rot_bonds": float(np.mean(feats.desc2d[:, 5])),
        "mean_arom_rings": float(np.mean(feats.desc2d[:, 6])),
        "mean_fsp3": float(np.mean(feats.desc2d[:, 7])),
        "mean_e_span": float(np.mean(feats.feat_3d[:, 1])),
        "electronic_backend": str(feats.electronic_backend),
        "embed_fallback_count": embed_fb,
        "aimnet_fallback_count": aimnet_fb,
        "aimnet_fallback_rate": float(aimnet_fb / max(len(ds.smiles), 1)),
    }

    y_test_svm = _fit_rbf_svr(feats.ecfp_chiral, ds.y, ds.train_idx, ds.test_idx)

    seed_metrics: dict[str, list[dict[str, float]]] = {m: [] for m in MODELS_ORDER}
    seed_preds: dict[str, list[np.ndarray]] = {m: [] for m in MODELS_ORDER}
    gate_stats: list[dict[str, float]] = []
    channel_deltas: dict[str, list[float]] = {ch: [] for ch in LFER_CHANNELS}

    for seed in SEEDS:
        a2_out = _train_delta_backbone(feats, lfer_atoms, ds.y, ds.train_idx, ds.test_idx, seed, mode="2d")
        l2_out = _train_delta_backbone(feats, lfer_atoms, ds.y, ds.train_idx, ds.test_idx, seed, mode="lfer_espan")

        y_val_svm = _fit_rbf_svr(feats.ecfp_chiral, ds.y, a2_out["inner_tr_idx"], a2_out["inner_va_idx"])
        y_val_true = ds.y[a2_out["inner_va_idx"]]

        g0_out = _fit_2param_gate(
            y_val_true, a2_out["y_pred_val"], y_val_svm, a2_out["s_max_val"],
            a2_out["y_pred_test"], y_test_svm, a2_out["s_max_test"],
        )
        g1_out = _fit_2param_gate(
            y_val_true, l2_out["y_pred_val"], y_val_svm, l2_out["s_max_val"],
            l2_out["y_pred_test"], y_test_svm, l2_out["s_max_test"],
        )
        g2_out = _fit_phys_gate_g2(
            feats, lfer_atoms, l2_out["env_counters"],
            l2_out["inner_tr_idx"], l2_out["inner_va_idx"],
            ds.train_idx, ds.test_idx,
            y_val_true, l2_out["y_pred_val"], y_val_svm, l2_out["s_max_val"],
            l2_out["y_pred_test"], y_test_svm, l2_out["s_max_test"],
            g1_ws=g1_out["ws"], g1_b=g1_out["b"],
        )

        gate_stats.append(
            {
                "ws": float(g2_out["ws"]),
                "b": float(g2_out["b"]),
                "we": float(g2_out["we"]),
                "wq": float(g2_out["wq"]),
                "alpha_mean": float(g2_out["alpha_mean"]),
            }
        )

        preds_map = {
            "B1c_RBF_SVM": y_test_svm,
            "A2_2D_Delta": a2_out["y_pred_test"],
            "L2_eSpan_LFER_Delta": l2_out["y_pred_test"],
            "G0_RBFSVM_Gate_A2": g0_out["y_pred_test"],
            "G1_RBFSVM_Gate_L2": g1_out["y_pred_test"],
            "G2_RBFSVM_PhysGate_L2": g2_out["y_pred_test"],
        }
        for m_name, yp in preds_map.items():
            seed_metrics[m_name].append(evaluate_predictions(y_te, yp, c_te))
            seed_preds[m_name].append(yp)

        base_cliff = seed_metrics["L2_eSpan_LFER_Delta"][-1]["cliff_rmse"]
        for ch_name, cols in LFER_CHANNELS.items():
            yp_zero, _ = _predict_delta_backbone(
                l2_out["model"], feats, l2_out["bundle_2d"], lfer_atoms, ds.test_idx, ds.train_idx,
                ds.y, l2_out["y_std"], l2_out["diff_scale"], l2_out["env_counters"],
                mode="lfer_espan", device=device, zero_cols=cols,
            )
            zero_cliff = compute_rmse(y_te[cmask], yp_zero[cmask])
            channel_deltas[ch_name].append(float(zero_cliff - base_cliff))

    metrics_summary: dict[str, dict[str, float]] = {}
    mean_preds: dict[str, np.ndarray] = {}
    npz_payload: dict[str, np.ndarray] = {"y_te": y_te, "c_te": c_te}

    for m_name in MODELS_ORDER:
        rmse_m, rmse_s = _safe_stat([r["rmse"] for r in seed_metrics[m_name]])
        cliff_m, cliff_s = _safe_stat([r["cliff_rmse"] for r in seed_metrics[m_name]])
        ncliff_m, ncliff_s = _safe_stat([r["non_cliff_rmse"] for r in seed_metrics[m_name]])
        sp_m, sp_s = _safe_stat([r["spearman"] for r in seed_metrics[m_name]])
        mae_m, _ = _safe_stat([r["mae"] for r in seed_metrics[m_name]])
        cmae_m, _ = _safe_stat([r["cliff_mae"] for r in seed_metrics[m_name]])
        metrics_summary[m_name] = {
            "rmse_mean": rmse_m,
            "rmse_std": rmse_s,
            "cliff_rmse_mean": cliff_m,
            "cliff_rmse_std": cliff_s,
            "non_cliff_rmse_mean": ncliff_m,
            "non_cliff_rmse_std": ncliff_s,
            "spearman_mean": sp_m,
            "spearman_std": sp_s,
            "mae_mean": mae_m,
            "cliff_mae_mean": cmae_m,
        }
        stacked = np.stack(seed_preds[m_name], axis=0)
        mean_preds[m_name] = np.mean(stacked, axis=0)
        npz_payload[f"{m_name}_seeds"] = stacked
        npz_payload[f"{m_name}_mean"] = mean_preds[m_name]

    def _wilcoxon_pair(m_a: str, m_b: str) -> dict[str, float]:
        ea = np.abs(y_te - mean_preds[m_a])
        eb = np.abs(y_te - mean_preds[m_b])
        return {
            "p_cliff": paired_wilcoxon(ea[cmask], eb[cmask]),
            "p_all": paired_wilcoxon(ea, eb),
            "cliff_rmse_diff": metrics_summary[m_a]["cliff_rmse_mean"] - metrics_summary[m_b]["cliff_rmse_mean"],
            "rmse_diff": metrics_summary[m_a]["rmse_mean"] - metrics_summary[m_b]["rmse_mean"],
        }

    wilcoxon_tests = {
        "G2_vs_SVM": _wilcoxon_pair("G2_RBFSVM_PhysGate_L2", "B1c_RBF_SVM"),
        "G2_vs_G0": _wilcoxon_pair("G2_RBFSVM_PhysGate_L2", "G0_RBFSVM_Gate_A2"),
        "G1_vs_G0": _wilcoxon_pair("G1_RBFSVM_Gate_L2", "G0_RBFSVM_Gate_A2"),
        "G2_vs_G1": _wilcoxon_pair("G2_RBFSVM_PhysGate_L2", "G1_RBFSVM_Gate_L2"),
    }

    lfer_attr = {
        ch: {"delta_cliff_rmse_mean": float(np.mean(vals)), "delta_cliff_rmse_std": float(np.std(vals))}
        for ch, vals in channel_deltas.items()
    }

    svm_c = metrics_summary["B1c_RBF_SVM"]["cliff_rmse_mean"]
    a2_c = metrics_summary["A2_2D_Delta"]["cliff_rmse_mean"]
    l2_c = metrics_summary["L2_eSpan_LFER_Delta"]["cliff_rmse_mean"]
    g0_c = metrics_summary["G0_RBFSVM_Gate_A2"]["cliff_rmse_mean"]
    g1_c = metrics_summary["G1_RBFSVM_Gate_L2"]["cliff_rmse_mean"]
    g2_c = metrics_summary["G2_RBFSVM_PhysGate_L2"]["cliff_rmse_mean"]
    g2_c_std = metrics_summary["G2_RBFSVM_PhysGate_L2"]["cliff_rmse_std"]
    g0_r = metrics_summary["G0_RBFSVM_Gate_A2"]["rmse_mean"]
    g2_r = metrics_summary["G2_RBFSVM_PhysGate_L2"]["rmse_mean"]

    flags = {
        "NEG_VS_SVM": bool(g2_c > svm_c),
        "NEG_VS_G0": bool(g2_c > g0_c),
        "NEG_OVERALL_VS_G0": bool(g2_r > g0_r),
        "HIGH_VAR": bool(g2_c_std > 0.025),
    }
    harmful_channels = [ch for ch, st in lfer_attr.items() if st["delta_cliff_rmse_mean"] < -0.0005]
    diagnosis = {
        "L2_WORSE_THAN_A2": bool(l2_c > a2_c),
        "PHYS_GATE_WORSE_THAN_G1": bool(g2_c > g1_c),
        "HARMFUL_CHANNELS": harmful_channels,
        "delta_l2_minus_a2": float(l2_c - a2_c),
        "delta_g2_minus_g1": float(g2_c - g1_c),
        "delta_g2_minus_g0": float(g2_c - g0_c),
        "delta_g2_minus_svm": float(g2_c - svm_c),
    }

    gate_mean = {
        "g2_ws": float(np.mean([g["ws"] for g in gate_stats])),
        "g2_b": float(np.mean([g["b"] for g in gate_stats])),
        "g2_we": float(np.mean([g["we"] for g in gate_stats])),
        "g2_wq": float(np.mean([g["wq"] for g in gate_stats])),
        "g2_alpha_mean": float(np.mean([g["alpha_mean"] for g in gate_stats])),
    }

    elapsed = round(time.time() - t0, 2)
    result = {
        "target": target,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": elapsed,
        "chemistry": chemistry_profile,
        "metrics": metrics_summary,
        "gate_mean": gate_mean,
        "wilcoxon": wilcoxon_tests,
        "lfer_attribution": lfer_attr,
        "flags": flags,
        "diagnosis": diagnosis,
    }

    np.savez_compressed(npz_path, **npz_payload)
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    active_flags = [k for k, v in flags.items() if v]
    alert_str = f" [ALERT: {', '.join(active_flags)}]" if active_flags else " [OK]"
    print(
        f"[DONE {elapsed:.1f}s] {target} (n={chemistry_profile['n_total']}, cliff={chemistry_profile['n_test_cliff']}, "
        f"rotb={chemistry_profile['mean_rot_bonds']:.1f}, espan={chemistry_profile['mean_e_span']:.1f}): "
        f"SVM={svm_c:.4f} | A2={a2_c:.4f} | L2={l2_c:.4f} | G0={g0_c:.4f} | G1={g1_c:.4f} | "
        f"G2={g2_c:.4f}±{g2_c_std:.4f}{alert_str}",
        flush=True,
    )
    if active_flags:
        causes = []
        if diagnosis["L2_WORSE_THAN_A2"]:
            causes.append(f"L2_WORSE_THAN_A2 ({diagnosis['delta_l2_minus_a2']:+.4f})")
        if diagnosis["PHYS_GATE_WORSE_THAN_G1"]:
            causes.append(f"PHYS_GATE_WORSE_THAN_G1 ({diagnosis['delta_g2_minus_g1']:+.4f})")
        if harmful_channels:
            causes.append(f"HARMFUL_CHANNELS={harmful_channels}")
        print(f"  -> Auto-diagnosis: {', '.join(causes) if causes else 'marginal variance/gate difference'}", flush=True)

    return result


def _cross_target_wilcoxon(df_wide: pd.DataFrame, col_a: str, col_b: str) -> dict[str, Any]:
    diff = (df_wide[col_a] - df_wide[col_b]).to_numpy(dtype=np.float64)
    wins = int(np.sum(diff < -1e-9))
    ties = int(np.sum(np.abs(diff) <= 1e-9))
    losses = int(np.sum(diff > 1e-9))
    if diff.size >= 2 and not np.all(np.abs(diff) < 1e-12):
        try:
            _, pval = stats.wilcoxon(df_wide[col_a], df_wide[col_b], alternative="two-sided")
            p = float(pval)
        except Exception:
            p = float("nan")
    else:
        p = 1.0
    return {
        "n": int(diff.size),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "mean_diff": float(np.mean(diff)),
        "median_diff": float(np.median(diff)),
        "p_value": p,
    }


def _plot_audit_figure(
    df_wide: pd.DataFrame,
    results: list[dict[str, Any]],
    fig_path: Path,
    title_prefix: str,
) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17.5, 5.4))

    # Panel A: Sorted per-target Delta Cliff RMSE
    df_sorted = df_wide.sort_values("delta_g2_g0_cliff", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(df_sorted))
    h = 0.38
    ax1.barh(y_pos - h / 2, df_sorted["delta_g2_svm_cliff"], height=h, color="#0072B2", alpha=0.88, label="G2 - B1c_SVM (<0 win)")
    ax1.barh(y_pos + h / 2, df_sorted["delta_g2_g0_cliff"], height=h, color="#D55E00", alpha=0.88, label="G2 - G0 (<0 win)")
    ax1.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_sorted["target"], fontsize=8)
    ax1.set_xlabel("Delta Cliff RMSE (< 0 = G2 better)")
    ax1.set_title(f"{title_prefix}: Per-Target Cliff Deltas", fontweight="bold")
    ax1.legend(frameon=False, fontsize=8, loc="lower right")
    ax1.grid(axis="x", linestyle="--", alpha=0.3)

    # Panel B: Target x 6-channel LFER attribution heatmap
    ch_names = list(LFER_CHANNELS.keys())
    target_order = df_sorted["target"].tolist()
    res_map = {r["target"]: r for r in results}
    mat = np.zeros((len(target_order), len(ch_names)), dtype=np.float64)
    for r_i, tgt in enumerate(target_order):
        for c_i, ch in enumerate(ch_names):
            mat[r_i, c_i] = float(res_map[tgt]["lfer_attribution"][ch]["delta_cliff_rmse_mean"])
    vmax = max(0.004, float(np.max(np.abs(mat))))
    im = ax2.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax2.set_xticks(np.arange(len(ch_names)))
    ax2.set_xticklabels([c.replace("_", "\n") for c in ch_names], fontsize=7)
    ax2.set_yticks(np.arange(len(target_order)))
    ax2.set_yticklabels(target_order, fontsize=8)
    for r_i in range(len(target_order)):
        for c_i in range(len(ch_names)):
            ax2.text(c_i, r_i, f"{mat[r_i, c_i]:+.3f}", ha="center", va="center", color="black", fontsize=6.5)
    ax2.set_title("L2 Zeroing Attribution (>0 Useful, <0 Noise)", fontweight="bold")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.03)

    # Panel C: Delta Cliff (G2 - G0) vs mean_e_span colored by assay type
    for assay, col, marker in [("Ki", "#009E73", "o"), ("EC50", "#CC79A7", "^")]:
        sub = df_wide[df_wide["assay_type"] == assay]
        if len(sub) == 0:
            continue
        ax3.scatter(sub["mean_e_span"], sub["delta_g2_g0_cliff"], color=col, marker=marker, s=65, edgecolor="black", linewidth=0.6, label=assay)
        for _, row in sub.iterrows():
            ax3.annotate(row["target"].replace("CHEMBL", ""), (row["mean_e_span"], row["delta_g2_g0_cliff"]), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax3.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax3.set_xlabel("Mean Conformational e_span (kcal/mol)")
    ax3.set_ylabel("Delta Cliff RMSE: G2 - G0 (< 0 = PhysChem Gain)")
    ax3.set_title("PhysChem Gain vs Conformational Span", fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.3)
    ax3.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def _generate_reports(results: list[dict[str, Any]], wave_arg: str, total_elapsed: float) -> tuple[Path, Path, Path]:
    out_dir = _out_root()
    fig_dir = _fig_dir()

    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []

    for r in results:
        tgt = r["target"]
        ch = r["chemistry"]
        gm = r["gate_mean"]
        m = r["metrics"]
        diag = r["diagnosis"]
        flg = r["flags"]

        for model_name in MODELS_ORDER:
            mm = m[model_name]
            long_rows.append(
                {
                    "target": tgt,
                    "assay_type": ch["assay_type"],
                    "model": model_name,
                    "rmse_mean": mm["rmse_mean"],
                    "rmse_std": mm["rmse_std"],
                    "cliff_rmse_mean": mm["cliff_rmse_mean"],
                    "cliff_rmse_std": mm["cliff_rmse_std"],
                    "non_cliff_rmse_mean": mm["non_cliff_rmse_mean"],
                    "spearman_mean": mm["spearman_mean"],
                    "n_total": ch["n_total"],
                    "n_test_cliff": ch["n_test_cliff"],
                }
            )

        wide_rows.append(
            {
                "target": tgt,
                "assay_type": ch["assay_type"],
                "n_total": ch["n_total"],
                "n_train": ch["n_train"],
                "n_test": ch["n_test"],
                "n_test_cliff": ch["n_test_cliff"],
                "cliff_frac": ch["cliff_frac"],
                "mean_mw": ch["mean_mw"],
                "mean_logp": ch["mean_logp"],
                "mean_tpsa": ch["mean_tpsa"],
                "mean_rot_bonds": ch["mean_rot_bonds"],
                "mean_e_span": ch["mean_e_span"],
                "aimnet_fallback_rate": ch["aimnet_fallback_rate"],
                "svm_cliff": m["B1c_RBF_SVM"]["cliff_rmse_mean"],
                "a2_cliff": m["A2_2D_Delta"]["cliff_rmse_mean"],
                "l2_cliff": m["L2_eSpan_LFER_Delta"]["cliff_rmse_mean"],
                "g0_cliff": m["G0_RBFSVM_Gate_A2"]["cliff_rmse_mean"],
                "g1_cliff": m["G1_RBFSVM_Gate_L2"]["cliff_rmse_mean"],
                "g2_cliff": m["G2_RBFSVM_PhysGate_L2"]["cliff_rmse_mean"],
                "g2_cliff_std": m["G2_RBFSVM_PhysGate_L2"]["cliff_rmse_std"],
                "svm_rmse": m["B1c_RBF_SVM"]["rmse_mean"],
                "g0_rmse": m["G0_RBFSVM_Gate_A2"]["rmse_mean"],
                "g1_rmse": m["G1_RBFSVM_Gate_L2"]["rmse_mean"],
                "g2_rmse": m["G2_RBFSVM_PhysGate_L2"]["rmse_mean"],
                "delta_g2_svm_cliff": diag["delta_g2_minus_svm"],
                "delta_g2_g0_cliff": diag["delta_g2_minus_g0"],
                "delta_g2_g1_cliff": diag["delta_g2_minus_g1"],
                "delta_l2_a2_cliff": diag["delta_l2_minus_a2"],
                "g2_we": gm["g2_we"],
                "g2_wq": gm["g2_wq"],
                "g2_alpha_mean": gm["g2_alpha_mean"],
                "NEG_VS_SVM": flg["NEG_VS_SVM"],
                "NEG_VS_G0": flg["NEG_VS_G0"],
                "NEG_OVERALL_VS_G0": flg["NEG_OVERALL_VS_G0"],
                "HIGH_VAR": flg["HIGH_VAR"],
                "L2_WORSE_THAN_A2": diag["L2_WORSE_THAN_A2"],
                "PHYS_GATE_WORSE_THAN_G1": diag["PHYS_GATE_WORSE_THAN_G1"],
            }
        )

    df_long = pd.DataFrame(long_rows)
    df_wide = pd.DataFrame(wide_rows)

    if wave_arg in ("1", "2", "3"):
        csv_path = out_dir / f"wave{wave_arg}_summary.csv"
    else:
        csv_path = out_dir / "all_completed_summary.csv"
    df_long.to_csv(csv_path, index=False)

    # Also keep all_completed_summary.csv updated from all per-target JSON checkpoints present on disk
    all_jsons = sorted((out_dir / "per_target").glob("*.json"))
    all_long_rows = []
    for jp in all_jsons:
        jr = json.loads(jp.read_text(encoding="utf-8"))
        for model_name in MODELS_ORDER:
            if model_name in jr.get("metrics", {}):
                mm = jr["metrics"][model_name]
                all_long_rows.append({"target": jr["target"], "model": model_name, **mm})
    if all_long_rows:
        pd.DataFrame(all_long_rows).to_csv(out_dir / "all_completed_summary.csv", index=False)

    if wave_arg == "1":
        fig_path = fig_dir / "wave1_10target_audit.png"
        md_path = ROOT_DIR / "refine-logs" / "WAVE1_10TARGET_AUDIT.md"
        title_prefix = "Wave 1 (10 Targets)"
    elif wave_arg in ("2", "3"):
        fig_path = fig_dir / f"wave{wave_arg}_audit.png"
        md_path = ROOT_DIR / "refine-logs" / f"WAVE{wave_arg}_AUDIT.md"
        title_prefix = f"Wave {wave_arg} ({len(results)} Targets)"
    else:
        fig_path = fig_dir / "benchmark30_audit.png"
        md_path = ROOT_DIR / "refine-logs" / "BENCHMARK30_REPORT.md"
        title_prefix = f"MoleculeACE ({len(results)} Targets)"

    _plot_audit_figure(df_wide, results, fig_path, title_prefix)

    cross_tests = {
        "Cliff: G2 vs B1c_SVM": _cross_target_wilcoxon(df_wide, "g2_cliff", "svm_cliff"),
        "Cliff: G2 vs G0": _cross_target_wilcoxon(df_wide, "g2_cliff", "g0_cliff"),
        "Cliff: G1 vs G0": _cross_target_wilcoxon(df_wide, "g1_cliff", "g0_cliff"),
        "Cliff: G2 vs G1": _cross_target_wilcoxon(df_wide, "g2_cliff", "g1_cliff"),
        "Overall: G2 vs B1c_SVM": _cross_target_wilcoxon(df_wide, "g2_rmse", "svm_rmse"),
        "Overall: G2 vs G0": _cross_target_wilcoxon(df_wide, "g2_rmse", "g0_rmse"),
        "Overall: G1 vs G0": _cross_target_wilcoxon(df_wide, "g1_rmse", "g0_rmse"),
        "Overall: G2 vs G1": _cross_target_wilcoxon(df_wide, "g2_rmse", "g1_rmse"),
    }

    corr_features = [
        "mean_rot_bonds",
        "mean_mw",
        "mean_tpsa",
        "mean_e_span",
        "n_train",
        "cliff_frac",
        "aimnet_fallback_rate",
        "g2_we",
        "g2_wq",
        "g2_alpha_mean",
    ]
    corr_rows = []
    for cf in corr_features:
        xv = df_wide[cf].to_numpy(dtype=np.float64)
        for label, ycol in [("Delta_Cliff(G2-G0)", "delta_g2_g0_cliff"), ("Delta_Cliff(G2-SVM)", "delta_g2_svm_cliff")]:
            yv = df_wide[ycol].to_numpy(dtype=np.float64)
            if xv.size >= 3 and np.std(xv) > 1e-12 and np.std(yv) > 1e-12:
                rho, p = stats.spearmanr(xv, yv)
                corr_rows.append({"feature": cf, "outcome": label, "rho": float(rho), "p": float(p)})
            else:
                corr_rows.append({"feature": cf, "outcome": label, "rho": float("nan"), "p": float("nan")})

    neg_targets = [
        r for r in results
        if any(r["flags"].values())
    ]

    lines: list[str] = [
        f"# {title_prefix} Audit & Negative-Result Diagnostic Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **Wave Mode**: `{wave_arg}` | **Targets Completed**: {len(results)} | **Elapsed**: {total_elapsed:.1f}s",
        "",
        "## 1. Negative-Result Triggers & Root-Cause Diagnostics (Priority Alert)",
        "",
    ]

    if not neg_targets:
        lines.append("No negative-result flags (`NEG_VS_SVM`, `NEG_VS_G0`, `NEG_OVERALL_VS_G0`, `HIGH_VAR`) triggered across any target.")
    else:
        lines.extend([
            f"Triggered on **{len(neg_targets)}/{len(results)}** targets:",
            "",
            "| Target | Assay | Flags | G2 - SVM (Cliff) | G2 - G0 (Cliff) | G2 - G1 (Gate) | L2 - A2 (Features) | Harmful LFER Channels |",
            "| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |",
        ])
        for r in neg_targets:
            flg_list = [k for k, v in r["flags"].items() if v]
            d = r["diagnosis"]
            harm = ", ".join(d["HARMFUL_CHANNELS"]) if d["HARMFUL_CHANNELS"] else "None"
            lines.append(
                f"| `{r['target']}` | {r['chemistry']['assay_type']} | `{'`, `'.join(flg_list)}` | "
                f"{d['delta_g2_minus_svm']:+.4f} | {d['delta_g2_minus_g0']:+.4f} | "
                f"{d['delta_g2_minus_g1']:+.4f} | {d['delta_l2_minus_a2']:+.4f} | {harm} |"
            )

    lines.extend([
        "",
        "## 2. Cross-Target Statistical Summary (Wilcoxon Signed-Rank Across Target Means)",
        "",
        "| Comparison | N Targets | Win / Tie / Loss | Mean Diff | Median Diff | Two-Sided p |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for name, st in cross_tests.items():
        lines.append(
            f"| {name} | {st['n']} | {st['wins']} / {st['ties']} / {st['losses']} | "
            f"{st['mean_diff']:+.5f} | {st['median_diff']:+.5f} | `{st['p_value']:.4e}` |"
        )

    lines.extend([
        "",
        "### Assay-Type Breakdown (`Ki` vs `EC50`)",
        "",
        "| Assay Group | N | Mean G2 Cliff RMSE | Mean G0 Cliff RMSE | Mean SVM Cliff RMSE | Mean (G2 - G0) | Mean (G2 - SVM) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for assay in ("Ki", "EC50"):
        sub = df_wide[df_wide["assay_type"] == assay]
        if len(sub) > 0:
            lines.append(
                f"| `{assay}` | {len(sub)} | {sub['g2_cliff'].mean():.4f} | {sub['g0_cliff'].mean():.4f} | "
                f"{sub['svm_cliff'].mean():.4f} | {sub['delta_g2_g0_cliff'].mean():+.4f} | {sub['delta_g2_svm_cliff'].mean():+.4f} |"
            )

    lines.extend([
        "",
        "## 3. Per-Target Ablation Table (5-Seed Mean ± Std Cliff RMSE)",
        "",
        "| Target | Assay | n (Test/Cliff) | B1c_SVM | A2_2D | L2_eSpan_LFER | G0 (SVM+A2) | G1 (SVM+L2) | G2_Prev | p(G2 vs SVM) | p(G2 vs G0) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in results:
        ch = r["chemistry"]
        m = r["metrics"]
        w = r["wilcoxon"]
        lines.append(
            f"| `{r['target']}` | {ch['assay_type']} | {ch['n_total']} ({ch['n_test']}/{ch['n_test_cliff']}) | "
            f"{m['B1c_RBF_SVM']['cliff_rmse_mean']:.4f} | "
            f"{m['A2_2D_Delta']['cliff_rmse_mean']:.4f} | "
            f"{m['L2_eSpan_LFER_Delta']['cliff_rmse_mean']:.4f} | "
            f"{m['G0_RBFSVM_Gate_A2']['cliff_rmse_mean']:.4f} | "
            f"{m['G1_RBFSVM_Gate_L2']['cliff_rmse_mean']:.4f} | "
            f"{m['G2_RBFSVM_PhysGate_L2']['cliff_rmse_mean']:.4f}±{m['G2_RBFSVM_PhysGate_L2']['cliff_rmse_std']:.4f} | "
            f"`{w['G2_vs_SVM']['p_cliff']:.2e}` | `{w['G2_vs_G0']['p_cliff']:.2e}` |"
        )

    lines.extend([
        "",
        "## 4. Target Chemistry & Gate Parameter Diagnostics",
        "",
        "| Target | Backend | AIMNet Fail | Mean MW | Mean LogP | Mean TPSA | Mean RotB | Mean e_span | G2 (ws, b, we, wq, alpha) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ])
    for r in results:
        ch = r["chemistry"]
        gm = r["gate_mean"]
        lines.append(
            f"| `{r['target']}` | `{ch['electronic_backend']}` | {ch['aimnet_fallback_count']}/{ch['n_total']} | "
            f"{ch['mean_mw']:.1f} | {ch['mean_logp']:.2f} | {ch['mean_tpsa']:.1f} | {ch['mean_rot_bonds']:.2f} | "
            f"{ch['mean_e_span']:.2f} | "
            f"ws={gm['g2_ws']:.2f}, b={gm['g2_b']:.2f}, we={gm['g2_we']:.3f}, wq={gm['g2_wq']:.3f}, alpha={gm['g2_alpha_mean']:.3f} |"
        )

    lines.extend([
        "",
        "## 5. Spearman Correlations: Performance Delta vs Chemical Properties",
        "",
        "| Property / Gate Param | Spearman vs Delta_Cliff(G2-G0) | p-val | Spearman vs Delta_Cliff(G2-SVM) | p-val |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    for cf in corr_features:
        r_g0 = [x for x in corr_rows if x["feature"] == cf and x["outcome"] == "Delta_Cliff(G2-G0)"][0]
        r_svm = [x for x in corr_rows if x["feature"] == cf and x["outcome"] == "Delta_Cliff(G2-SVM)"][0]
        lines.append(
            f"| `{cf}` | {r_g0['rho']:+.3f} | `{r_g0['p']:.3f}` | {r_svm['rho']:+.3f} | `{r_svm['p']:.3f}` |"
        )

    lines.extend([
        "",
        f"Audit Figure: `{fig_path.as_posix()}`",
        f"Summary CSV: `{csv_path.as_posix()}`",
    ])

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, fig_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run wave-based 30-target MoleculeACE benchmark for G2_Prev.")
    parser.add_argument("--wave", choices=["1", "2", "3", "all"], default="1")
    parser.add_argument("--targets", type=str, default="", help="Optional comma-separated target names.")
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--force", action="store_true", default=False)
    args = parser.parse_args()

    if args.targets.strip():
        target_list = [t.strip() for t in args.targets.split(",") if t.strip()]
    elif args.wave == "1":
        target_list = list(WAVE_1_TARGETS)
    elif args.wave == "2":
        target_list = list(WAVE_2_TARGETS)
    elif args.wave == "3":
        target_list = list(WAVE_3_TARGETS)
    else:
        target_list = list(ALL_30_TARGETS)

    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] Starting MoleculeACE wave={args.wave} "
        f"({len(target_list)} targets, n_workers={args.n_workers}, force={args.force})",
        flush=True,
    )
    t0 = time.time()
    results = [_evaluate_single_target(tgt, n_workers=args.n_workers, force=args.force) for tgt in target_list]
    total_elapsed = time.time() - t0
    csv_path, fig_path, md_path = _generate_reports(results, wave_arg=args.wave, total_elapsed=total_elapsed)
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] Wave={args.wave} completed in {total_elapsed:.1f}s.\n"
        f"  Report: {md_path}\n  CSV: {csv_path}\n  Figure: {fig_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()

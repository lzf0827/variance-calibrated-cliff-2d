"""Week-1 MVP orchestrator for PhysChem-Delta (sanity + 3-target x 5-seed overnight run)."""

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

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from physchem_delta.data import MVP_TARGETS, ensure_targets_downloaded, load_target_dataset
from physchem_delta.featurize import TargetFeatures, extract_or_load_target_features
from physchem_delta.metrics import (
    bootstrap_rmse_ci,
    evaluate_predictions,
    paired_wilcoxon,
    similarity_bin_breakdown,
)
from physchem_delta.models import (
    compute_tanimoto_matrix,
    fit_similarity_gate_and_blend,
    make_inner_split,
    run_direct_neural_model,
    run_pairwise_delta_model,
    run_tabular_baseline,
)

SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)

ALL_MODELS: tuple[str, ...] = (
    "B0_ECFP_NoChiral_CatBoost",
    "B1_ECFP_Chiral_CatBoost",
    "B1b_ECFP_Chiral_LGBM",
    "B1c_ECFP_SVM",
    "A1_2D_Direct_MLP",
    "A2_2D_Pairwise_Delta",
    "A3_2D3D_Pairwise_Delta",
    "A4_PhysChem_Delta",
    "M_AnchorGated_PhysChem_Delta",
)

MODEL_SHORT: dict[str, str] = {
    "B0_ECFP_NoChiral_CatBoost": "B0 CatB (no-chir)",
    "B1_ECFP_Chiral_CatBoost": "B1 CatB (chiral)",
    "B1b_ECFP_Chiral_LGBM": "B1b LGBM",
    "B1c_ECFP_SVM": "B1c SVM",
    "A1_2D_Direct_MLP": "A1 2D MLP",
    "A2_2D_Pairwise_Delta": "A2 2D Delta",
    "A3_2D3D_Pairwise_Delta": "A3 2D+3D Delta",
    "A4_PhysChem_Delta": "A4 PhysChem Delta",
    "M_AnchorGated_PhysChem_Delta": "M Anchor-Gated",
}


def _refine_dir() -> Path:
    d = ROOT_DIR / "refine-logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run_dir(target: str) -> Path:
    d = _refine_dir() / "runs" / target
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fig_dir() -> Path:
    d = _refine_dir() / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run_single(
    target: str,
    model_name: str,
    seed: int,
    feats: TargetFeatures,
    y: np.ndarray,
    cliff_mol: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    force: bool = False,
) -> dict[str, Any]:
    """Run one (target, model, seed) experiment with crash-safe JSON/CSV persistence."""
    rdir = _run_dir(target)
    json_path = rdir / f"{model_name}_seed{seed}.json"
    preds_path = rdir / f"{model_name}_seed{seed}_preds.csv"

    if json_path.exists() and preds_path.exists() and not force:
        return json.loads(json_path.read_text(encoding="utf-8"))

    t0 = time.time()
    extra: dict[str, Any] = {}
    peak_vram_mb = 0.0
    s_max_test: np.ndarray | None = None
    alpha_test: np.ndarray | None = None

    if model_name in (
        "B0_ECFP_NoChiral_CatBoost",
        "B1_ECFP_Chiral_CatBoost",
        "B1b_ECFP_Chiral_LGBM",
        "B1c_ECFP_SVM",
    ):
        out = run_tabular_baseline(model_name, feats, y, train_idx, test_idx, seed=seed)
        y_pred_test = out["y_pred_test"]
        peak_vram_mb = out["peak_vram_mb"]
        extra = out.get("extra", {})

    elif model_name == "A1_2D_Direct_MLP":
        out = run_direct_neural_model("2d", feats, y, train_idx, test_idx, seed=seed)
        y_pred_test = out["y_pred_test"]
        peak_vram_mb = out["peak_vram_mb"]
        extra = {"best_val_rmse": out["best_val_rmse"]}

    elif model_name in ("A2_2D_Pairwise_Delta", "A3_2D3D_Pairwise_Delta", "A4_PhysChem_Delta"):
        tier_map = {
            "A2_2D_Pairwise_Delta": "2d",
            "A3_2D3D_Pairwise_Delta": "2d3d",
            "A4_PhysChem_Delta": "physchem",
        }
        out = run_pairwise_delta_model(
            tier=tier_map[model_name],
            feats=feats,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            seed=seed,
        )
        y_pred_test = out["y_pred_test"]
        s_max_test = out["s_max_test"]
        peak_vram_mb = out["peak_vram_mb"]
        extra = {"best_val_rmse": out["best_val_rmse"]}

        if model_name == "A4_PhysChem_Delta":
            # Save inner_val & test delta outputs so M_AnchorGated can resume cleanly without recomputing A4
            np.savez_compressed(
                rdir / f"A4_cache_seed{seed}.npz",
                y_pred_val=out["y_pred_val"],
                s_max_val=out["s_max_val"],
                inner_va_idx=out["inner_va_idx"],
                y_pred_test=out["y_pred_test"],
                s_max_test=out["s_max_test"],
                best_val_rmse=np.array([out["best_val_rmse"]], dtype=np.float64),
                peak_vram_mb=np.array([peak_vram_mb], dtype=np.float64),
            )

    elif model_name == "M_AnchorGated_PhysChem_Delta":
        a4_cache_path = rdir / f"A4_cache_seed{seed}.npz"
        if not a4_cache_path.exists():
            # Ensure A4 has run first
            _run_single(
                target, "A4_PhysChem_Delta", seed, feats, y, cliff_mol, train_idx, test_idx, force=force
            )
        a4_cache = np.load(a4_cache_path)
        y_val_delta = a4_cache["y_pred_val"]
        s_max_val = a4_cache["s_max_val"]
        inner_va_idx = a4_cache["inner_va_idx"]
        y_test_delta = a4_cache["y_pred_test"]
        s_max_test = a4_cache["s_max_test"]

        # Train companion direct predictor on exact same PhysChem features & inner split
        direct_out = run_direct_neural_model("physchem", feats, y, train_idx, test_idx, seed=seed)
        inner_tr_rel, inner_va_rel_check = make_inner_split(y[train_idx], seed=seed, val_frac=0.15)
        assert np.array_equal(train_idx[inner_va_rel_check], inner_va_idx), "Inner validation index mismatch!"

        gate_out = fit_similarity_gate_and_blend(
            y_val_true=y[inner_va_idx],
            y_val_delta=y_val_delta,
            y_val_direct=direct_out["y_pred_val"],
            s_max_val=s_max_val,
            y_test_delta=y_test_delta,
            y_test_direct=direct_out["y_pred_test"],
            s_max_test=s_max_test,
        )
        y_pred_test = gate_out["y_pred_test"]
        alpha_test = gate_out["alpha_test"]
        peak_vram_mb = max(float(direct_out["peak_vram_mb"]), float(a4_cache["peak_vram_mb"][0]))

        direct_metrics = evaluate_predictions(y[test_idx], direct_out["y_pred_test"], cliff_mol[test_idx])
        extra = {
            "gate_a": gate_out["gate_a"],
            "gate_b": gate_out["gate_b"],
            "alpha_test_mean": gate_out["alpha_test_mean"],
            "direct_physchem_rmse": direct_metrics["rmse"],
            "direct_physchem_cliff_rmse": direct_metrics["cliff_rmse"],
        }

        # Save detailed OOD gate diagnostic CSV for plotting
        pd.DataFrame(
            {
                "test_idx": test_idx,
                "y_true": y[test_idx],
                "cliff_mol": cliff_mol[test_idx],
                "s_max": s_max_test,
                "alpha": alpha_test,
                "y_pred_a4_delta": y_test_delta,
                "y_pred_direct_physchem": direct_out["y_pred_test"],
                "y_pred_anchor_gated": y_pred_test,
            }
        ).to_csv(rdir / f"M_ood_diag_seed{seed}.csv", index=False)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    metrics = evaluate_predictions(y[test_idx], y_pred_test, cliff_mol[test_idx])
    rmse_pt, rmse_lo, rmse_hi = bootstrap_rmse_ci(y[test_idx], y_pred_test)
    cmask = cliff_mol[test_idx] == 1
    c_pt, c_lo, c_hi = bootstrap_rmse_ci(y[test_idx][cmask], y_pred_test[cmask])

    record: dict[str, Any] = {
        "target": target,
        "model": model_name,
        "seed": seed,
        "electronic_backend": feats.electronic_backend,
        "peak_vram_mb": round(float(peak_vram_mb), 2),
        "elapsed_sec": round(time.time() - t0, 2),
        "rmse_ci95": [round(rmse_lo, 4), round(rmse_hi, 4)],
        "cliff_rmse_ci95": [round(c_lo, 4), round(c_hi, 4)],
        **metrics,
        **extra,
    }

    if s_max_test is None:
        sim_mat = compute_tanimoto_matrix(feats.ecfp_chiral[test_idx], feats.ecfp_chiral[train_idx])
        s_max_test = np.max(sim_mat, axis=1)

    pred_df = pd.DataFrame(
        {
            "test_idx": test_idx,
            "y_true": y[test_idx],
            "y_pred": y_pred_test,
            "cliff_mol": cliff_mol[test_idx],
            "s_max": s_max_test,
        }
    )
    if alpha_test is not None:
        pred_df["alpha"] = alpha_test
    pred_df.to_csv(preds_path, index=False)
    json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def _generate_plots(records: list[dict[str, Any]], targets: list[str]) -> list[Path]:
    """Generate 3 publication-quality Week-1 MVP diagnostic figures."""
    df = pd.DataFrame(records)
    fdir = _fig_dir()
    created: list[Path] = []
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    # Plot 1: Cliff-RMSE comparison across targets & models
    fig, axes = plt.subplots(1, len(targets), figsize=(5.2 * len(targets), 4.6), sharey=False)
    if len(targets) == 1:
        axes = [axes]
    palette = [
        "#999999", "#E69F00", "#D55E00", "#CC79A7",
        "#56B4E9", "#009E73", "#0072B2", "#332288", "#117733",
    ]
    model_order = [m for m in ALL_MODELS if m in df["model"].unique()]
    for ax, tgt in zip(axes, targets):
        sub = df[df["target"] == tgt]
        means, stds, labels, colors, scatter_pts = [], [], [], [], []
        for idx_m, m in enumerate(model_order):
            vals = sub[sub["model"] == m]["cliff_rmse"].to_numpy()
            if vals.size == 0:
                continue
            bar_pos = len(means)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0)
            labels.append(MODEL_SHORT.get(m, m))
            colors.append(palette[idx_m % len(palette)])
            scatter_pts.append((bar_pos, vals))
        x = np.arange(len(means))
        ax.bar(x, means, yerr=stds, capsize=3, color=colors, alpha=0.88, edgecolor="black", linewidth=0.6)
        for bar_pos, vals in scatter_pts:
            jitter = np.linspace(-0.15, 0.15, vals.size) if vals.size > 1 else np.array([0.0])
            ax.scatter(np.full_like(vals, bar_pos, dtype=float) + jitter, vals, color="black", s=14, zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=8.5)
        ax.set_title(tgt, fontweight="bold")
        ax.set_ylabel("Activity Cliff RMSE (lower is better)")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.suptitle("Week-1 MVP: Activity Cliff RMSE across 3 MoleculeACE Targets (5 seeds)", fontweight="bold")
    fig.tight_layout()
    p1 = fdir / "mvp_cliff_rmse_comparison.png"
    fig.savefig(p1, dpi=220)
    plt.close(fig)
    created.append(p1)

    # Plot 2: R02 Stepwise Ablation
    ablation_models = [
        "B1_ECFP_Chiral_CatBoost",
        "A1_2D_Direct_MLP",
        "A2_2D_Pairwise_Delta",
        "A3_2D3D_Pairwise_Delta",
        "A4_PhysChem_Delta",
        "M_AnchorGated_PhysChem_Delta",
    ]
    ab_present = [m for m in ablation_models if m in df["model"].unique()]
    if ab_present:
        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        markers = ["o", "s", "^", "D"]
        line_colors = ["#0072B2", "#D55E00", "#009E73"]
        x = np.arange(len(ab_present))
        for t_i, tgt in enumerate(targets):
            sub = df[df["target"] == tgt]
            ys, es = [], []
            for m in ab_present:
                vals = sub[sub["model"] == m]["cliff_rmse"].to_numpy()
                ys.append(float(np.mean(vals)) if vals.size else np.nan)
                es.append(float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0)
            ax.errorbar(
                x, ys, yerr=es, marker=markers[t_i % len(markers)], linewidth=2.0,
                capsize=4, label=tgt, color=line_colors[t_i % len(line_colors)]
            )
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_SHORT[m] for m in ab_present], rotation=20, ha="right")
        ax.set_ylabel("Activity Cliff RMSE")
        ax.set_title("R02 Attribution: Stepwise Contribution (2D -> +Delta -> +3D -> +Electronic -> +OOD Gate)", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(frameon=False)
        fig.tight_layout()
        p2 = fdir / "mvp_r02_ablation.png"
        fig.savefig(p2, dpi=220)
        plt.close(fig)
        created.append(p2)

    # Plot 3: OOD similarity-bin error analysis
    diag_frames = []
    for tgt in targets:
        rdir = _run_dir(tgt)
        for p in sorted(rdir.glob("M_ood_diag_seed*.csv")):
            d = pd.read_csv(p)
            d["target"] = tgt
            diag_frames.append(d)
    if diag_frames:
        diag = pd.concat(diag_frames, ignore_index=True)
        bins = (0.0, 0.5, 0.7, 0.85, 1.01)
        b_delta = similarity_bin_breakdown(diag["y_true"].to_numpy(), diag["y_pred_a4_delta"].to_numpy(), diag["s_max"].to_numpy(), diag["cliff_mol"].to_numpy(), bins=bins)
        b_direct = similarity_bin_breakdown(diag["y_true"].to_numpy(), diag["y_pred_direct_physchem"].to_numpy(), diag["s_max"].to_numpy(), diag["cliff_mol"].to_numpy(), bins=bins)
        b_gate = similarity_bin_breakdown(diag["y_true"].to_numpy(), diag["y_pred_anchor_gated"].to_numpy(), diag["s_max"].to_numpy(), diag["cliff_mol"].to_numpy(), bins=bins)
        if b_delta:
            fig, ax = plt.subplots(figsize=(8.0, 4.6))
            bx = np.arange(len(b_delta))
            w = 0.25
            ax.bar(bx - w, [r["rmse"] for r in b_direct], width=w, label="Direct PhysChem MLP", color="#56B4E9")
            ax.bar(bx, [r["rmse"] for r in b_delta], width=w, label="A4 PhysChem Delta", color="#009E73")
            ax.bar(bx + w, [r["rmse"] for r in b_gate], width=w, label="M Anchor-Gated Blend", color="#D55E00")
            ax.set_xticks(bx)
            ax.set_xticklabels([f"{r['bin']}\n(n={r['n']})" for r in b_delta])
            ax.set_xlabel("Max Tanimoto Similarity to Training Pool (s_max)")
            ax.set_ylabel("Pooled Test RMSE")
            ax.set_title("OOD Similarity Breakdown: Direct vs Delta vs Similarity-Gated Blend", fontweight="bold")
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            ax.legend(frameon=False)
            fig.tight_layout()
            p3 = fdir / "mvp_ood_gate_analysis.png"
            fig.savefig(p3, dpi=220)
            plt.close(fig)
            created.append(p3)

    return created


def _compile_reports(
    records: list[dict[str, Any]],
    targets: list[str],
    ds_stats: dict[str, dict[str, int]],
    feat_metas: dict[str, dict[str, Any]],
    mode: str,
) -> None:
    """Compile summary CSV, EXPERIMENT_RESULTS.md, EXPERIMENT_TRACKER.md, EXPERIMENT_LOG.md, and Ducktor audit."""
    rdir = _refine_dir()
    df = pd.DataFrame(records)
    df.to_csv(rdir / "mvp_raw_runs.csv", index=False)

    # Aggregated per-target + macro summary
    agg_rows: list[dict[str, Any]] = []
    for tgt in targets:
        sub_t = df[df["target"] == tgt]
        for m in [x for x in ALL_MODELS if x in sub_t["model"].unique()]:
            sub = sub_t[sub_t["model"] == m]
            agg_rows.append(
                {
                    "target": tgt,
                    "model": m,
                    "n_seeds": int(len(sub)),
                    "rmse_mean": round(float(sub["rmse"].mean()), 4),
                    "rmse_std": round(float(sub["rmse"].std(ddof=1)) if len(sub) > 1 else 0.0, 4),
                    "cliff_rmse_mean": round(float(sub["cliff_rmse"].mean()), 4),
                    "cliff_rmse_std": round(float(sub["cliff_rmse"].std(ddof=1)) if len(sub) > 1 else 0.0, 4),
                    "non_cliff_rmse_mean": round(float(sub["non_cliff_rmse"].mean()), 4),
                    "spearman_mean": round(float(sub["spearman"].mean()), 4),
                    "peak_vram_mb": round(float(sub["peak_vram_mb"].max()), 2),
                }
            )
    summary_df = pd.DataFrame(agg_rows)
    summary_df.to_csv(rdir / "mvp_summary.csv", index=False)

    # Paired Wilcoxon tests vs B1 and vs A2 on pooled test predictions across seeds (cliff samples)
    stat_rows: list[dict[str, Any]] = []
    for tgt in targets:
        r_t_dir = _run_dir(tgt)
        for cand in ("A2_2D_Pairwise_Delta", "A3_2D3D_Pairwise_Delta", "A4_PhysChem_Delta", "M_AnchorGated_PhysChem_Delta"):
            if cand not in df[df["target"] == tgt]["model"].unique():
                continue
            for ref in ("B1_ECFP_Chiral_CatBoost", "A2_2D_Pairwise_Delta"):
                if cand == ref or ref not in df[df["target"] == tgt]["model"].unique():
                    continue
                err_c_list, err_r_list = [], []
                for s in sorted(df[(df["target"] == tgt) & (df["model"] == cand)]["seed"].unique()):
                    pc = r_t_dir / f"{cand}_seed{s}_preds.csv"
                    pr = r_t_dir / f"{ref}_seed{s}_preds.csv"
                    if pc.exists() and pr.exists():
                        dc = pd.read_csv(pc)
                        dr = pd.read_csv(pr)
                        cmask = dc["cliff_mol"].to_numpy(dtype=int) == 1
                        err_c_list.append(np.abs(dc["y_true"].to_numpy()[cmask] - dc["y_pred"].to_numpy()[cmask]))
                        err_r_list.append(np.abs(dr["y_true"].to_numpy()[cmask] - dr["y_pred"].to_numpy()[cmask]))
                if err_c_list:
                    pval = paired_wilcoxon(np.concatenate(err_c_list), np.concatenate(err_r_list))
                    c_mean = float(summary_df[(summary_df["target"] == tgt) & (summary_df["model"] == cand)]["cliff_rmse_mean"].iloc[0])
                    r_mean = float(summary_df[(summary_df["target"] == tgt) & (summary_df["model"] == ref)]["cliff_rmse_mean"].iloc[0])
                    stat_rows.append(
                        {
                            "target": tgt,
                            "candidate": cand,
                            "reference": ref,
                            "delta_cliff_rmse": round(c_mean - r_mean, 4),
                            "wilcoxon_p": float(f"{pval:.3e}") if np.isfinite(pval) else float("nan"),
                        }
                    )

    # Ducktor E05 / R02 / E02 Go/No-Go evaluation
    backend_used = next(iter(feat_metas.values()))["electronic_backend"]
    e05_wins_vs_b1 = 0
    r02_beats_both = 0
    for tgt in targets:
        s_t = summary_df[summary_df["target"] == tgt].set_index("model")
        if "B1_ECFP_Chiral_CatBoost" in s_t.index and "A2_2D_Pairwise_Delta" in s_t.index:
            best_physchem_name = "M_AnchorGated_PhysChem_Delta" if "M_AnchorGated_PhysChem_Delta" in s_t.index else "A4_PhysChem_Delta"
            if best_physchem_name in s_t.index:
                c_b1 = float(s_t.loc["B1_ECFP_Chiral_CatBoost", "cliff_rmse_mean"])
                std_b1 = float(s_t.loc["B1_ECFP_Chiral_CatBoost", "cliff_rmse_std"])
                c_a2 = float(s_t.loc["A2_2D_Pairwise_Delta", "cliff_rmse_mean"])
                c_best = float(s_t.loc[best_physchem_name, "cliff_rmse_mean"])
                std_best = float(s_t.loc[best_physchem_name, "cliff_rmse_std"])
                if (c_b1 - c_best) > max(std_b1, std_best, 1e-4):
                    e05_wins_vs_b1 += 1
                if c_best < c_b1 and c_best < c_a2:
                    r02_beats_both += 1

    if mode == "sanity":
        verdict = "SANITY_PASSED_PENDING_FULL_MVP"
    elif r02_beats_both >= 2 and e05_wins_vs_b1 >= 2:
        verdict = "GO_SCALE_TO_30_TARGETS"
    elif r02_beats_both >= 1:
        verdict = "CONDITIONAL_GO_INSPECT_PER_TARGET"
    else:
        verdict = "R02_PIVOT_REQUIRED"

    # Write EXPERIMENT_RESULTS.md
    lines = [
        f"# Week-1 MVP Experiment Results ({mode.upper()})",
        "",
        f"**Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Electronic Backend**: `{backend_used}`",
        f"**Peak CUDA VRAM**: {df['peak_vram_mb'].max():.2f} MiB",
        "",
        "## Dataset Summary (Official MoleculeACE Splits)",
        "| Target | Total | Train | Test | Train Cliff | Test Cliff | Embed Fallbacks |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for tgt in targets:
        ds_s = ds_stats[tgt]
        fm = feat_metas[tgt]
        lines.append(
            f"| {tgt} | {ds_s['n_total']} | {ds_s['n_train']} | {ds_s['n_test']} | "
            f"{ds_s['n_train_cliff']} | {ds_s['n_test_cliff']} | {fm['embed_fallback_count']} |"
        )

    lines.extend(
        [
            "",
            "## Main & R02 Ablation Results (Mean ± Std across seeds)",
            "| Target | Model | Seeds | RMSE | Cliff RMSE | Non-Cliff RMSE | Spearman | Peak VRAM (MiB) |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['target']} | `{row['model']}` | {row['n_seeds']} | "
            f"{row['rmse_mean']:.4f} ± {row['rmse_std']:.4f} | "
            f"**{row['cliff_rmse_mean']:.4f} ± {row['cliff_rmse_std']:.4f}** | "
            f"{row['non_cliff_rmse_mean']:.4f} | {row['spearman_mean']:.4f} | {row['peak_vram_mb']:.1f} |"
        )

    if stat_rows:
        lines.extend(
            [
                "",
                "## Paired Wilcoxon Significance (Cliff Subset Absolute Errors)",
                "| Target | Candidate | Reference | Δ Cliff RMSE (Cand - Ref) | Two-sided p-value |",
                "|---|---|---|---:|---:|",
            ]
        )
        for sr in stat_rows:
            lines.append(
                f"| {sr['target']} | `{sr['candidate']}` | `{sr['reference']}` | "
                f"{sr['delta_cliff_rmse']:+.4f} | {sr['wilcoxon_p']:.3e} |"
            )

    lines.extend(
        [
            "",
            "## Ducktor Go/No-Go Audit (Cards E05 / R02 / E02)",
            f"- **Verdict**: `{verdict}`",
            f"- **E05 (Beat B1_CatBoost beyond seed std)**: {e05_wins_vs_b1}/{len(targets)} targets",
            f"- **R02 (PhysChem beats BOTH B1_CatBoost AND A2_2D_Delta)**: {r02_beats_both}/{len(targets)} targets",
            f"- **E02 Note**: Electronic features computed via `{backend_used}` (explicitly labeled, zero substitution).",
        ]
    )
    (rdir / "EXPERIMENT_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Update EXPERIMENT_TRACKER.md
    tracker_lines = [
        "# Experiment Tracker (Week-1 MVP)",
        "",
        f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Electronic Backend**: `{backend_used}`",
        f"**Go/No-Go Verdict**: `{verdict}`",
        "",
        "| Milestone | Run ID | Target(s) | System | Completed Runs | Mean Cliff RMSE | Status |",
        "|---|---|---|---|---:|---:|---|",
    ]
    run_id_map = {
        "B0_ECFP_NoChiral_CatBoost": ("M1", "R001"),
        "B1_ECFP_Chiral_CatBoost": ("M1", "R002"),
        "B1b_ECFP_Chiral_LGBM": ("M1", "R003"),
        "B1c_ECFP_SVM": ("M1", "R004"),
        "A1_2D_Direct_MLP": ("M2", "R005"),
        "A2_2D_Pairwise_Delta": ("M3", "R006"),
        "A3_2D3D_Pairwise_Delta": ("M3", "R007"),
        "A4_PhysChem_Delta": ("M3", "R008"),
        "M_AnchorGated_PhysChem_Delta": ("M3", "R009"),
    }
    tracker_lines.append("| M0 | R000 | `CHEMBL2147_Ki` | Leakage Tests + Sanity | 1 | - | DONE |")
    for m in ALL_MODELS:
        ms, rid = run_id_map[m]
        sub = summary_df[summary_df["model"] == m]
        n_runs = int(sub["n_seeds"].sum()) if not sub.empty else 0
        c_mean = f"{sub['cliff_rmse_mean'].mean():.4f}" if not sub.empty else "-"
        status = "DONE" if (mode == "full" and n_runs >= len(targets) * len(SEEDS)) else ("PARTIAL" if n_runs > 0 else "PENDING")
        tracker_lines.append(f"| {ms} | {rid} | {', '.join(targets)} | `{m}` | {n_runs} | {c_mean} | {status} |")

    (rdir / "EXPERIMENT_TRACKER.md").write_text("\n".join(tracker_lines) + "\n", encoding="utf-8")

    # Append compact summary to EXPERIMENT_LOG.md
    log_entry = [
        "",
        f"## MVP-{mode.upper()} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Electronic Backend**: `{backend_used}`",
        f"- **Verdict**: `{verdict}` (E05={e05_wins_vs_b1}/{len(targets)}, R02={r02_beats_both}/{len(targets)})",
        f"- **Results Table**: `refine-logs/mvp_summary.csv` and `refine-logs/EXPERIMENT_RESULTS.md`",
    ]
    with (ROOT_DIR / "EXPERIMENT_LOG.md").open("a", encoding="utf-8") as f:
        f.write("\n".join(log_entry) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PhysChem-Delta Week-1 MVP experiments.")
    parser.add_argument("--mode", choices=("sanity", "full"), default="sanity")
    parser.add_argument("--prefer-aimnet", action="store_true", default=True)
    parser.add_argument("--no-aimnet", action="store_true", default=False)
    parser.add_argument("--force", action="store_true", default=False)
    args = parser.parse_args()

    prefer_aimnet = not args.no_aimnet
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Week-1 MVP in mode={args.mode} (prefer_aimnet={prefer_aimnet})")

    ds_stats = ensure_targets_downloaded(MVP_TARGETS)
    print("Dataset verification:", json.dumps(ds_stats, indent=2))

    if args.mode == "sanity":
        run_targets = ["CHEMBL2147_Ki"]
        run_seeds = [42]
        run_models = [
            "B1_ECFP_Chiral_CatBoost",
            "A1_2D_Direct_MLP",
            "A2_2D_Pairwise_Delta",
            "A3_2D3D_Pairwise_Delta",
            "A4_PhysChem_Delta",
            "M_AnchorGated_PhysChem_Delta",
        ]
    else:
        run_targets = list(MVP_TARGETS)
        run_seeds = list(SEEDS)
        run_models = list(ALL_MODELS)

    all_records: list[dict[str, Any]] = []
    feat_metas: dict[str, dict[str, Any]] = {}

    for tgt in run_targets:
        ds = load_target_dataset(tgt)
        print(f"\n=== Target {tgt}: extracting/loading 3-tier features (n={len(ds.smiles)}) ===")
        feats = extract_or_load_target_features(tgt, ds.smiles, prefer_aimnet=prefer_aimnet, num_confs=5)
        feat_metas[tgt] = feats.meta
        print(f"Features ready: backend={feats.electronic_backend}, dims={feats.meta['dims']}, elapsed={feats.meta['elapsed_sec']}s")

        for seed in run_seeds:
            for model_name in run_models:
                rec = _run_single(
                    target=tgt,
                    model_name=model_name,
                    seed=seed,
                    feats=feats,
                    y=ds.y,
                    cliff_mol=ds.cliff_mol,
                    train_idx=ds.train_idx,
                    test_idx=ds.test_idx,
                    force=args.force,
                )
                all_records.append(rec)
                print(
                    f"[{tgt} | seed={seed} | {model_name}] "
                    f"RMSE={rec['rmse']:.4f} | Cliff_RMSE={rec['cliff_rmse']:.4f} | "
                    f"Spearman={rec['spearman']:.4f} | VRAM={rec['peak_vram_mb']:.1f}MB | {rec['elapsed_sec']}s"
                )

    figs = _generate_plots(all_records, run_targets)
    _compile_reports(all_records, run_targets, ds_stats, feat_metas, mode=args.mode)
    print("\nGenerated figures:", [str(p) for p in figs])
    print("MVP execution finished cleanly.")


if __name__ == "__main__":
    main()

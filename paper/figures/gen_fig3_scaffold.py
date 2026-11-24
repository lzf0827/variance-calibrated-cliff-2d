"""Generate Figure 3: Strict Bemis-Murcko Scaffold Split Robustness & Label-Deoverlapped Self-Limiting Weights."""

from __future__ import annotations

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jcim_plot_style import PALETTE, REFINELOGS_DIR, apply_jcim_style, panel_tag, save_jcim_fig


def _load_smax_bins(targets: list[str]) -> tuple[list[str], list[int], dict[str, list[float]]]:
    models = ("K_RBF_SVM", "K_Multi_SVR", "Y_FiLM_2DLFER", "E_Multi_DirectMLP_OptW", "E_Multi_FiLM_OptW", "E_Multi_FiLM_GroupOptW")
    npz_dir = REFINELOGS_DIR / "scaffold_robustness_results" / "per_target"

    ys, smaxes = [], []
    preds = {m: [] for m in models}
    for tgt in targets:
        z = np.load(npz_dir / f"{tgt}_scaffold_preds.npz")
        ys.append(z["y_pooled"])
        smaxes.append(z["s_max_pooled"])
        for m in models:
            preds[m].append(z[f"{m}_pooled"])

    y_vec = np.concatenate(ys, axis=0)
    s_vec = np.concatenate(smaxes, axis=0)
    p_map = {m: np.concatenate(preds[m], axis=0) for m in models}

    regimes = [
        (r"$s_{\max} < 0.50$", s_vec < 0.50),
        ("[0.50, 0.70)", (s_vec >= 0.50) & (s_vec < 0.70)),
        (r"$s_{\max} \geq 0.70$", s_vec >= 0.70),
    ]
    lbls, cnts = [], []
    rmses = {m: [] for m in models}
    for lbl, mask in regimes:
        lbls.append(lbl)
        cnts.append(int(np.sum(mask)))
        for m in models:
            rmses[m].append(float(np.sqrt(np.mean((y_vec[mask] - p_map[m][mask]) ** 2))))
    return lbls, cnts, rmses


def main() -> None:
    apply_jcim_style(base_font=7.5)

    scaf = pd.read_csv(REFINELOGS_DIR / "scaffold_robustness_results" / "scaffold_summary.csv")
    wide = pd.read_csv(REFINELOGS_DIR / "scaffold_robustness_results" / "scaffold_wide.csv")
    s_lbls, s_cnts, s_rmses = _load_smax_bins(wide["target"].tolist())

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.0, 2.75), gridspec_kw={"wspace": 0.36})

    # Panel A: 10-Target Mean Overall RMSE on Strict Murcko Scaffold Split
    bar_specs = [
        ("A1_Direct_MLP", "DirectMLP", PALETTE["lightgray"]),
        ("B1_CatBoost", "CatBoost", PALETTE["skyblue"]),
        ("Y_FiLM_2DLFER", "Y_FiLM", PALETTE["yellow"]),
        ("K_RBF_SVM", "RBF-SVM", PALETTE["darkgray"]),
        ("K_Multi_SVR", "MultiSVR", PALETTE["blue"]),
        ("E_Multi_DirectMLP_GroupOptW", "GrpOptW-MLP", PALETTE["purple"]),
        ("E_Multi_FiLM", "FiLM(0.5)", PALETTE["amber"]),
        ("E_Multi_FiLM_PriorN", "PriorN-FiLM", PALETTE["teal"]),
        ("E_Multi_FiLM_OptW", "OptW-FiLM", PALETTE["orange"]),
        ("E_Multi_FiLM_GroupOptW", "GrpOptW-FiLM", PALETTE["vermilion"]),
    ]
    agg = scaf.groupby("model")["rmse_mean"].mean()
    x = np.arange(len(bar_specs))
    ovals = [float(agg.loc[m]) for m, _, _ in bar_specs]
    cols = [c for _, _, c in bar_specs]
    lbls = [l for _, l, _ in bar_specs]

    ax1.bar(x, ovals, width=0.66, color=cols, edgecolor="black", linewidth=0.45)
    for xi, ov in zip(x, ovals):
        ax1.text(xi, ov + 0.0018, f"{ov:.3f}", ha="center", va="bottom", fontsize=5.1, rotation=65)
    ax1.set_xticks(x)
    ax1.set_xticklabels(lbls, rotation=38, ha="right", fontsize=5.8)
    ax1.set_ylim(0.695, max(ovals) + 0.025)
    ax1.set_ylabel("10-Target Scaffold Overall RMSE")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    panel_tag(ax1, "A", x=-0.19, y=1.04)

    # Panel B: Similarity-Stratified Pooled OOD RMSE
    xb = np.arange(len(s_lbls))
    curves = [
        ("Y_FiLM_2DLFER", "Y-FiLM", PALETTE["amber"], ":^"),
        ("K_RBF_SVM", "RBF-SVM", PALETTE["darkgray"], "--o"),
        ("K_Multi_SVR", "MultiSVR", PALETTE["blue"], "--s"),
        ("E_Multi_DirectMLP_OptW", "OptW-MLP", PALETTE["purple"], "-.v"),
        ("E_Multi_FiLM_OptW", "OptW-FiLM", PALETTE["orange"], "-D"),
        ("E_Multi_FiLM_GroupOptW", "GrpOptW-FiLM", PALETTE["vermilion"], "-P"),
    ]
    for m_key, lbl, col, fmt in curves:
        ax2.plot(xb, s_rmses[m_key], fmt, color=col, linewidth=1.35, markersize=3.8, label=lbl)
    ax2.set_xticks(xb)
    ax2.set_xticklabels([f"{l}\n($n={n}$)" for l, n in zip(s_lbls, s_cnts)], fontsize=5.9)
    ax2.set_xlabel(r"Nearest-Train Tanimoto ($s_{\max}$)")
    ax2.set_ylabel("Pooled OOD RMSE")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(frameon=False, fontsize=5.5, loc="upper right")
    panel_tag(ax2, "B", x=-0.19, y=1.04)

    # Panel C: Stratified vs GroupKFold Learned w*_FiLM with Non-Overlapping Manual Offsets
    ax3.scatter(wide["scaf_mean_smax"], wide["w_strat_film"], color=PALETTE["orange"], s=28, edgecolor="black", linewidth=0.5, label=rf"Strat OOF (mean = {wide['w_strat_film'].mean():.3f})", zorder=3)
    ax3.scatter(wide["scaf_mean_smax"], wide["w_group_film"], color=PALETTE["blue"], marker="^", s=34, edgecolor="black", linewidth=0.5, label=rf"GroupKFold OOF (mean = {wide['w_group_film'].mean():.3f})", zorder=4)
    for _, r in wide.iterrows():
        ax3.plot([r["scaf_mean_smax"], r["scaf_mean_smax"]], [r["w_strat_film"], r["w_group_film"]], color="#777777", alpha=0.5, linewidth=0.75, zorder=2)

    # Non-overlapping absolute coordinates (tx, ty, ha, anchor_col) verified against scatter positions
    label_specs = {
        "CHEMBL1862_Ki": (0.562, 0.335, "left", "w_strat_film"),
        "CHEMBL1871_Ki": (0.602, 0.165, "left", "w_group_film"),
        "CHEMBL228_Ki": (0.656, 0.015, "left", "w_group_film"),
        "CHEMBL2147_Ki": (0.638, 0.595, "left", "w_group_film"),
        "CHEMBL204_Ki": (0.688, 0.505, "right", "w_strat_film"),
        "CHEMBL218_EC50": (0.726, 0.565, "left", "w_group_film"),
        "CHEMBL244_Ki": (0.744, 0.415, "left", "w_strat_film"),
        "CHEMBL2835_Ki": (0.790, 0.595, "center", "w_group_film"),
        "CHEMBL237_Ki": (0.734, 0.175, "left", "w_group_film"),
        "CHEMBL235_EC50": (0.686, 0.105, "right", "w_group_film"),
    }
    for _, r in wide.iterrows():
        tgt = r["target"]
        short = tgt.replace("CHEMBL", "C")
        tx, ty, ha, anchor_col = label_specs[tgt]
        x0 = float(r["scaf_mean_smax"])
        y0 = float(r[anchor_col])
        ax3.annotate(
            short,
            xy=(x0, y0),
            xytext=(tx, ty),
            ha=ha,
            va="center",
            fontsize=5.2,
            color="#222222",
            arrowprops=dict(arrowstyle="-", lw=0.45, color="#888888", shrinkA=0, shrinkB=2),
        )

    ax3.set_xlabel(r"Scaffold Test–Train Mean $s_{\max}$", fontsize=7.3)
    ax3.set_ylabel(r"Learned OOF Weight ($w^*_{\mathrm{FiLM}}$)")
    ax3.set_xlim(0.555, 0.830)
    ax3.set_ylim(-0.02, 0.72)
    ax3.grid(True, linestyle="--", alpha=0.3)
    ax3.legend(frameon=False, fontsize=5.6, loc="upper left")
    panel_tag(ax3, "C", x=-0.19, y=1.04)

    save_jcim_fig(fig, "fig3_scaffold", also_tiff=False)


if __name__ == "__main__":
    main()

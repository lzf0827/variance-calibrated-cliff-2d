"""Generate Figure 2: 30-Target Main Results, Learned w* Regime Reversal, and |Delta y| Anatomy Inset."""

from __future__ import annotations

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jcim_plot_style import PALETTE, REFINELOGS_DIR, ROOT_DIR, apply_jcim_style, panel_tag, save_jcim_fig

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from physchem_delta.data import load_target_dataset
from physchem_delta.models import compute_tanimoto_matrix
from physchem_delta.run_jcim_30targets import extract_or_load_2d_features


def _compute_analog_potency_bins(targets: list[str]) -> tuple[list[str], list[int], dict[str, list[float]]]:
    models = ("K_RBF_SVM", "K_Multi_SVR", "E_Multi_DirectMLP_OptW", "E_Multi_FiLM_OptW")
    npz_dir = REFINELOGS_DIR / "jcim30_step_a_results" / "per_target"

    recs: list[dict[str, float]] = []
    for tgt in targets:
        ds = load_target_dataset(tgt)
        feats = extract_or_load_2d_features(tgt, ds.smiles)
        z = np.load(npz_dir / f"{tgt}_optw_preds.npz")

        sim = compute_tanimoto_matrix(feats.ecfp_chiral[ds.test_idx], feats.ecfp_chiral[ds.train_idx])
        best_pos = np.argmax(sim, axis=1)
        s_max = sim[np.arange(ds.test_idx.size), best_pos].astype(np.float64)
        abs_dy = np.abs(ds.y[ds.test_idx] - ds.y[ds.train_idx[best_pos]])

        mask = s_max >= 0.70
        y_sub = ds.y[ds.test_idx][mask]
        dy_sub = abs_dy[mask]
        p_sub = {m: z[f"{m}_mean"].astype(np.float64)[mask] for m in models}
        for i in range(y_sub.size):
            row = {"y": float(y_sub[i]), "abs_dy": float(dy_sub[i])}
            for m in models:
                row[m] = float(p_sub[m][i])
            recs.append(row)

    df = pd.DataFrame(recs)
    bins = [
        ("< 0.5", df["abs_dy"] < 0.5),
        ("[0.5, 1.0)", (df["abs_dy"] >= 0.5) & (df["abs_dy"] < 1.0)),
        ("[1.0, 1.5)", (df["abs_dy"] >= 1.0) & (df["abs_dy"] < 1.5)),
        (r"$\geq 1.5$", df["abs_dy"] >= 1.5),
    ]
    labels, counts = [], []
    rmses: dict[str, list[float]] = {m: [] for m in models}
    for lbl, bmask in bins:
        sub = df[bmask]
        labels.append(lbl)
        counts.append(int(len(sub)))
        y_v = sub["y"].to_numpy()
        for m in models:
            rmses[m].append(float(np.sqrt(np.mean((y_v - sub[m].to_numpy()) ** 2))))
    return labels, counts, rmses


def main() -> None:
    apply_jcim_style(base_font=7.5)

    step_a = pd.read_csv(REFINELOGS_DIR / "jcim30_step_a_results" / "step_a_summary.csv")
    wide = pd.read_csv(REFINELOGS_DIR / "jcim30_step_a_results" / "step_a_wide.csv")
    bin_lbls, bin_cnts, bin_rmses = _compute_analog_potency_bins(wide["target"].tolist())

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.0, 2.75), gridspec_kw={"wspace": 0.36})

    # Panel A: 30-Target Mean Cliff RMSE & Overall RMSE
    bar_specs = [
        ("A1_Direct_MLP", "DirectMLP", PALETTE["lightgray"]),
        ("B1_CatBoost", "CatBoost", PALETTE["skyblue"]),
        ("K_RBF_SVM", "RBF-SVM", PALETTE["darkgray"]),
        ("K_Multi_SVR", "MultiSVR", PALETTE["blue"]),
        ("E_Multi_DirectMLP_OptW", "OptW-MLP", PALETTE["purple"]),
        ("E_Multi_CatBoost_OptW", "OptW-Cat", PALETTE["indigo"]),
        ("E_Multi_FiLM", "FiLM(0.5)", PALETTE["amber"]),
        ("E_Multi_FiLM_PriorN", "PriorN-FiLM", PALETTE["teal"]),
        ("E_Multi_FiLM_OptW", "OptW-FiLM", PALETTE["orange"]),
    ]
    agg = step_a.groupby("model")[["cliff_rmse_mean", "rmse_mean"]].mean()
    x = np.arange(len(bar_specs))
    w = 0.38
    cvals = [float(agg.loc[m, "cliff_rmse_mean"]) for m, _, _ in bar_specs]
    ovals = [float(agg.loc[m, "rmse_mean"]) for m, _, _ in bar_specs]
    cols = [c for _, _, c in bar_specs]
    lbls = [l for _, l, _ in bar_specs]

    ax1.bar(x - w / 2, cvals, width=w, color=cols, edgecolor="black", linewidth=0.45, label="Cliff RMSE")
    ax1.bar(x + w / 2, ovals, width=w, color=cols, alpha=0.45, edgecolor="black", linewidth=0.45, hatch="///", label="Overall RMSE")
    for xi, cv in zip(x, cvals):
        ax1.text(xi - w / 2, cv + 0.002, f"{cv:.3f}", ha="center", va="bottom", fontsize=5.2, rotation=65)
    ax1.set_xticks(x)
    ax1.set_xticklabels(lbls, rotation=35, ha="right", fontsize=6.0)
    ax1.set_ylim(0.56, max(cvals) + 0.036)
    ax1.set_ylabel("30-Target Mean RMSE")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.legend(frameon=False, fontsize=6.0, loc="upper right")
    panel_tag(ax1, "A", x=-0.19, y=1.04)

    # Panel B: Learned Bates-Granger w*_FiLM vs n_total
    rho_w, p_w = stats.spearmanr(wide["n_total"], wide["w_film"])
    for assay, col, mk, lbl_a in [("Ki", PALETTE["blue"], "o", r"$\mathrm{K_i}$"), ("EC50", PALETTE["orange"], "^", r"$\mathrm{EC_{50}}$")]:
        sub = wide[wide["assay_type"] == assay]
        ax2.scatter(sub["n_total"], sub["w_film"], color=col, marker=mk, s=26, edgecolor="black", linewidth=0.45, label=rf"{lbl_a} ($N={len(sub)}$)")
    ax2.axvline(1000, color="#555555", linestyle="--", linewidth=0.8, label=r"$n = 1000$")
    ng = np.linspace(float(wide["n_total"].min()), float(wide["n_total"].max()), 100)
    ntr = ng * 0.8
    ax2.plot(ng, 0.5 * ntr / (ntr + 1000.0), "k:", linewidth=1.1, label=r"$\mathrm{PriorN}(n)$")
    ax2.text(0.97, 0.96, rf"Spearman $\rho={rho_w:+.3f}$" + "\n" + rf"$p={p_w:.3f}$", transform=ax2.transAxes, ha="right", va="top", fontsize=6.0, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#CCCCCC", lw=0.5))
    ax2.set_xlabel(r"Target Dataset Size ($n_{\mathrm{total}}$)")
    ax2.set_ylabel(r"Learned Nested-OOF Weight ($w^*$)")
    ax2.set_ylim(-0.02, 0.65)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(frameon=False, fontsize=5.8, loc="lower right")
    panel_tag(ax2, "B", x=-0.19, y=1.04)

    # Panel C: |Delta y| Potency-Jump Curves + Zoomed Delta vs RBF-SVM Inset
    xb = np.arange(len(bin_lbls))
    curves = [
        ("K_RBF_SVM", "RBF-SVM", PALETTE["darkgray"], "--o"),
        ("K_Multi_SVR", "MultiSVR", PALETTE["blue"], "-.s"),
        ("E_Multi_DirectMLP_OptW", "OptW-MLP", PALETTE["purple"], ":^"),
        ("E_Multi_FiLM_OptW", "OptW-FiLM", PALETTE["orange"], "-D"),
    ]
    for m_key, lbl, col, fmt in curves:
        ax3.plot(xb, bin_rmses[m_key], fmt, color=col, linewidth=1.4, markersize=3.8, label=lbl)
    ax3.set_xticks(xb)
    ax3.set_xticklabels([f"{l}\n($n={n}$)" for l, n in zip(bin_lbls, bin_cnts)], fontsize=5.8)
    ax3.set_xlim(-0.32, 3.35)
    ax3.set_ylim(0.34, 1.28)
    ax3.set_xlabel(r"$|\Delta y|$ to Analog ($s_{\max} \geq 0.70$)", fontsize=7.4)
    ax3.set_ylabel("Pooled Analog RMSE")
    ax3.grid(True, linestyle="--", alpha=0.3)
    ax3.legend(frameon=False, fontsize=5.7, loc="upper left")

    # Zoomed Delta RMSE vs RBF-SVM inset in lower-right whitespace
    ax_in = ax3.inset_axes([0.55, 0.08, 0.43, 0.35])
    ref = np.asarray(bin_rmses["K_RBF_SVM"])
    for m_key, _, col, fmt in curves[1:]:
        diff = np.asarray(bin_rmses[m_key]) - ref
        ax_in.plot(xb, diff, fmt, color=col, linewidth=1.1, markersize=3.0)
    ax_in.axhline(0.0, color=PALETTE["darkgray"], linestyle="--", linewidth=0.7)
    ax_in.set_xticks(xb)
    ax_in.set_xticklabels(["<0.5", "0.5–1", "1–1.5", r"$\geq$1.5"], fontsize=4.6)
    ax_in.tick_params(axis="y", labelsize=4.6, pad=1)
    ax_in.text(0.96, 0.94, r"$\Delta$RMSE vs SVM", transform=ax_in.transAxes, ha="right", va="top", fontsize=4.8, fontweight="bold")
    ax_in.set_facecolor("#FAFAFA")
    ax_in.grid(True, linestyle=":", alpha=0.4)
    panel_tag(ax3, "C", x=-0.19, y=1.04)

    save_jcim_fig(fig, "fig2_main30", also_tiff=False)


if __name__ == "__main__":
    main()

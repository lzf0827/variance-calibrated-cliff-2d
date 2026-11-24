"""Generate Figure 4: 4-Panel Mechanistic Dissection of Ligand-Only Failure Modes (10 Wave-1 Targets)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jcim_plot_style import PALETTE, REFINELOGS_DIR, apply_jcim_style, panel_tag, save_jcim_fig


def main() -> None:
    apply_jcim_style(base_font=7.5)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.95), gridspec_kw={"hspace": 0.46, "wspace": 0.34})
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # Panel A: 6-Channel LFER Attribution in L2_eSpan_LFER_Delta
    w1_jsons = sorted((REFINELOGS_DIR / "benchmark30_results" / "per_target").glob("*.json"))
    js = [json.loads(p.read_text(encoding="utf-8")) for p in w1_jsons]
    ch_order = [
        ("Crippen_Delta_MR", "2D Crippen MR", PALETTE["teal"]),
        ("Crippen_Delta_LogP", "2D Crippen LogP", PALETTE["teal"]),
        ("Delta_TPSA", "2D Ertl TPSA", PALETTE["skyblue"]),
        ("AIMNet2_Delta_q", r"AIMNet2 $\Delta q$", PALETTE["lightgray"]),
        ("Chiral_Stereo", "Chiral / Stereo", PALETTE["amber"]),
        ("3D_eSpan_Geom", r"3D Geom + $e_{\mathrm{span}}$", PALETTE["crimson"]),
    ]
    ch_vals = [float(np.mean([j["lfer_attribution"][k]["delta_cliff_rmse_mean"] for j in js])) for k, _, _ in ch_order]
    ch_lbls = [lbl for _, lbl, _ in ch_order]
    ch_cols = [c for _, _, c in ch_order]
    ya = np.arange(len(ch_order))

    ax_a.barh(ya, ch_vals, height=0.58, color=ch_cols, edgecolor="black", linewidth=0.5)
    ax_a.axvline(0.0, color="#111111", linewidth=0.8)
    for yi, val in zip(ya, ch_vals):
        off = 0.00010 if val >= 0 else -0.00010
        ha = "left" if val >= 0 else "right"
        ax_a.text(val + off, yi, f"{val:+.5f}", va="center", ha=ha, fontsize=5.8)
    ax_a.set_yticks(ya)
    ax_a.set_yticklabels(ch_lbls, fontsize=6.5)
    ax_a.invert_yaxis()
    ax_a.set_xlim(-0.0029, 0.0025)
    ax_a.set_xlabel(r"Channel-Zeroing $\Delta$Cliff RMSE on L2 ($>0$ Useful)", fontsize=7.0)
    ax_a.text(0.97, 0.06, "Y_FiLM_Gated3DQM:\n" + r"$g_{\mathrm{3D/QM}} = 0.056$–$0.073$" + "\n" + r"(init $0.0759$; $p=0.922$ NS)", transform=ax_a.transAxes, ha="right", va="bottom", fontsize=5.6, bbox=dict(boxstyle="round,pad=0.25", fc="#FAFAFA", ec="#CCCCCC", lw=0.6))
    ax_a.grid(axis="x", linestyle="--", alpha=0.3)
    panel_tag(ax_a, "A", x=-0.22, y=1.04)

    # Panel B: Failure Mode #2 — Reused-Val Physical Gate (G2_Prev) vs Un-Gated 50/50 Mean (Avg50_L2)
    npzs = sorted((REFINELOGS_DIR / "benchmark30_results" / "per_target").glob("*_preds.npz"))
    tgts_b, diffs_b = [], []
    g2_all, a50_all = [], []
    for p in npzs:
        t = p.stem.replace("_preds", "")
        z = np.load(p)
        cm = z["c_te"] == 1
        yt = z["y_te"][cm]
        g2_t = float(np.mean([np.sqrt(np.mean((yt - z["G2_RBFSVM_PhysGate_L2_seeds"][s][cm]) ** 2)) for s in range(5)]))
        a50_t = float(np.mean([np.sqrt(np.mean((yt - (0.5 * z["B1c_RBF_SVM_seeds"][s][cm] + 0.5 * z["L2_eSpan_LFER_Delta_seeds"][s][cm])) ** 2)) for s in range(5)]))
        tgts_b.append(t.replace("CHEMBL", "C"))
        diffs_b.append(g2_t - a50_t)
        g2_all.append(g2_t)
        a50_all.append(a50_t)
    order_b = np.argsort(diffs_b)
    tgts_b = [tgts_b[i] for i in order_b]
    diffs_b = [diffs_b[i] for i in order_b]
    _, p_gate = stats.wilcoxon(g2_all, a50_all)
    yb = np.arange(len(tgts_b))
    cols_b = [PALETTE["teal"] if d < 0 else PALETTE["crimson"] for d in diffs_b]
    ax_b.barh(yb, diffs_b, height=0.62, color=cols_b, edgecolor="black", linewidth=0.45)
    ax_b.axvline(0.0, color="#111111", linewidth=0.8)
    ax_b.set_yticks(yb)
    ax_b.set_yticklabels(tgts_b, fontsize=6.0)
    ax_b.set_xlabel(r"$\Delta$Cliff RMSE ($\mathrm{G2\_Prev} - \mathrm{Avg50\_L2}$)")
    ax_b.text(0.97, 0.07, r"$2\mathrm{W} / 8\mathrm{L}$ ($p = 1.95 \times 10^{-2}$)" + f"\nG2 = {np.mean(g2_all):.4f} vs. Avg50 = {np.mean(a50_all):.4f}", transform=ax_b.transAxes, ha="right", va="bottom", fontsize=5.7, bbox=dict(boxstyle="round,pad=0.25", fc="#FDEDEC", ec=PALETTE["crimson"], lw=0.6))
    ax_b.grid(axis="x", linestyle="--", alpha=0.3)
    panel_tag(ax_b, "B", x=-0.18, y=1.04)

    # Panel C: Failure Mode #3 — OOF Additive Kernel-Residual Stacking vs y-Space Blending
    pa = pd.read_csv(REFINELOGS_DIR / "path_a_modular_results" / "path_a_summary.csv")
    pa_agg = pa.groupby("model")[["cliff_rmse_mean", "rmse_mean"]].mean()
    c_specs = [
        ("E_SVM_A2Delta", "E_SVM_A2\n(y-Blend)", PALETTE["teal"]),
        ("B1c_RBF_SVM", "RBF-SVM\n(Base)", PALETTE["darkgray"]),
        ("E_SVM_DirectMLP", "SVM+MLP\n(y-Blend)", PALETTE["purple"]),
        ("KR_FiLM_Gated3DQM", "KR_3DQM\n(OOF)", PALETTE["amber"]),
        ("KR_FiLM_2DLFER", "KR_FiLM\n(OOF)", PALETTE["orange"]),
        ("KR_2D_Delta", "KR_2D\n(OOF)", PALETTE["crimson"]),
    ]
    xc = np.arange(len(c_specs))
    w = 0.36
    cv_c = [float(pa_agg.loc[m, "cliff_rmse_mean"]) for m, _, _ in c_specs]
    ov_c = [float(pa_agg.loc[m, "rmse_mean"]) for m, _, _ in c_specs]
    col_c = [c for _, _, c in c_specs]
    ax_c.bar(xc - w / 2, cv_c, width=w, color=col_c, edgecolor="black", linewidth=0.5, label="Cliff RMSE")
    ax_c.bar(xc + w / 2, ov_c, width=w, color=col_c, alpha=0.45, edgecolor="black", linewidth=0.5, hatch="///", label="Overall RMSE")
    for xi, cv in zip(xc, cv_c):
        ax_c.text(xi - w / 2, cv + 0.002, f"{cv:.3f}", ha="center", va="bottom", fontsize=5.4, rotation=55)
    ax_c.set_xticks(xc)
    ax_c.set_xticklabels([l.replace("\n", " ") for _, l, _ in c_specs], rotation=26, ha="right", fontsize=5.7)
    ax_c.set_ylim(0.58, 0.878)
    ax_c.set_ylabel("10-Target Mean RMSE")
    ax_c.text(0.03, 0.96, "KR vs. E_SVM_A2:\n" + r"$2\mathrm{W} / 8\mathrm{L}$, $p = 9.77 \times 10^{-3}$", transform=ax_c.transAxes, va="top", fontsize=5.6, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#CCCCCC", lw=0.6))
    ax_c.grid(axis="y", linestyle="--", alpha=0.3)
    ax_c.legend(frameon=False, fontsize=5.6, loc="upper right")
    panel_tag(ax_c, "C", x=-0.16, y=1.04)

    # Panel D: Failure Mode #4 — Smooth-SAR vs Activity-Cliff Pareto Trade-Off (GINE + CrossAttn)
    o13 = pd.read_csv(REFINELOGS_DIR / "opt13_modular_results" / "opt13_summary.csv")
    o13_agg = o13.groupby("model")[["rmse_mean", "cliff_rmse_mean"]].mean()
    d_specs = [
        ("K_RBF_SVM", "K_RBF_SVM", PALETTE["darkgray"], "o", (-48, 4)),
        ("K_Multi_SVR", "K_Multi_SVR", PALETTE["blue"], "s", (-52, 4)),
        ("E_SVM_A2Delta", "E_SVM_A2", PALETTE["teal"], "^", (6, -3)),
        ("E_Multi_A2Delta", "E_Multi_A2", PALETTE["skyblue"], "v", (-6, 10)),
        ("E_Multi_FiLM", "E_Multi_FiLM (0.5)", PALETTE["orange"], "D", (-24, -12)),
        ("E_Multi_Attn", "E_Multi_Attn", PALETTE["purple"], "P", (6, -3)),
        ("E_Multi_GraphAttn", "E_Multi_GraphAttn", PALETTE["crimson"], "*", (-4, 9)),
    ]
    for m_key, lbl, col, mk, (ox, oy) in d_specs:
        xv = float(o13_agg.loc[m_key, "rmse_mean"])
        yv = float(o13_agg.loc[m_key, "cliff_rmse_mean"])
        ms = 55 if mk == "*" else 34
        ax_d.scatter(xv, yv, color=col, marker=mk, s=ms, edgecolor="black", linewidth=0.5, zorder=4)
        ax_d.annotate(lbl, xy=(xv, yv), xytext=(ox, oy), textcoords="offset points", fontsize=5.7, fontweight="bold" if "FiLM" in lbl or "GraphAttn" in lbl else "normal")

    # Arrow highlighting Cliff degradation vs Overall gain
    ax_d.annotate(
        "",
        xy=(float(o13_agg.loc["E_Multi_GraphAttn", "rmse_mean"]), float(o13_agg.loc["E_Multi_GraphAttn", "cliff_rmse_mean"])),
        xytext=(float(o13_agg.loc["E_Multi_FiLM", "rmse_mean"]), float(o13_agg.loc["E_Multi_FiLM", "cliff_rmse_mean"])),
        arrowprops=dict(arrowstyle="->", lw=1.0, ls="--", color=PALETTE["crimson"]),
    )
    ax_d.text(0.97, 0.44, "GraphAttn vs. A2 Blend:\n" + r"Overall $-0.0025$" + "\n" + r"Cliff $+0.0052$ ($2\mathrm{W}/8\mathrm{L}$, $p=0.0195$)", transform=ax_d.transAxes, ha="right", va="center", fontsize=5.4, bbox=dict(boxstyle="round,pad=0.22", fc="#FDEDEC", ec=PALETTE["crimson"], lw=0.6))
    ax_d.set_xlabel("Overall RMSE (Smooth SAR, Lower = Better)")
    ax_d.set_ylabel("Activity Cliff RMSE (Lower = Better)")
    ax_d.set_xlim(0.6262, 0.6402)
    ax_d.set_ylim(0.7330, 0.7505)
    ax_d.grid(True, linestyle="--", alpha=0.3)
    panel_tag(ax_d, "D", x=-0.16, y=1.04)

    save_jcim_fig(fig, "fig4_diagnostics", also_tiff=False)


if __name__ == "__main__":
    main()

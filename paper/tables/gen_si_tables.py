"""Programmatically generate ACS JCIM Supporting Information Tables S1-S5 from verified data."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
REFINELOGS_DIR = ROOT_DIR / "refine-logs"
OUT_DIR = Path(__file__).resolve().parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from physchem_delta.data import load_target_dataset
from physchem_delta.eval_plan_a import _lfer_atom_matrix
from physchem_delta.featurize import compute_local_pair_deltas, precompute_mol_env_counters
from physchem_delta.models import compute_tanimoto_matrix
from physchem_delta.run_jcim_30targets import _cross_target_stats, extract_or_load_2d_features


def _esc(s: str) -> str:
    return s.replace("_", r"\_")


def _fmt_p(p: float) -> str:
    if p < 1e-2:
        base, exp = f"{p:.2e}".split("e")
        return f"${base} \\times 10^{{{int(exp)}}}$"
    return f"{p:.4f}"


def _build_si_table1() -> str:
    wide = pd.read_csv(REFINELOGS_DIR / "jcim30_step_a_results" / "step_a_wide.csv")
    wide = wide.sort_values("n_total", ascending=False)
    lines = [
        r"\begin{longtable}{lcccccc}",
        r"\caption{Complete Dataset Characteristics and Learned Split-Matched Calibration Weights Across All 30 Official MoleculeACE Targets (Sorted by $n_{\mathrm{total}}$).} \label{tab:si_datasets} \\",
        r"\toprule",
        r"Target & Assay & $n_{\mathrm{total}}$ & $n_{\mathrm{train}}$ & $n_{\mathrm{test}}$ ($n_{\mathrm{cliff}}$) & $w^*_{\mathrm{FiLM}}$ & $w_{\mathrm{prior}}$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Target & Assay & $n_{\mathrm{total}}$ & $n_{\mathrm{train}}$ & $n_{\mathrm{test}}$ ($n_{\mathrm{cliff}}$) & $w^*_{\mathrm{FiLM}}$ & $w_{\mathrm{prior}}$ \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for _, r in wide.iterrows():
        assay_tex = r"$\mathrm{K_i}$" if r["assay_type"] == "Ki" else r"$\mathrm{EC_{50}}$"
        lines.append(
            f"\\texttt{{{_esc(str(r['target']))}}} & {assay_tex} & {int(r['n_total'])} & "
            f"{int(r['n_train'])} & {int(r['n_test'])} ({int(r['n_test_cliff'])}) & "
            f"{float(r['w_film']):.3f} & {float(r['w_prior']):.3f} \\\\"
        )
    lines.append(r"\end{longtable}")
    return "\n".join(lines) + "\n"


def _build_si_table2() -> str:
    step_a = pd.read_csv(REFINELOGS_DIR / "jcim30_step_a_results" / "step_a_summary.csv")
    wide = pd.read_csv(REFINELOGS_DIR / "jcim30_step_a_results" / "step_a_wide.csv")
    agg = step_a.groupby("model")[["rmse_mean", "cliff_rmse_mean", "non_cliff_rmse_mean", "cliff_mae_mean", "spearman_mean"]].mean()
    agg = agg.sort_values("cliff_rmse_mean", ascending=True)

    b_ov = float(agg["rmse_mean"].min())
    b_cl = float(agg["cliff_rmse_mean"].min())
    b_nc = float(agg["non_cliff_rmse_mean"].min())
    b_cm = float(agg["cliff_mae_mean"].min())
    b_sp = float(agg["spearman_mean"].max())

    def _b(v: float, target: float) -> str:
        s = f"{v:.4f}"
        return f"\\textbf{{{s}}}" if abs(v - target) < 1e-6 else s

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Complete 17-Model Step-A Benchmark Leaderboard Across 30 MoleculeACE Targets ($5$ Seeds) and Zero-Parameter \texttt{PriorN\_FiLM} Paired Wilcoxon Tests.}",
        r"\label{tab:si_main30_full}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Model & Overall RMSE $\downarrow$ & Cliff RMSE $\downarrow$ & Non-Cliff RMSE $\downarrow$ & Cliff MAE $\downarrow$ & Spearman $\uparrow$ \\",
        r"\midrule",
    ]
    for m, r in agg.iterrows():
        lines.append(
            f"\\texttt{{{_esc(str(m))}}} & {_b(float(r['rmse_mean']), b_ov)} & {_b(float(r['cliff_rmse_mean']), b_cl)} & "
            f"{_b(float(r['non_cliff_rmse_mean']), b_nc)} & {_b(float(r['cliff_mae_mean']), b_cm)} & {_b(float(r['spearman_mean']), b_sp)} \\\\"
        )
    lines.extend([
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{Cross-Target Paired Wilcoxon Signed-Rank Tests for \texttt{E\_Multi\_FiLM\_PriorN} ($N=30$)}} \\",
        r"Comparison (\texttt{PriorN\_FiLM} vs. Ref) & Metric & Win / Loss & Mean $\Delta$ & 95\% Bootstrap CI & Two-Sided $p$ \\",
        r"\midrule",
    ])
    for lbl, mtr, ca, cb in [
        (r"vs. \texttt{K\_RBF\_SVM}", "Cliff", "E_Multi_FiLM_PriorN_cliff", "K_RBF_SVM_cliff"),
        (r"vs. \texttt{K\_Multi\_SVR}", "Cliff", "E_Multi_FiLM_PriorN_cliff", "K_Multi_SVR_cliff"),
        (r"vs. \texttt{E\_Multi\_DirectMLP\_PriorN}", "Cliff", "E_Multi_FiLM_PriorN_cliff", "E_Multi_DirectMLP_PriorN_cliff"),
        (r"vs. \texttt{E\_Multi\_CatBoost\_PriorN}", "Cliff", "E_Multi_FiLM_PriorN_cliff", "E_Multi_CatBoost_PriorN_cliff"),
        (r"vs. \texttt{K\_RBF\_SVM}", "Overall", "E_Multi_FiLM_PriorN_rmse", "K_RBF_SVM_rmse"),
        (r"vs. \texttt{K\_Multi\_SVR}", "Overall", "E_Multi_FiLM_PriorN_rmse", "K_Multi_SVR_rmse"),
        (r"vs. \texttt{E\_Multi\_DirectMLP\_PriorN}", "Overall", "E_Multi_FiLM_PriorN_rmse", "E_Multi_DirectMLP_PriorN_rmse"),
    ]:
        st = _cross_target_stats(wide, ca, cb)
        lines.append(
            f"{lbl} & {mtr} & {st['wins']}/{st['losses']} & {st['mean_diff']:+.5f} & "
            f"[{st['ci95_lo']:+.5f}, {st['ci95_hi']:+.5f}] & {_fmt_p(float(st['p_value']))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def _build_si_table3() -> str:
    scaf = pd.read_csv(REFINELOGS_DIR / "scaffold_robustness_results" / "scaffold_summary.csv")
    wide = pd.read_csv(REFINELOGS_DIR / "scaffold_robustness_results" / "scaffold_wide.csv")
    agg = scaf.groupby("model")[["rmse_mean", "mae_mean", "cliff_rmse_mean", "non_cliff_rmse_mean", "spearman_mean"]].mean().sort_values("rmse_mean")

    # Compute pooled OOD s_max bins directly from per_target npz
    npz_dir = REFINELOGS_DIR / "scaffold_robustness_results" / "per_target"
    bin_models = ("K_RBF_SVM", "K_Multi_SVR", "Y_FiLM_2DLFER", "E_Multi_DirectMLP_OptW", "E_Multi_FiLM", "E_Multi_FiLM_OptW", "E_Multi_FiLM_GroupOptW")
    ys, smaxes, cliffs = [], [], []
    pmap = {m: [] for m in bin_models}
    for t in wide["target"].tolist():
        z = np.load(npz_dir / f"{t}_scaffold_preds.npz")
        ys.append(z["y_pooled"])
        smaxes.append(z["s_max_pooled"])
        cliffs.append(z["c_pooled"])
        for m in bin_models:
            pmap[m].append(z[f"{m}_pooled"])
    yv = np.concatenate(ys)
    sv = np.concatenate(smaxes)
    cv = np.concatenate(cliffs)
    pv = {m: np.concatenate(pmap[m]) for m in bin_models}

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Complete 11-Model Strict Bemis--Murcko Scaffold Holdout Leaderboard, Per-Target Weight Shrinkage, and Similarity-Stratified Pooled OOD RMSE ($10$ Targets $\times$ $2$ Splits, $6{,}907$ Predictions).}",
        r"\label{tab:si_scaffold_full}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Model & Overall RMSE & MAE & Orig-Cliff RMSE & Non-Cliff RMSE & Spearman \\",
        r"\midrule",
    ]
    for m, r in agg.iterrows():
        lines.append(
            f"\\texttt{{{_esc(str(m))}}} & {float(r['rmse_mean']):.4f} & {float(r['mae_mean']):.4f} & "
            f"{float(r['cliff_rmse_mean']):.4f} & {float(r['non_cliff_rmse_mean']):.4f} & {float(r['spearman_mean']):.4f} \\\\"
        )

    lines.extend([
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{Pooled Test Predictions Stratified by Nearest-Train ECFP4 Similarity ($s_{\max}$)}} \\",
        r"Similarity Stratum ($n_{\mathrm{mol}}$) & \texttt{K\_RBF\_SVM} & \texttt{K\_Multi\_SVR} & \texttt{Y\_FiLM} & \texttt{OptW\_FiLM} & \texttt{GrpOptW\_FiLM} \\",
        r"\midrule",
    ])
    for lbl, mask in [
        (r"$s_{\max} < 0.50$ (Far OOD)", sv < 0.50),
        (r"$0.50 \le s_{\max} < 0.70$ (Scaffold Hop)", (sv >= 0.50) & (sv < 0.70)),
        (r"$s_{\max} \ge 0.70$ (Near Bioisostere)", sv >= 0.70),
        (r"Original \texttt{cliff\_mol == 1}", cv == 1),
    ]:
        vals = [float(np.sqrt(np.mean((yv[mask] - pv[m][mask]) ** 2))) for m in ("K_RBF_SVM", "K_Multi_SVR", "Y_FiLM_2DLFER", "E_Multi_FiLM_OptW", "E_Multi_FiLM_GroupOptW")]
        best = min(vals)
        cells = [f"\\textbf{{{v:.4f}}}" if abs(v - best) < 1e-6 else f"{v:.4f}" for v in vals]
        lines.append(f"{lbl} ({int(np.sum(mask))}) & " + " & ".join(cells) + r" \\")

    lines.extend([
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{Per-Target Overall RMSE and Learned Weights (Stratified $w^*_{\mathrm{Strat}}$ vs.\ Scaffold \texttt{GroupKFold} $w^*_{\mathrm{Grp}}$)}} \\",
        r"Target ($n$, Scaffolds) & $s_{\max}$ (Orig $\rightarrow$ Scaf) & $w^*_{\mathrm{Strat}} \rightarrow w^*_{\mathrm{Grp}}$ & \texttt{K\_Multi\_SVR} & \texttt{OptW\_FiLM} & \texttt{GrpOptW\_FiLM} \\",
        r"\midrule",
    ])
    for _, r in wide.iterrows():
        lines.append(
            f"\\texttt{{{_esc(str(r['target']))}}} ({int(r['n_total'])}, {int(r['n_total_scaffolds'])}) & "
            f"{float(r['orig_mean_smax']):.3f} $\\rightarrow$ {float(r['scaf_mean_smax']):.3f} & "
            f"{float(r['w_strat_film']):.3f} $\\rightarrow$ {float(r['w_group_film']):.3f} & "
            f"{float(r['K_Multi_SVR_rmse']):.4f} & {float(r['E_Multi_FiLM_OptW_rmse']):.4f} & {float(r['E_Multi_FiLM_GroupOptW_rmse']):.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def _build_si_table4() -> str:
    w1 = pd.read_csv(REFINELOGS_DIR / "benchmark30_results" / "wave1_summary.csv").pivot(index="target", columns="model", values="cliff_rmse_mean")
    pa = pd.read_csv(REFINELOGS_DIR / "path_a_modular_results" / "path_a_summary.csv").pivot(index="target", columns="model", values="cliff_rmse_mean")
    o13 = pd.read_csv(REFINELOGS_DIR / "opt13_modular_results" / "opt13_summary.csv").pivot(index="target", columns="model", values="cliff_rmse_mean")
    pv2 = pd.read_csv(REFINELOGS_DIR / "plan_a_v2_summary.csv").pivot(index="target", columns="model", values="cliff_rmse_mean")

    npzs = sorted((REFINELOGS_DIR / "benchmark30_results" / "per_target").glob("*_preds.npz"))
    a50_map: dict[str, float] = {}
    for p in npzs:
        t = p.stem.replace("_preds", "")
        z = np.load(p)
        cm = z["c_te"] == 1
        yt = z["y_te"][cm]
        a50_map[t] = float(np.mean([np.sqrt(np.mean((yt - (0.5 * z["B1c_RBF_SVM_seeds"][s][cm] + 0.5 * z["L2_eSpan_LFER_Delta_seeds"][s][cm])) ** 2)) for s in range(5)]))

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Per-Target Wave-1 Negative-Diagnostic Cliff RMSE Matrix ($10$ Targets $\times$ $5$ Seeds) and 3-Target Boltzmann Conformer Extension.}",
        r"\label{tab:si_negatives_full}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"Target & \texttt{G0} (2D) & \texttt{G2\_Prev} & \texttt{Avg50\_L2} & \texttt{E\_SVM\_A2} & \texttt{KR\_2D} & \texttt{E\_Multi\_A2} & \texttt{E\_GraphAttn} & \texttt{Y\_Gated3DQM} \\",
        r"\midrule",
    ]
    for tgt in sorted(w1.index.tolist()):
        lines.append(
            f"\\texttt{{{_esc(tgt)}}} & {w1.loc[tgt, 'G0_RBFSVM_Gate_A2']:.4f} & {w1.loc[tgt, 'G2_RBFSVM_PhysGate_L2']:.4f} & "
            f"{a50_map[tgt]:.4f} & {pa.loc[tgt, 'E_SVM_A2Delta']:.4f} & {pa.loc[tgt, 'KR_2D_Delta']:.4f} & "
            f"{o13.loc[tgt, 'E_Multi_A2Delta']:.4f} & {o13.loc[tgt, 'E_Multi_GraphAttn']:.4f} & {o13.loc[tgt, 'Y_FiLM_Gated3DQM']:.4f} \\\\"
        )
    lines.append(
        f"\\textbf{{Wave-1 Mean}} & {w1['G0_RBFSVM_Gate_A2'].mean():.4f} & {w1['G2_RBFSVM_PhysGate_L2'].mean():.4f} & "
        f"{np.mean(list(a50_map.values())):.4f} & {pa['E_SVM_A2Delta'].mean():.4f} & {pa['KR_2D_Delta'].mean():.4f} & "
        f"{o13['E_Multi_A2Delta'].mean():.4f} & {o13['E_Multi_GraphAttn'].mean():.4f} & {o13['Y_FiLM_Gated3DQM'].mean():.4f} \\\\"
    )

    lines.extend([
        r"\midrule",
        r"\multicolumn{9}{l}{\textit{3-Target MVP 5-Conformer $300\,\mathrm{K}$ Boltzmann Probe (\texttt{G\_S2\_Boltz\_SkipLFER} vs.\ Single-Conformer Controls)}} \\",
        r"MVP Target & \texttt{B1c\_SVM} & \texttt{G0} (2D) & \texttt{G2\_Prev} & \texttt{G\_S1\_Single} & \texttt{G\_S2\_Boltz} & $\Delta$(\texttt{G\_S2}-\texttt{G2}) & Pooled $p$ & Outcome \\",
        r"\midrule",
    ])
    pvals_mvp = {"CHEMBL204_Ki": "0.7652", "CHEMBL2147_Ki": "0.0228", "CHEMBL237_Ki": "0.6859"}
    for tgt in ["CHEMBL204_Ki", "CHEMBL2147_Ki", "CHEMBL237_Ki"]:
        d_s2 = float(pv2.loc[tgt, "G_S2_Boltz_SkipLFER"] - pv2.loc[tgt, "G2_Prev_SingleConcat"])
        outcome = "Win" if d_s2 < -1e-9 else "Loss"
        lines.append(
            f"\\texttt{{{_esc(tgt)}}} & {pv2.loc[tgt, 'B1c_RBF_SVM']:.4f} & {pv2.loc[tgt, 'G0_RBFSVM_Gate_A2']:.4f} & "
            f"{pv2.loc[tgt, 'G2_Prev_SingleConcat']:.4f} & {pv2.loc[tgt, 'G_S1_Single_SkipLFER']:.4f} & "
            f"{pv2.loc[tgt, 'G_S2_Boltz_SkipLFER']:.4f} & {d_s2:+.4f} & {pvals_mvp[tgt]} & {outcome} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def _select_case_details(target: str, role: str) -> dict[str, object]:
    ds = load_target_dataset(target)
    feats = extract_or_load_2d_features(target, ds.smiles)
    z = np.load(REFINELOGS_DIR / "jcim30_step_a_results" / "per_target" / f"{target}_optw_preds.npz")
    sim = compute_tanimoto_matrix(feats.ecfp_chiral[ds.test_idx], feats.ecfp_chiral[ds.train_idx])
    bp = np.argmax(sim, axis=1)
    s_max = sim[np.arange(ds.test_idx.size), bp]
    nearest_tr = ds.train_idx[bp]
    y_te = ds.y[ds.test_idx]
    y_anc = ds.y[nearest_tr]
    dy_true = y_te - y_anc

    pr_rbf = z["K_RBF_SVM_mean"].astype(np.float64)
    pr_mul = z["K_Multi_SVR_mean"].astype(np.float64)
    pr_mlp = z["E_Multi_DirectMLP_OptW_mean"].astype(np.float64)
    pr_film = z["E_Multi_FiLM_OptW_mean"].astype(np.float64)

    err_film = np.abs(y_te - pr_film)
    err_best = np.minimum.reduce([np.abs(y_te - pr_rbf), np.abs(y_te - pr_mul), np.abs(y_te - pr_mlp)])
    elig = (s_max >= 0.75) & (np.abs(dy_true) >= 1.2)
    if not np.any(elig):
        elig = (s_max >= 0.70) & (np.abs(dy_true) >= 1.0)
    cand = np.where(elig)[0]
    margin = err_best - err_film
    pos = int(cand[np.argmax(margin[cand])]) if role == "WIN" else int(cand[np.argmin(margin[cand])])

    lfer_atoms = _lfer_atom_matrix(feats)
    env_c = precompute_mol_env_counters(feats.atom_ptr, feats.atom_env_hash)
    d_mat = compute_local_pair_deltas(
        np.array([ds.test_idx[pos]], dtype=np.int64),
        np.array([nearest_tr[pos]], dtype=np.int64),
        feats.atom_ptr,
        feats.atom_env_hash,
        lfer_atoms,
        env_counters=env_c,
    )[0]
    return {
        "target": target,
        "role": role,
        "s_max": float(s_max[pos]),
        "y_anc": float(y_anc[pos]),
        "y_true": float(y_te[pos]),
        "dy": float(dy_true[pos]),
        "d_logp": float(d_mat[1]),
        "d_mr": float(d_mat[2]),
        "d_tpsa": float(d_mat[3]),
        "K_RBF_SVM": float(pr_rbf[pos]),
        "K_Multi_SVR": float(pr_mul[pos]),
        "OptW_MLP": float(pr_mlp[pos]),
        "OptW_FiLM": float(pr_film[pos]),
        "test_smiles": ds.smiles[int(ds.test_idx[pos])],
        "anchor_smiles": ds.smiles[int(nearest_tr[pos])],
    }


def _wrap_smiles_tex(smi: str) -> str:
    """Escape SMILES safely for LaTeX and insert invisible line-break points every ~24 chars."""
    out: list[str] = []
    since_brk = 0
    in_brack = 0
    esc_map = {"#": r"\#", "%": r"\%", "&": r"\&", "$": r"\$", "_": r"\_", "{": r"\{", "}": r"\}", "^": r"\^{}", "~": r"\~{}"}
    for ch in smi:
        if ch == "[":
            in_brack += 1
        elif ch == "]":
            in_brack = max(0, in_brack - 1)
        out.append(esc_map.get(ch, ch))
        since_brk += 1
        if since_brk >= 22 and in_brack == 0 and ch in (")", "=", "-", "c", "C", "N", "O"):
            out.append(r"\allowbreak{}")
            since_brk = 0
    return "".join(out)


def _build_si_table5() -> str:
    specs = [
        ("CHEMBL244_Ki", "WIN (Protease)"),
        ("CHEMBL4792_Ki", "WIN (Kinase)"),
        ("CHEMBL237_Ki", "WIN (GPCR)"),
        ("CHEMBL3979_EC50", "WIN (EC50)"),
        ("CHEMBL231_Ki", "FAIL ($n=973$)"),
        ("CHEMBL4203_Ki", "FAIL ($n=731$)"),
    ]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Verified Representative Medicinal-Chemistry Pairs (Four WIN + Two Small-$n$ Conservatism Cases) with Unmatched-Atom RDKit 2D-LFER Deltas and Canonical SMILES.}",
        r"\label{tab:si_cases}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r"Target & Role & $s_{\max}$ & $y_{\mathrm{anc}} \rightarrow y_{\mathrm{true}}$ ($\Delta y$) & $\Delta\mathrm{LogP}/\Delta\mathrm{MR}/\Delta\mathrm{TPSA}$ & \texttt{RBF\_SVM} & \texttt{MultiSVR} & \texttt{OptW\_MLP} & \texttt{OptW\_FiLM} \\",
        r"\midrule",
    ]
    cases = []
    for tgt, label in specs:
        role = "WIN" if "WIN" in label else "FAIL"
        c = _select_case_details(tgt, role)
        c["label"] = label
        cases.append(c)
        preds = [c["K_RBF_SVM"], c["K_Multi_SVR"], c["OptW_MLP"], c["OptW_FiLM"]]
        errs = [abs(c["y_true"] - p) for p in preds]
        best_err = min(errs)
        p_cells = [f"\\textbf{{{p:.3f}}}" if abs(e - best_err) < 1e-9 else f"{p:.3f}" for p, e in zip(preds, errs)]
        lines.append(
            f"\\texttt{{{_esc(tgt)}}} & {label} & {c['s_max']:.3f} & "
            f"{c['y_anc']:.2f} $\\rightarrow$ {c['y_true']:.2f} ({c['dy']:+.2f}) & "
            f"{c['d_logp']:+.2f} / {c['d_mr']:+.1f} / {c['d_tpsa']:+.1f} & "
            + " & ".join(p_cells) + r" \\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        "",
        r"\vspace{2pt}\raggedright\footnotesize Note: $\Delta y$ values are computed from unrounded experimental potencies prior to 2-decimal display.",
        r"\end{table}",
        "",
    ])

    lines.extend([
        r"\subsection*{Verified Case SMILES Strings}",
        r"\begin{description}\scriptsize",
    ])
    for c in cases:
        lines.append(
            f"  \\item[\\texttt{{{_esc(str(c['target']))}}} ({c['label']})] "
            f"Test: \\texttt{{{_wrap_smiles_tex(str(c['test_smiles']))}}} \\\\ "
            f"Anchor: \\texttt{{{_wrap_smiles_tex(str(c['anchor_smiles']))}}}"
        )
    lines.append(r"\end{description}")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = [
        ("si_table1_datasets.tex", _build_si_table1()),
        ("si_table2_main30_full.tex", _build_si_table2()),
        ("si_table3_scaffold_full.tex", _build_si_table3()),
        ("si_table4_negatives_full.tex", _build_si_table4()),
        ("si_table5_cases.tex", _build_si_table5()),
    ]
    for fname, txt in specs:
        p = OUT_DIR / fname
        p.write_text(txt, encoding="utf-8")
        print(f"Saved: {p.name}", flush=True)


if __name__ == "__main__":
    main()

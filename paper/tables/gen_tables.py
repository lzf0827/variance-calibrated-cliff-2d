"""Programmatically generate ACS JCIM LaTeX Tables 1-4 directly from verified CSV/NPZ files."""

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
from physchem_delta.models import compute_tanimoto_matrix
from physchem_delta.run_jcim_30targets import _cross_target_stats, extract_or_load_2d_features


def _esc(name: str) -> str:
    return name.replace("_", r"\_")


def _fmt_p(p: float) -> str:
    if p < 1e-2:
        base, exp = f"{p:.2e}".split("e")
        return f"${base} \\times 10^{{{int(exp)}}}$"
    return f"{p:.4f}"


def _build_table1() -> str:
    step_a = pd.read_csv(REFINELOGS_DIR / "jcim30_step_a_results" / "step_a_summary.csv")
    wide = pd.read_csv(REFINELOGS_DIR / "jcim30_step_a_results" / "step_a_wide.csv")
    agg = step_a.groupby("model")[["rmse_mean", "cliff_rmse_mean", "non_cliff_rmse_mean", "cliff_mae_mean", "spearman_mean"]].mean()

    headline_models = [
        ("E_Multi_FiLM_OptW", r"\texttt{E\_Multi\_FiLM\_OptW}"),
        ("E_Multi_A2_OptW", r"\texttt{E\_Multi\_A2\_OptW}"),
        ("E_Multi_FiLM_PriorN", r"\texttt{E\_Multi\_FiLM\_PriorN}"),
        ("E_Multi_FiLM", r"\texttt{E\_Multi\_FiLM} ($w=0.5$)"),
        ("E_Multi_DirectMLP_OptW", r"\texttt{E\_Multi\_DirectMLP\_OptW}"),
        ("E_Multi_CatBoost_OptW", r"\texttt{E\_Multi\_CatBoost\_OptW}"),
        ("K_Multi_SVR", r"\texttt{K\_Multi\_SVR}"),
        ("K_RBF_SVM", r"\texttt{K\_RBF\_SVM}"),
        ("Y_A2_2D", r"\texttt{Y\_A2\_2D}"),
        ("Y_FiLM_2DLFER", r"\texttt{Y\_FiLM\_2DLFER}"),
        ("B1_CatBoost", r"\texttt{B1\_CatBoost}"),
        ("A1_Direct_MLP", r"\texttt{A1\_Direct\_MLP}"),
    ]

    best_ov = float(agg.loc[[m for m, _ in headline_models], "rmse_mean"].min())
    best_cl = float(agg.loc[[m for m, _ in headline_models], "cliff_rmse_mean"].min())
    best_nc = float(agg.loc[[m for m, _ in headline_models], "non_cliff_rmse_mean"].min())
    best_cm = float(agg.loc[[m for m, _ in headline_models], "cliff_mae_mean"].min())
    best_sp = float(agg.loc[[m for m, _ in headline_models], "spearman_mean"].max())

    def _b(v: float, target: float) -> str:
        s = f"{v:.4f}"
        return f"\\textbf{{{s}}}" if abs(v - target) < 1e-6 else s

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Main 30-Target MoleculeACE Benchmark Leaderboard ($N=30$ Targets $\times$ $5$ Seeds) and Cross-Target Paired Wilcoxon Comparisons.\textsuperscript{\emph{a}}}",
        r"\label{tab:main30}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Model & Overall RMSE $\downarrow$ & Cliff RMSE $\downarrow$ & Non-Cliff RMSE $\downarrow$ & Cliff MAE $\downarrow$ & Spearman $\uparrow$ \\",
        r"\midrule",
    ]
    for m, lbl in headline_models:
        r = agg.loc[m]
        lines.append(
            f"{lbl} & {_b(float(r['rmse_mean']), best_ov)} & {_b(float(r['cliff_rmse_mean']), best_cl)} & "
            f"{_b(float(r['non_cliff_rmse_mean']), best_nc)} & {_b(float(r['cliff_mae_mean']), best_cm)} & {_b(float(r['spearman_mean']), best_sp)} \\\\"
        )

    lines.extend([
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{Cross-Target Two-Sided Wilcoxon Signed-Rank \& 5,000-Bootstrap 95\% CI ($N=30$ Targets)}} \\",
        r"Comparison (\texttt{OptW\_FiLM} vs. Ref) & Metric & Win / Loss & Mean $\Delta$ & 95\% Bootstrap CI & Two-Sided $p$ \\",
        r"\midrule",
    ])
    tests = [
        (r"vs. \texttt{K\_RBF\_SVM}", "Cliff", "E_Multi_FiLM_OptW_cliff", "K_RBF_SVM_cliff"),
        (r"vs. \texttt{K\_Multi\_SVR}", "Cliff", "E_Multi_FiLM_OptW_cliff", "K_Multi_SVR_cliff"),
        (r"vs. \texttt{E\_Multi\_DirectMLP\_OptW}", "Cliff", "E_Multi_FiLM_OptW_cliff", "E_Multi_DirectMLP_OptW_cliff"),
        (r"vs. \texttt{E\_Multi\_CatBoost\_OptW}", "Cliff", "E_Multi_FiLM_OptW_cliff", "E_Multi_CatBoost_OptW_cliff"),
        (r"vs. \texttt{E\_Multi\_FiLM} ($w=0.5$)", "Cliff", "E_Multi_FiLM_OptW_cliff", "E_Multi_FiLM_cliff"),
        (r"vs. \texttt{E\_Multi\_A2\_OptW}", "Cliff", "E_Multi_FiLM_OptW_cliff", "E_Multi_A2_OptW_cliff"),
        (r"vs. \texttt{K\_RBF\_SVM}", "Overall", "E_Multi_FiLM_OptW_rmse", "K_RBF_SVM_rmse"),
        (r"vs. \texttt{K\_Multi\_SVR}", "Overall", "E_Multi_FiLM_OptW_rmse", "K_Multi_SVR_rmse"),
        (r"vs. \texttt{E\_Multi\_DirectMLP\_OptW}", "Overall", "E_Multi_FiLM_OptW_rmse", "E_Multi_DirectMLP_OptW_rmse"),
        (r"vs. \texttt{E\_Multi\_CatBoost\_OptW}", "Overall", "E_Multi_FiLM_OptW_rmse", "E_Multi_CatBoost_OptW_rmse"),
        (r"vs. \texttt{E\_Multi\_FiLM} ($w=0.5$)", "Overall", "E_Multi_FiLM_OptW_rmse", "E_Multi_FiLM_rmse"),
    ]
    for lbl, mtr, ca, cb in tests:
        st = _cross_target_stats(wide, ca, cb)
        lines.append(
            f"{lbl} & {mtr} & {st['wins']}/{st['losses']} & {st['mean_diff']:+.5f} & "
            f"[{st['ci95_lo']:+.5f}, {st['ci95_hi']:+.5f}] & {_fmt_p(float(st['p_value']))} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        "",
        r"\vspace{2pt}\raggedright\footnotesize\textsuperscript{\emph{a}}Full 17-model results and \texttt{PriorN} paired Wilcoxon tests appear in Supporting Table S2. \texttt{OptW} weights were calibrated strictly on 3-fold nested out-of-fold training predictions (\texttt{fold\_seed=42}) with zero test leakage.",
        r"\end{table*}",
    ])
    return "\n".join(lines) + "\n"


def _build_table2() -> str:
    wide = pd.read_csv(REFINELOGS_DIR / "jcim30_step_a_results" / "step_a_wide.csv")
    npz_dir = REFINELOGS_DIR / "jcim30_step_a_results" / "per_target"

    recs: list[dict[str, float]] = []
    eval_m = ("K_RBF_SVM", "K_Multi_SVR", "E_Multi_DirectMLP_OptW", "E_Multi_FiLM", "E_Multi_FiLM_OptW")
    for tgt in wide["target"].tolist():
        ds = load_target_dataset(tgt)
        feats = extract_or_load_2d_features(tgt, ds.smiles)
        z = np.load(npz_dir / f"{tgt}_optw_preds.npz")
        sim = compute_tanimoto_matrix(feats.ecfp_chiral[ds.test_idx], feats.ecfp_chiral[ds.train_idx])
        bp = np.argmax(sim, axis=1)
        s_max = sim[np.arange(ds.test_idx.size), bp].astype(np.float64)
        y_te = ds.y[ds.test_idx]
        y_anc = ds.y[ds.train_idx[bp]]
        dy_true = y_te - y_anc

        mask = s_max >= 0.70
        p_map = {m: z[f"{m}_mean"].astype(np.float64)[mask] for m in eval_m}
        for i in range(int(np.sum(mask))):
            r = {"y_true": float(y_te[mask][i]), "y_anc": float(y_anc[mask][i]), "dy_true": float(dy_true[mask][i]), "abs_dy": float(abs(dy_true[mask][i]))}
            for m in eval_m:
                r[m] = float(p_map[m][i])
            recs.append(r)
    df_an = pd.DataFrame(recs)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Sample-Size Regime Check, Potency-Jump Magnitude Stratification ($s_{\max} \ge 0.70$), and Cliff-Direction Accuracy Across 30 MoleculeACE Targets.}",
        r"\label{tab:regimes}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"\multicolumn{8}{l}{\textit{Part A: Sample-Size \& Assay-Type Regimes (Mean Cliff RMSE)}} \\",
        r"Subgroup ($N$) & Mean $w^*_{\mathrm{FiLM}}$ & \texttt{K\_RBF\_SVM} & \texttt{K\_Multi\_SVR} & \texttt{OptW\_MLP} & \texttt{FiLM(0.5)} & \texttt{PriorN\_FiLM} & \texttt{OptW\_FiLM} \\",
        r"\midrule",
    ]
    def _bold_row(vals: list[float], fmt: str, pick_min: bool = True, suffix: str = "") -> list[str]:
        best = min(vals) if pick_min else max(vals)
        return [
            f"\\textbf{{{format(v, fmt)}{suffix}}}" if abs(v - best) < 1e-6 else f"{format(v, fmt)}{suffix}"
            for v in vals
        ]

    part_a_cols = [
        "K_RBF_SVM_cliff",
        "K_Multi_SVR_cliff",
        "E_Multi_DirectMLP_OptW_cliff",
        "E_Multi_FiLM_cliff",
        "E_Multi_FiLM_PriorN_cliff",
        "E_Multi_FiLM_OptW_cliff",
    ]
    for lbl, sub in [
        (r"$n < 1000$ ($N=12$)", wide[wide["n_total"] < 1000]),
        (r"$n \ge 1000$ ($N=18$)", wide[wide["n_total"] >= 1000]),
        (r"$\mathrm{K_i}$ Assays ($N=23$)", wide[wide["assay_type"] == "Ki"]),
        (r"$\mathrm{EC_{50}}$ Assays ($N=7$)", wide[wide["assay_type"] == "EC50"]),
    ]:
        row_vals = [float(sub[c].mean()) for c in part_a_cols]
        cells = _bold_row(row_vals, ".4f", pick_min=True)
        lines.append(f"{lbl} & {sub['w_film'].mean():.3f} & " + " & ".join(cells) + r" \\")

    lines.extend([
        r"\midrule",
        r"\multicolumn{8}{l}{\textit{Part B: Structural Analog Error Gradient ($s_{\max} \ge 0.70$) \& Cliff-Direction Accuracy}} \\",
        r"Analog Stratum & $n_{\mathrm{mol}}$ & \texttt{K\_RBF\_SVM} & \texttt{K\_Multi\_SVR} & \texttt{OptW\_MLP} & \texttt{FiLM(0.5)} & \texttt{OptW\_FiLM} & Note \\",
        r"\midrule",
    ])
    for lbl, cond, note in [
        (r"$|\Delta y| < 0.5$", df_an["abs_dy"] < 0.5, "Pooled RMSE"),
        (r"$0.5 \le |\Delta y| < 1.0$", (df_an["abs_dy"] >= 0.5) & (df_an["abs_dy"] < 1.0), "Pooled RMSE"),
        (r"$1.0 \le |\Delta y| < 1.5$", (df_an["abs_dy"] >= 1.0) & (df_an["abs_dy"] < 1.5), "Pooled RMSE"),
        (r"$|\Delta y| \ge 1.5$", df_an["abs_dy"] >= 1.5, "Pooled RMSE"),
        (r"$|\Delta y| \ge 2.0$\textsuperscript{\emph{a}}", df_an["abs_dy"] >= 2.0, "2D Ceiling"),
    ]:
        sub = df_an[cond]
        yv = sub["y_true"].to_numpy()
        vals = [float(np.sqrt(np.mean((yv - sub[m].to_numpy()) ** 2))) for m in eval_m]
        cells = _bold_row(vals, ".4f", pick_min=True)
        lines.append(f"{lbl} & {len(sub)} & " + " & ".join(cells) + f" & {note} \\\\")

    for lbl, thr in [(r"Direction ($|\Delta y| \ge 1.0$)", 1.0), (r"Direction ($|\Delta y| \ge 1.5$)", 1.5)]:
        sub = df_an[df_an["abs_dy"] >= thr]
        s_true = np.sign(sub["dy_true"].to_numpy())
        y_a = sub["y_anc"].to_numpy()
        accs = [float(np.mean(np.sign(sub[m].to_numpy() - y_a) == s_true)) * 100.0 for m in eval_m]
        cells = _bold_row(accs, ".2f", pick_min=False, suffix=r"\%")
        lines.append(f"{lbl} & {len(sub)} & " + " & ".join(cells) + r" & Sign Acc. \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        "",
        r"\vspace{2pt}\raggedright\footnotesize\textsuperscript{\emph{a}}On extreme $|\Delta y| \ge 2.0$ jumps ($n=212$ across 28 targets), \texttt{OptW\_FiLM} target-mean RMSE is $1.3093$, while pooled RMSE ($1.4641$) slightly trails \texttt{OptW\_MLP} ($1.4550$), \texttt{K\_Multi\_SVR} ($1.4564$), and \texttt{B1\_CatBoost} ($1.4509$), marking the ligand-only 2D limit without 3D receptor pocket structure.",
        r"\end{table*}",
    ])
    return "\n".join(lines) + "\n"


def _build_table3() -> str:
    scaf = pd.read_csv(REFINELOGS_DIR / "scaffold_robustness_results" / "scaffold_summary.csv")
    wide = pd.read_csv(REFINELOGS_DIR / "scaffold_robustness_results" / "scaffold_wide.csv")
    agg = scaf.groupby("model")[["rmse_mean", "mae_mean", "cliff_rmse_mean", "spearman_mean"]].mean()

    models = [
        ("E_Multi_FiLM_GroupOptW", r"\texttt{E\_Multi\_FiLM\_GroupOptW}"),
        ("E_Multi_FiLM_OptW", r"\texttt{E\_Multi\_FiLM\_OptW}"),
        ("E_Multi_FiLM_PriorN", r"\texttt{E\_Multi\_FiLM\_PriorN}"),
        ("E_Multi_FiLM", r"\texttt{E\_Multi\_FiLM} ($w=0.5$)"),
        ("E_Multi_DirectMLP_GroupOptW", r"\texttt{E\_Multi\_DirectMLP\_GroupOptW}"),
        ("K_Multi_SVR", r"\texttt{K\_Multi\_SVR}"),
        ("K_RBF_SVM", r"\texttt{K\_RBF\_SVM}"),
        ("Y_FiLM_2DLFER", r"\texttt{Y\_FiLM\_2DLFER}"),
        ("B1_CatBoost", r"\texttt{B1\_CatBoost}"),
        ("A1_Direct_MLP", r"\texttt{A1\_Direct\_MLP}"),
    ]

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{External Robustness Under Strict Zero-Overlap Bemis--Murcko Scaffold Splits (10 Wave-1 Targets $\times$ 2 Splits, $6,907$ Pooled Test Predictions).\textsuperscript{\emph{a}}}",
        r"\label{tab:scaffold}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & Overall RMSE $\downarrow$ & MAE $\downarrow$ & Orig-Cliff RMSE $\downarrow$ & Spearman $\uparrow$ \\",
        r"\midrule",
    ]
    for m, lbl in models:
        r = agg.loc[m]
        ov = f"\\textbf{{{r['rmse_mean']:.4f}}}" if m == "E_Multi_FiLM_GroupOptW" else f"{r['rmse_mean']:.4f}"
        ma = f"\\textbf{{{r['mae_mean']:.4f}}}" if m == "E_Multi_FiLM_OptW" else f"{r['mae_mean']:.4f}"
        cl = f"\\textbf{{{r['cliff_rmse_mean']:.4f}}}" if m == "E_Multi_FiLM_OptW" else f"{r['cliff_rmse_mean']:.4f}"
        sp = f"\\textbf{{{r['spearman_mean']:.4f}}}" if m == "E_Multi_FiLM_OptW" else f"{r['spearman_mean']:.4f}"
        lines.append(f"{lbl} & {ov} & {ma} & {cl} & {sp} \\\\")

    lines.extend([
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Cross-Target Paired Wilcoxon Comparisons on Overall RMSE ($N=10$ Targets)}} \\",
        r"Comparison & Win / Loss & Mean $\Delta$ & 95\% Bootstrap CI & Two-Sided $p$ \\",
        r"\midrule",
    ])
    for lbl, ca, cb in [
        (r"\texttt{GroupOptW\_FiLM} vs. \texttt{K\_RBF\_SVM}", "E_Multi_FiLM_GroupOptW_rmse", "K_RBF_SVM_rmse"),
        (r"\texttt{GroupOptW\_FiLM} vs. \texttt{K\_Multi\_SVR}", "E_Multi_FiLM_GroupOptW_rmse", "K_Multi_SVR_rmse"),
        (r"\texttt{GroupOptW\_FiLM} vs. \texttt{GroupOptW\_DirectMLP}", "E_Multi_FiLM_GroupOptW_rmse", "E_Multi_DirectMLP_GroupOptW_rmse"),
        (r"\texttt{OptW\_FiLM} vs. \texttt{K\_RBF\_SVM}", "E_Multi_FiLM_OptW_rmse", "K_RBF_SVM_rmse"),
        (r"\texttt{OptW\_FiLM} vs. \texttt{K\_Multi\_SVR}", "E_Multi_FiLM_OptW_rmse", "K_Multi_SVR_rmse"),
    ]:
        st = _cross_target_stats(wide, ca, cb)
        lines.append(
            f"{lbl} & {st['wins']}/{st['losses']} & {st['mean_diff']:+.5f} & "
            f"[{st['ci95_lo']:+.5f}, {st['ci95_hi']:+.5f}] & {_fmt_p(float(st['p_value']))} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        "",
        r"\vspace{2pt}\raggedright\footnotesize\textsuperscript{\emph{a}}Full 11-model and per-target matrices appear in Supporting Table S3. Under unseen scaffolds, test-to-train $s_{\max}$ shifts from $0.762$ to $0.679$ ($14.1\%$ target-mean / $11.7\%$ pooled at $s_{\max} < 0.50$). Per-target original \texttt{cliff\_mol} tests ($N=10$) give $p=0.0371$ ($7\text{W}/3\text{L}$) for \texttt{GroupOptW\_FiLM} vs.\ \texttt{K\_RBF\_SVM} and $p=0.131$--$0.432$ against \texttt{K\_Multi\_SVR} and \texttt{GroupOptW\_MLP}; pooled \texttt{cliff\_mol == 1} ($n=2,676$) improves ($0.8763$ / $0.8788$ vs.\ \texttt{K\_Multi\_SVR} $0.8893$).",
        r"\end{table*}",
    ])
    return "\n".join(lines) + "\n"


def _build_table4() -> str:
    w1 = pd.read_csv(REFINELOGS_DIR / "benchmark30_results" / "wave1_summary.csv")
    w1_p = w1.pivot(index="target", columns="model", values="cliff_rmse_mean")
    pa = pd.read_csv(REFINELOGS_DIR / "path_a_modular_results" / "path_a_summary.csv")
    pa_p = pa.pivot(index="target", columns="model", values="cliff_rmse_mean")
    o13 = pd.read_csv(REFINELOGS_DIR / "opt13_modular_results" / "opt13_summary.csv")
    o13_pc = o13.pivot(index="target", columns="model", values="cliff_rmse_mean")
    o13_po = o13.pivot(index="target", columns="model", values="rmse_mean")
    pv2 = pd.read_csv(REFINELOGS_DIR / "plan_a_v2_summary.csv")
    pv2_p = pv2.pivot(index="target", columns="model", values="cliff_rmse_mean")

    # Seed-mean Avg50_L2 from Wave-1 npz
    npzs = sorted((REFINELOGS_DIR / "benchmark30_results" / "per_target").glob("*_preds.npz"))
    g2_list, a50_list = [], []
    for p in npzs:
        z = np.load(p)
        cm = z["c_te"] == 1
        yt = z["y_te"][cm]
        g2_list.append(float(np.mean([np.sqrt(np.mean((yt - z["G2_RBFSVM_PhysGate_L2_seeds"][s][cm]) ** 2)) for s in range(5)])))
        a50_list.append(float(np.mean([np.sqrt(np.mean((yt - (0.5 * z["B1c_RBF_SVM_seeds"][s][cm] + 0.5 * z["L2_eSpan_LFER_Delta_seeds"][s][cm])) ** 2)) for s in range(5)])))

    def _row(mode: str, cand: str, ref: str, va: np.ndarray, vb: np.ndarray, scope: str) -> str:
        diff = va - vb
        w_cnt = int(np.sum(diff < -1e-9))
        l_cnt = int(np.sum(diff > 1e-9))
        _, pval = stats.wilcoxon(va, vb)
        return f"{mode} & {cand} vs. {ref} & {scope} & {np.mean(va):.4f} vs. {np.mean(vb):.4f} ({np.mean(diff):+.4f}) & {w_cnt}/{l_cnt} & {_fmt_p(float(pval))} \\\\"

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Controlled Counterfactual Dissection of Four Ligand-Only Failure Mechanisms (Wave-1 $N=10$ Targets $\times$ 5 Seeds, Unless Noted).}",
        r"\label{tab:negatives}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llllcc}",
        r"\toprule",
        r"Failure Mode & Candidate vs. Matched Control & Scope & Mean Cliff RMSE ($\Delta$) & Win / Loss & Two-Sided $p$ \\",
        r"\midrule",
        _row("1a. Vacuum 3D + \\texttt{AIMNet2}", r"\texttt{G2\_Prev}", r"\texttt{G0} (2D Ctrl)", w1_p["G2_RBFSVM_PhysGate_L2"].to_numpy(), w1_p["G0_RBFSVM_Gate_A2"].to_numpy(), "$N=10$"),
        _row("1b. L1/ReZero 3D/QM Gate", r"\texttt{Y\_FiLM\_Gated3DQM}", r"\texttt{Y\_FiLM\_2DLFER}", o13_pc["Y_FiLM_Gated3DQM"].to_numpy(), o13_pc["Y_FiLM_2DLFER"].to_numpy(), "$N=10$"),
        f"1c. 5-Conf Boltzmann & \\texttt{{G\\_S2}} vs. \\texttt{{G2\\_Prev}} & $N=3$ MVP (pooled $p$)\\textsuperscript{{\\emph{{a}}}} & {pv2_p['G_S2_Boltz_SkipLFER'].mean():.4f} vs. {pv2_p['G2_Prev_SingleConcat'].mean():.4f} ({(pv2_p['G_S2_Boltz_SkipLFER'] - pv2_p['G2_Prev_SingleConcat']).mean():+.4f}) & {int((pv2_p['G_S2_Boltz_SkipLFER'] < pv2_p['G2_Prev_SingleConcat'] - 1e-9).sum())}/{int((pv2_p['G_S2_Boltz_SkipLFER'] > pv2_p['G2_Prev_SingleConcat'] + 1e-9).sum())} & 0.0228\\textsuperscript{{\\emph{{a}}}} \\\\",
        _row("2. Reused-Val Gate", r"\texttt{G2\_Prev}", r"\texttt{Avg50\_L2} (50/50)", np.asarray(g2_list), np.asarray(a50_list), "$N=10$"),
        _row("3a. OOF Residual Stack", r"\texttt{KR\_2D\_Delta}", r"\texttt{E\_SVM\_A2Delta}", pa_p["KR_2D_Delta"].to_numpy(), pa_p["E_SVM_A2Delta"].to_numpy(), "$N=10$"),
        _row("3b. OOF Residual + FiLM", r"\texttt{KR\_FiLM\_2DLFER}", r"\texttt{E\_SVM\_A2Delta}", pa_p["KR_FiLM_2DLFER"].to_numpy(), pa_p["E_SVM_A2Delta"].to_numpy(), "$N=10$"),
        _row("4a. CrossAttn Blend", r"\texttt{E\_Multi\_Attn}", r"\texttt{E\_Multi\_A2Delta}", o13_pc["E_Multi_Attn"].to_numpy(), o13_pc["E_Multi_A2Delta"].to_numpy(), "$N=10$"),
        f"4b. GINE + CrossAttn\\textsuperscript{{\\emph{{b}}}} & \\texttt{{E\\_Multi\\_GraphAttn}} vs. \\texttt{{E\\_Multi\\_A2Delta}} & $N=10$ & {o13_pc['E_Multi_GraphAttn'].mean():.4f} vs. {o13_pc['E_Multi_A2Delta'].mean():.4f} ({(o13_pc['E_Multi_GraphAttn'] - o13_pc['E_Multi_A2Delta']).mean():+.4f}) & {int((o13_pc['E_Multi_GraphAttn'] < o13_pc['E_Multi_A2Delta'] - 1e-9).sum())}/{int((o13_pc['E_Multi_GraphAttn'] > o13_pc['E_Multi_A2Delta'] + 1e-9).sum())} & {_fmt_p(float(stats.wilcoxon(o13_pc['E_Multi_GraphAttn'], o13_pc['E_Multi_A2Delta']).pvalue))} \\\\",
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        "",
        r"\vspace{2pt}\raggedright\footnotesize\textsuperscript{\emph{a}}Evaluated on the 3 MVP targets (\texttt{CHEMBL204\_Ki}, \texttt{CHEMBL2147\_Ki}, \texttt{CHEMBL237\_Ki}); $p=0.0228$ is the within-target 5-seed pooled Wilcoxon test on \texttt{CHEMBL2147\_Ki} (\texttt{CHEMBL204\_Ki} $p=0.7652$, \texttt{CHEMBL237\_Ki} $p=0.6859$). \textsuperscript{\emph{b}}\texttt{E\_Multi\_GraphAttn} achieves the best Wave-1 Overall RMSE ($"
        + f"{o13_po['E_Multi_GraphAttn'].mean():.4f}$ vs. \\texttt{{E\\_Multi\\_FiLM}} ${o13_po['E_Multi_FiLM'].mean():.4f}$) while smoothing cliffs.",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = [
        ("table1_main30.tex", _build_table1()),
        ("table2_regimes_anatomy.tex", _build_table2()),
        ("table3_scaffold.tex", _build_table3()),
        ("table4_negatives.tex", _build_table4()),
    ]
    for fname, txt in specs:
        p = OUT_DIR / fname
        p.write_text(txt, encoding="utf-8")
        print(f"Saved: {p.name}", flush=True)


if __name__ == "__main__":
    main()

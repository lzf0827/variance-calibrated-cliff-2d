"""Step A: Strictly Leak-Free 3-Fold Nested OOF Bates-Granger Weighting + |Delta pKi| Gradient Anatomy + Chemical Cases."""

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
from sklearn.model_selection import StratifiedKFold

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from physchem_delta.data import load_target_dataset
from physchem_delta.eval_opt13_modular import _fit_kernel_bases, _train_yspace_modular
from physchem_delta.eval_plan_a import _lfer_atom_matrix, _train_delta_backbone
from physchem_delta.featurize import compute_local_pair_deltas, precompute_mol_env_counters
from physchem_delta.metrics import compute_mae, compute_rmse, evaluate_predictions
from physchem_delta.models import (
    _train_direct_mlp_with_inner_split,
    build_tier_matrices,
    compute_tanimoto_matrix,
    make_inner_split,
    run_tabular_baseline,
)
from physchem_delta.run_30_targets import ALL_30_TARGETS
from physchem_delta.run_jcim_30targets import _cross_target_stats, extract_or_load_2d_features

STEP_A_MODELS: tuple[str, ...] = (
    "K_RBF_SVM",
    "K_Multi_SVR",
    "B1_CatBoost",
    "A1_Direct_MLP",
    "Y_A2_2D",
    "Y_FiLM_2DLFER",
    "E_Multi_CatBoost",
    "E_Multi_DirectMLP",
    "E_Multi_A2Delta",
    "E_Multi_FiLM",
    "E_Multi_CatBoost_OptW",
    "E_Multi_DirectMLP_OptW",
    "E_Multi_A2_OptW",
    "E_Multi_FiLM_OptW",
    "E_Multi_CatBoost_PriorN",
    "E_Multi_DirectMLP_PriorN",
    "E_Multi_FiLM_PriorN",
)


def _out_root() -> Path:
    d = ROOT_DIR / "refine-logs" / "jcim30_step_a_results"
    (d / "per_target").mkdir(parents=True, exist_ok=True)
    return d


def _fig_dir() -> Path:
    d = ROOT_DIR / "refine-logs" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _bates_granger_weight(y_true: np.ndarray, p_base: np.ndarray, p_cand: np.ndarray) -> float:
    """Closed-form minimum-MSE convex weight w in [0, 1] for (1 - w)*p_base + w*p_cand."""
    e_k = np.asarray(y_true, dtype=np.float64) - np.asarray(p_base, dtype=np.float64)
    e_m = np.asarray(y_true, dtype=np.float64) - np.asarray(p_cand, dtype=np.float64)
    diff = e_k - e_m
    denom = float(np.dot(diff, diff))
    if denom <= 1e-12:
        return 0.0
    w = float(np.dot(e_k, diff) / denom)
    return float(np.clip(w, 0.0, 1.0))


def _stratified_3fold_indices(y_train: np.ndarray, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    n = y_train.size
    n_bins = min(5, max(2, n // 45))
    bins = np.zeros((n,), dtype=int)
    if n >= 30:
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(y_train, qs)
        edges[0] -= 1e-6
        edges[-1] += 1e-6
        bins = np.digitize(y_train, edges[1:-1])
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros((n, 1)), bins))


def _evaluate_target_step_a(target: str, force: bool, save_cache: bool) -> dict[str, Any]:
    out_dir = _out_root() / "per_target"
    json_path = out_dir / f"{target}_optw.json"
    npz_path = out_dir / f"{target}_optw_preds.npz"

    if save_cache and json_path.exists() and npz_path.exists() and not force:
        cached = json.loads(json_path.read_text(encoding="utf-8"))
        if all(m in cached.get("metrics", {}) for m in STEP_A_MODELS):
            print(f"[CACHE] {target}: loaded Step-A OptW checkpoint", flush=True)
            return cached

    t0 = time.time()
    ds = load_target_dataset(target)
    feats = extract_or_load_2d_features(target, ds.smiles)
    lfer_atoms = _lfer_atom_matrix(feats)
    dummy_graphs = [{"x_cont": np.zeros((1, 3), dtype=np.float32)}] * len(ds.smiles)

    jcim_npz_path = ROOT_DIR / "refine-logs" / "jcim30_results" / "per_target" / f"{target}_preds.npz"
    if not jcim_npz_path.exists():
        raise FileNotFoundError(f"Missing JCIM-30 test predictions: {jcim_npz_path}")
    j30 = np.load(jcim_npz_path)

    y_te = ds.y[ds.test_idx]
    c_te = ds.cliff_mol[ds.test_idx]
    assert np.allclose(y_te, j30["y_te"]) and np.array_equal(c_te, j30["c_te"]), f"Split mismatch on {target}"

    # Part I: Strictly leak-free 3-fold nested OOF inside train_idx ONLY
    n_tr = ds.train_idx.size
    oof_k = np.zeros((n_tr,), dtype=np.float64)
    oof_film = np.zeros((n_tr,), dtype=np.float64)
    oof_a2 = np.zeros((n_tr,), dtype=np.float64)
    oof_mlp = np.zeros((n_tr,), dtype=np.float64)
    oof_cat = np.zeros((n_tr,), dtype=np.float64)

    fold_seed = 42
    for tr_rel_f, va_rel_f in _stratified_3fold_indices(ds.y[ds.train_idx], seed=fold_seed):
        tr_f = ds.train_idx[tr_rel_f]
        va_f = ds.train_idx[va_rel_f]

        _, oof_k[va_rel_f] = _fit_kernel_bases(feats, ds.y, tr_f, va_f)
        oof_cat[va_rel_f] = run_tabular_baseline("B1_ECFP_Chiral_CatBoost", feats, ds.y, tr_f, va_f, seed=fold_seed)["y_pred_test"]

        in_tr_rel, in_va_rel = make_inner_split(ds.y[tr_f], seed=fold_seed, val_frac=0.15)
        in_tr_idx = tr_f[in_tr_rel]
        in_va_idx = tr_f[in_va_rel]
        bundle_f = build_tier_matrices(feats, tier="2d", fit_idx=in_tr_idx)

        oof_mlp[va_rel_f] = _train_direct_mlp_with_inner_split(
            bundle_f.x_mol, ds.y, tr_f, in_tr_rel, in_va_rel, va_f, seed=fold_seed
        )["y_pred_test"]
        oof_a2[va_rel_f] = _train_delta_backbone(
            feats, lfer_atoms, ds.y, tr_f, va_f, seed=fold_seed, mode="2d"
        )["y_pred_test"]
        oof_film[va_rel_f] = _train_yspace_modular(
            "Y_FiLM_2DLFER", feats, dummy_graphs, lfer_atoms, ds.y,
            tr_f, in_tr_idx, in_va_idx, va_f, seed=fold_seed,
        )["y_pred_test"]

    y_tr_vals = ds.y[ds.train_idx]
    w_star = {
        "film": _bates_granger_weight(y_tr_vals, oof_k, oof_film),
        "a2": _bates_granger_weight(y_tr_vals, oof_k, oof_a2),
        "mlp": _bates_granger_weight(y_tr_vals, oof_k, oof_mlp),
        "cat": _bates_granger_weight(y_tr_vals, oof_k, oof_cat),
    }
    w_prior = float(0.5 * n_tr / (n_tr + 1000.0))

    oof_rmse = {
        "K_Multi_SVR": compute_rmse(y_tr_vals, oof_k),
        "Y_FiLM_2DLFER": compute_rmse(y_tr_vals, oof_film),
        "Y_A2_2D": compute_rmse(y_tr_vals, oof_a2),
        "A1_Direct_MLP": compute_rmse(y_tr_vals, oof_mlp),
        "B1_CatBoost": compute_rmse(y_tr_vals, oof_cat),
    }

    # Apply scalar weights cleanly to cached 5-seed test predictions (shape: (5, n_test))
    k_mul = j30["K_Multi_SVR_seeds"].astype(np.float64)
    k_rbf = j30["K_RBF_SVM_seeds"].astype(np.float64)
    p_cat = j30["B1_CatBoost_seeds"].astype(np.float64)
    p_mlp = j30["A1_Direct_MLP_seeds"].astype(np.float64)
    p_a2 = j30["Y_A2_2D_seeds"].astype(np.float64)
    p_film = j30["Y_FiLM_2DLFER_seeds"].astype(np.float64)

    preds_seeds_map: dict[str, np.ndarray] = {
        "K_RBF_SVM": k_rbf,
        "K_Multi_SVR": k_mul,
        "B1_CatBoost": p_cat,
        "A1_Direct_MLP": p_mlp,
        "Y_A2_2D": p_a2,
        "Y_FiLM_2DLFER": p_film,
        "E_Multi_CatBoost": 0.5 * k_mul + 0.5 * p_cat,
        "E_Multi_DirectMLP": 0.5 * k_mul + 0.5 * p_mlp,
        "E_Multi_A2Delta": 0.5 * k_mul + 0.5 * p_a2,
        "E_Multi_FiLM": 0.5 * k_mul + 0.5 * p_film,
        "E_Multi_CatBoost_OptW": (1.0 - w_star["cat"]) * k_mul + w_star["cat"] * p_cat,
        "E_Multi_DirectMLP_OptW": (1.0 - w_star["mlp"]) * k_mul + w_star["mlp"] * p_mlp,
        "E_Multi_A2_OptW": (1.0 - w_star["a2"]) * k_mul + w_star["a2"] * p_a2,
        "E_Multi_FiLM_OptW": (1.0 - w_star["film"]) * k_mul + w_star["film"] * p_film,
        "E_Multi_CatBoost_PriorN": (1.0 - w_prior) * k_mul + w_prior * p_cat,
        "E_Multi_DirectMLP_PriorN": (1.0 - w_prior) * k_mul + w_prior * p_mlp,
        "E_Multi_FiLM_PriorN": (1.0 - w_prior) * k_mul + w_prior * p_film,
    }

    metrics_summary: dict[str, dict[str, float]] = {}
    npz_payload: dict[str, np.ndarray] = {"y_te": y_te, "c_te": c_te}
    for m_name in STEP_A_MODELS:
        arr = preds_seeds_map[m_name]
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"Non-finite values in {target} {m_name}")
        seed_res = [evaluate_predictions(y_te, arr[s], c_te) for s in range(arr.shape[0])]
        metrics_summary[m_name] = {
            "rmse_mean": float(np.mean([r["rmse"] for r in seed_res])),
            "rmse_std": float(np.std([r["rmse"] for r in seed_res])),
            "cliff_rmse_mean": float(np.mean([r["cliff_rmse"] for r in seed_res])),
            "cliff_rmse_std": float(np.std([r["cliff_rmse"] for r in seed_res])),
            "non_cliff_rmse_mean": float(np.mean([r["non_cliff_rmse"] for r in seed_res])),
            "mae_mean": float(np.mean([r["mae"] for r in seed_res])),
            "cliff_mae_mean": float(np.mean([r["cliff_mae"] for r in seed_res])),
            "spearman_mean": float(np.mean([r["spearman"] for r in seed_res])),
        }
        npz_payload[f"{m_name}_mean"] = np.mean(arr, axis=0)
        npz_payload[f"{m_name}_seeds"] = arr

    elapsed = round(time.time() - t0, 2)
    result = {
        "target": target,
        "assay_type": "EC50" if target.endswith("_EC50") else "Ki",
        "n_total": int(len(ds.smiles)),
        "n_train": int(n_tr),
        "n_test": int(y_te.size),
        "n_test_cliff": int(np.sum(c_te == 1)),
        "elapsed_sec": elapsed,
        "w_star": w_star,
        "w_prior": w_prior,
        "oof_rmse": oof_rmse,
        "metrics": metrics_summary,
    }

    if save_cache:
        np.savez_compressed(npz_path, **npz_payload)
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    m = metrics_summary
    print(
        f"[DONE {elapsed:.1f}s] {target} (n={len(ds.smiles)}): "
        f"RBF={m['K_RBF_SVM']['cliff_rmse_mean']:.4f} | Multi={m['K_Multi_SVR']['cliff_rmse_mean']:.4f} | "
        f"MLP_OptW={m['E_Multi_DirectMLP_OptW']['cliff_rmse_mean']:.4f} | "
        f"FiLM_0.5={m['E_Multi_FiLM']['cliff_rmse_mean']:.4f} | "
        f"FiLM_OptW={m['E_Multi_FiLM_OptW']['cliff_rmse_mean']:.4f} (w*={w_star['film']:.3f}) | "
        f"FiLM_Prior={m['E_Multi_FiLM_PriorN']['cliff_rmse_mean']:.4f} (wp={w_prior:.3f})",
        flush=True,
    )
    return result


def _compute_potency_gradient_and_cases(results: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute Part II (|Delta pKi| bins & direction accuracy) and Part III (4 WIN + 2 FAIL medchem cases)."""
    out_dir = _out_root() / "per_target"
    eval_models = (
        "K_RBF_SVM",
        "K_Multi_SVR",
        "B1_CatBoost",
        "A1_Direct_MLP",
        "E_Multi_DirectMLP_OptW",
        "E_Multi_CatBoost_OptW",
        "E_Multi_FiLM",
        "E_Multi_FiLM_OptW",
    )

    all_records: list[dict[str, Any]] = []
    case_candidates: dict[str, dict[str, Any]] = {}

    win_targets_pool = {
        "Protease": ["CHEMBL244_Ki", "CHEMBL204_Ki"],
        "Kinase": ["CHEMBL4792_Ki", "CHEMBL2147_Ki"],
        "GPCR": ["CHEMBL237_Ki", "CHEMBL219_Ki"],
        "EC50": ["CHEMBL3979_EC50", "CHEMBL235_EC50"],
    }
    fail_targets = ["CHEMBL231_Ki", "CHEMBL4203_Ki"]
    wanted_case_targets = {t for pool in win_targets_pool.values() for t in pool} | set(fail_targets)

    for r in results:
        tgt = r["target"]
        ds = load_target_dataset(tgt)
        feats = extract_or_load_2d_features(tgt, ds.smiles)
        z = np.load(out_dir / f"{tgt}_optw_preds.npz")

        sim_te_tr = compute_tanimoto_matrix(feats.ecfp_chiral[ds.test_idx], feats.ecfp_chiral[ds.train_idx])
        best_pos = np.argmax(sim_te_tr, axis=1)
        s_max = sim_te_tr[np.arange(ds.test_idx.size), best_pos]
        nearest_tr = ds.train_idx[best_pos]
        y_te = ds.y[ds.test_idx]
        y_anc = ds.y[nearest_tr]
        dy_true = y_te - y_anc
        abs_dy = np.abs(dy_true)

        preds_mean = {m: z[f"{m}_mean"].astype(np.float64) for m in eval_models}
        for k_pos in range(y_te.size):
            rec = {
                "target": tgt,
                "s_max": float(s_max[k_pos]),
                "y_true": float(y_te[k_pos]),
                "y_anc": float(y_anc[k_pos]),
                "dy_true": float(dy_true[k_pos]),
                "abs_dy": float(abs_dy[k_pos]),
            }
            for m in eval_models:
                rec[m] = float(preds_mean[m][k_pos])
            all_records.append(rec)

        if tgt in wanted_case_targets:
            lfer_atoms = _lfer_atom_matrix(feats)
            env_counters = precompute_mol_env_counters(feats.atom_ptr, feats.atom_env_hash)
            d_mat = compute_local_pair_deltas(
                ds.test_idx, nearest_tr, feats.atom_ptr, feats.atom_env_hash, lfer_atoms, env_counters=env_counters
            )
            # lfer_atoms cols [1,2,3] = [crip_logp, crip_mr, tpsa_c]; sum_diff at [1,2,3]
            case_candidates[tgt] = {
                "ds": ds,
                "s_max": s_max,
                "nearest_tr": nearest_tr,
                "y_te": y_te,
                "y_anc": y_anc,
                "dy_true": dy_true,
                "abs_dy": abs_dy,
                "preds": preds_mean,
                "d_logp": d_mat[:, 1],
                "d_mr": d_mat[:, 2],
                "d_tpsa": d_mat[:, 3],
                "w_film": r["w_star"]["film"],
            }

    df_rec = pd.DataFrame(all_records)
    bins_def = [
        ("< 0.5", lambda x: x < 0.5),
        ("[0.5, 1.0)", lambda x: (x >= 0.5) & (x < 1.0)),
        ("[1.0, 1.5)", lambda x: (x >= 1.0) & (x < 1.5)),
        (">= 1.5", lambda x: x >= 1.5),
        (">= 2.0", lambda x: x >= 2.0),
    ]

    def _compute_strat(sub_df: pd.DataFrame) -> list[dict[str, Any]]:
        rows_out = []
        for blbl, pred_fn in bins_def:
            bdf = sub_df[pred_fn(sub_df["abs_dy"])]
            yt = bdf["y_true"].to_numpy()
            row: dict[str, Any] = {"bin": blbl, "n_mols": int(len(bdf)), "n_targets": int(bdf["target"].nunique())}
            for m in eval_models:
                yp = bdf[m].to_numpy()
                row[f"{m}_pooled_rmse"] = compute_rmse(yt, yp)
                row[f"{m}_pooled_mae"] = compute_mae(yt, yp)
                # Per-target mean RMSE (across targets with >=1 sample in bin)
                tgt_rmses = [
                    compute_rmse(grp["y_true"].to_numpy(), grp[m].to_numpy())
                    for _, grp in bdf.groupby("target")
                ]
                row[f"{m}_tgtmean_rmse"] = float(np.mean(tgt_rmses)) if tgt_rmses else float("nan")
            rows_out.append(row)
        return rows_out

    strat_all = _compute_strat(df_rec)
    strat_analogs = _compute_strat(df_rec[df_rec["s_max"] >= 0.70])

    # Cliff-direction accuracy on steep structural analogs (s_max >= 0.70 & |dy_true| >= thresh)
    dir_rows = []
    for thresh in (1.0, 1.5):
        sdf = df_rec[(df_rec["s_max"] >= 0.70) & (df_rec["abs_dy"] >= thresh)]
        true_sign = np.sign(sdf["dy_true"].to_numpy())
        drow: dict[str, Any] = {"threshold": f"s_max>=0.70 & |dy|>={thresh:.1f}", "n_analogs": int(len(sdf))}
        for m in eval_models:
            pred_sign = np.sign(sdf[m].to_numpy() - sdf["y_anc"].to_numpy())
            drow[m] = float(np.mean(pred_sign == true_sign)) if len(sdf) > 0 else float("nan")
        dir_rows.append(drow)

    # Part III: Extract 4 diverse WIN cases and 2 honest FAIL cases
    cases_out: list[dict[str, Any]] = []
    for fam, pool in win_targets_pool.items():
        best_case = None
        for tgt in pool:
            cinfo = case_candidates[tgt]
            for min_dy in (1.5, 1.0):
                mask = (cinfo["s_max"] >= 0.75) & (np.abs(cinfo["dy_true"]) >= min_dy)
                if not np.any(mask):
                    continue
                idxs = np.where(mask)[0]
                pr = cinfo["preds"]
                e_opt = np.abs(cinfo["y_te"][idxs] - pr["E_Multi_FiLM_OptW"][idxs])
                e_rbf = np.abs(cinfo["y_te"][idxs] - pr["K_RBF_SVM"][idxs])
                e_mul = np.abs(cinfo["y_te"][idxs] - pr["K_Multi_SVR"][idxs])
                e_mlp = np.abs(cinfo["y_te"][idxs] - pr["E_Multi_DirectMLP_OptW"][idxs])
                margins = np.minimum(np.minimum(e_rbf, e_mul), e_mlp) - e_opt
                best_pos = int(idxs[int(np.argmax(margins))])
                best_margin = float(np.max(margins))
                if best_margin > 0.0:
                    best_case = (tgt, best_pos, best_margin)
                    break
            if best_case is not None:
                break
        if best_case is not None:
            tgt, pos, margin = best_case
            cinfo = case_candidates[tgt]
            ds = cinfo["ds"]
            g_te = int(ds.test_idx[pos])
            g_anc = int(cinfo["nearest_tr"][pos])
            pr = cinfo["preds"]
            cases_out.append(
                {
                    "role": f"WIN ({fam})",
                    "target": tgt,
                    "w_film": round(float(cinfo["w_film"]), 3),
                    "test_smiles": ds.smiles[g_te],
                    "anchor_smiles": ds.smiles[g_anc],
                    "s_max": round(float(cinfo["s_max"][pos]), 3),
                    "y_true": round(float(cinfo["y_te"][pos]), 3),
                    "y_anchor": round(float(cinfo["y_anc"][pos]), 3),
                    "dy_true": round(float(cinfo["dy_true"][pos]), 3),
                    "d_logp": round(float(cinfo["d_logp"][pos]), 3),
                    "d_mr": round(float(cinfo["d_mr"][pos]), 3),
                    "d_tpsa": round(float(cinfo["d_tpsa"][pos]), 2),
                    "K_RBF_SVM": round(float(pr["K_RBF_SVM"][pos]), 3),
                    "K_Multi_SVR": round(float(pr["K_Multi_SVR"][pos]), 3),
                    "E_Multi_DirectMLP_OptW": round(float(pr["E_Multi_DirectMLP_OptW"][pos]), 3),
                    "E_Multi_FiLM": round(float(pr["E_Multi_FiLM"][pos]), 3),
                    "E_Multi_FiLM_OptW": round(float(pr["E_Multi_FiLM_OptW"][pos]), 3),
                    "margin_vs_best_ctrl": round(margin, 3),
                }
            )

    for tgt in fail_targets:
        cinfo = case_candidates[tgt]
        ds = cinfo["ds"]
        cmask = ds.cliff_mol[ds.test_idx] == 1
        cand_mask = cmask & (cinfo["s_max"] >= 0.65)
        if not np.any(cand_mask):
            cand_mask = cmask
        idxs = np.where(cand_mask)[0]
        pr = cinfo["preds"]
        e_opt = np.abs(cinfo["y_te"][idxs] - pr["E_Multi_FiLM_OptW"][idxs])
        e_mul = np.abs(cinfo["y_te"][idxs] - pr["K_Multi_SVR"][idxs])
        e_rbf = np.abs(cinfo["y_te"][idxs] - pr["K_RBF_SVM"][idxs])
        loss_margin = e_opt - np.minimum(e_mul, e_rbf)
        pos = int(idxs[int(np.argmax(loss_margin))])
        g_te = int(ds.test_idx[pos])
        g_anc = int(cinfo["nearest_tr"][pos])
        cases_out.append(
            {
                "role": "FAIL (n<1000 Honest Case)",
                "target": tgt,
                "w_film": round(float(cinfo["w_film"]), 3),
                "test_smiles": ds.smiles[g_te],
                "anchor_smiles": ds.smiles[g_anc],
                "s_max": round(float(cinfo["s_max"][pos]), 3),
                "y_true": round(float(cinfo["y_te"][pos]), 3),
                "y_anchor": round(float(cinfo["y_anc"][pos]), 3),
                "dy_true": round(float(cinfo["dy_true"][pos]), 3),
                "d_logp": round(float(cinfo["d_logp"][pos]), 3),
                "d_mr": round(float(cinfo["d_mr"][pos]), 3),
                "d_tpsa": round(float(cinfo["d_tpsa"][pos]), 2),
                "K_RBF_SVM": round(float(pr["K_RBF_SVM"][pos]), 3),
                "K_Multi_SVR": round(float(pr["K_Multi_SVR"][pos]), 3),
                "E_Multi_DirectMLP_OptW": round(float(pr["E_Multi_DirectMLP_OptW"][pos]), 3),
                "E_Multi_FiLM": round(float(pr["E_Multi_FiLM"][pos]), 3),
                "E_Multi_FiLM_OptW": round(float(pr["E_Multi_FiLM_OptW"][pos]), 3),
                "margin_vs_best_ctrl": round(float(-np.max(loss_margin)), 3),
            }
        )

    return {"eval_models": eval_models, "strat_all": strat_all, "strat_analogs": strat_analogs, "dir_rows": dir_rows}, cases_out


def _plot_step_a(
    df_long: pd.DataFrame,
    df_wide: pd.DataFrame,
    anatomy: dict[str, Any],
    fig_path: Path,
) -> None:
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18.5, 5.8))

    # Panel A: 30-Target Mean Cliff RMSE & Overall RMSE
    bar_models = [
        ("A1_Direct_MLP", "DirectMLP", "#BBBBBB"),
        ("B1_CatBoost", "CatBoost", "#A6CEE3"),
        ("K_RBF_SVM", "RBF-SVM", "#777777"),
        ("K_Multi_SVR", "MultiKernel-SVR", "#0072B2"),
        ("E_Multi_DirectMLP_OptW", "Multi+MLP(OptW)", "#CC79A7"),
        ("E_Multi_CatBoost_OptW", "Multi+Cat(OptW)", "#8C6BB1"),
        ("E_Multi_FiLM", "Multi+FiLM(w=0.5)", "#E69F00"),
        ("E_Multi_FiLM_PriorN", "Multi+FiLM(PriorN)", "#009E73"),
        ("E_Multi_FiLM_OptW", "Multi+FiLM(OptW)", "#D55E00"),
    ]
    agg = df_long.groupby("model")[["cliff_rmse_mean", "rmse_mean"]].mean()
    x = np.arange(len(bar_models))
    w = 0.38
    cvals = [float(agg.loc[m, "cliff_rmse_mean"]) for m, _, _ in bar_models]
    ovals = [float(agg.loc[m, "rmse_mean"]) for m, _, _ in bar_models]
    cols = [c for _, _, c in bar_models]
    lbls = [l for _, l, _ in bar_models]

    ax1.bar(x - w / 2, cvals, width=w, color=cols, edgecolor="black", linewidth=0.5, label="Cliff RMSE")
    ax1.bar(x + w / 2, ovals, width=w, color=cols, alpha=0.45, edgecolor="black", linewidth=0.5, hatch="//", label="Overall RMSE")
    for xi, cv in zip(x, cvals):
        ax1.text(xi - w / 2, cv + 0.002, f"{cv:.4f}", ha="center", va="bottom", fontsize=6.8, rotation=45)
    ax1.set_xticks(x)
    ax1.set_xticklabels(lbls, rotation=28, ha="right", fontsize=7.5)
    ax1.set_ylim(0.55, max(cvals) + 0.04)
    ax1.set_ylabel("30-Target Mean RMSE (Lower = Better)")
    ax1.set_title("A. Leak-Free OptW & PriorN Comparison (N=30)", fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.legend(frameon=False, fontsize=7.5, loc="upper right")

    # Panel B: Learned Bates-Granger w*_film vs n_total
    rho_w, p_w = stats.spearmanr(df_wide["n_total"], df_wide["w_film"])
    for assay, col, mk in [("Ki", "#0072B2", "o"), ("EC50", "#D55E00", "^")]:
        sub = df_wide[df_wide["assay_type"] == assay]
        ax2.scatter(sub["n_total"], sub["w_film"], color=col, marker=mk, s=55, edgecolor="black", linewidth=0.6, label=f"{assay} (N={len(sub)})")
    ax2.axvline(1000, color="gray", linestyle="--", linewidth=0.9, label="n = 1000 Boundary")
    n_grid = np.linspace(float(df_wide["n_total"].min()), float(df_wide["n_total"].max()), 100)
    ntr_grid = n_grid * 0.8
    ax2.plot(n_grid, 0.5 * ntr_grid / (ntr_grid + 1000.0), "k:", linewidth=1.2, label="w_prior(n)")
    ax2.set_xlabel("Dataset Size (n_total)")
    ax2.set_ylabel("Learned Nested-OOF Bates-Granger Weight w*_FiLM")
    ax2.set_title(f"B. Learned w*_FiLM vs n_total (rho={rho_w:+.3f}, p={p_w:.3f})", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(frameon=False, fontsize=7.5, loc="lower right")

    # Panel C: Error Curves Across |Delta pKi| Potency-Jump Bins (s_max >= 0.70 Analogs)
    sa = anatomy["strat_analogs"][:4]  # <0.5, [0.5,1.0), [1.0,1.5), >=1.5
    xb = np.arange(len(sa))
    xlbls = [f"{row['bin']}\n(n={row['n_mols']})" for row in sa]
    curve_specs = [
        ("K_RBF_SVM", "K_RBF_SVM", "#777777", "--o"),
        ("K_Multi_SVR", "K_Multi_SVR", "#0072B2", "--s"),
        ("E_Multi_DirectMLP_OptW", "Multi+MLP(OptW)", "#CC79A7", "-.^"),
        ("E_Multi_FiLM_OptW", "Multi+FiLM(OptW)", "#D55E00", "-D"),
    ]
    for m_key, lbl, col, fmt in curve_specs:
        ys = [row[f"{m_key}_pooled_rmse"] for row in sa]
        ax3.plot(xb, ys, fmt, color=col, linewidth=1.8, markersize=5, label=lbl)
    ax3.set_xticks(xb)
    ax3.set_xticklabels(xlbls, fontsize=7.8)
    ax3.set_xlabel("|Delta y| to Nearest Training Analog (s_max >= 0.70)")
    ax3.set_ylabel("Pooled RMSE")
    ax3.set_title("C. Potency-Jump Gradient on Structural Analogs", fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.3)
    ax3.legend(frameon=False, fontsize=7.5, loc="upper left")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def _generate_step_a_report(
    results: list[dict[str, Any]],
    anatomy: dict[str, Any],
    cases_out: list[dict[str, Any]],
    total_elapsed: float,
) -> None:
    out_dir = _out_root()
    fig_dir = _fig_dir()

    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []
    for r in results:
        tgt = r["target"]
        wrow: dict[str, Any] = {
            "target": tgt,
            "assay_type": r["assay_type"],
            "n_total": r["n_total"],
            "n_train": r["n_train"],
            "n_test": r["n_test"],
            "n_test_cliff": r["n_test_cliff"],
            "w_film": r["w_star"]["film"],
            "w_a2": r["w_star"]["a2"],
            "w_mlp": r["w_star"]["mlp"],
            "w_cat": r["w_star"]["cat"],
            "w_prior": r["w_prior"],
        }
        for m_name in STEP_A_MODELS:
            mm = r["metrics"][m_name]
            long_rows.append({"target": tgt, "assay_type": r["assay_type"], "n_total": r["n_total"], "model": m_name, **mm})
            wrow[f"{m_name}_cliff"] = mm["cliff_rmse_mean"]
            wrow[f"{m_name}_rmse"] = mm["rmse_mean"]
            wrow[f"{m_name}_noncliff"] = mm["non_cliff_rmse_mean"]
            wrow[f"{m_name}_spearman"] = mm["spearman_mean"]
        wide_rows.append(wrow)

    df_long = pd.DataFrame(long_rows)
    df_wide = pd.DataFrame(wide_rows)
    csv_path = out_dir / "step_a_summary.csv"
    wide_csv = out_dir / "step_a_wide.csv"
    df_long.to_csv(csv_path, index=False)
    df_wide.to_csv(wide_csv, index=False)

    fig_path = fig_dir / "jcim_step_a_optw_and_anatomy.png"
    _plot_step_a(df_long, df_wide, anatomy, fig_path)

    agg = df_long.groupby("model")[["rmse_mean", "cliff_rmse_mean", "non_cliff_rmse_mean", "cliff_mae_mean", "spearman_mean"]].mean()
    agg_sorted = agg.sort_values("cliff_rmse_mean")

    md_path = ROOT_DIR / "refine-logs" / "JCIM_STEP_A_ANALYSIS.md"
    lines: list[str] = [
        "# JCIM Step A: Leak-Free Nested-OOF Bates-Granger Weighting, |ΔpKi| Anatomy & Case Studies",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **Targets**: {len(results)} | **Elapsed**: {total_elapsed:.1f}s",
        "",
        "## 1. 30-Target Leaderboard with Leak-Free `OptW` and `PriorN` (Sorted by Cliff RMSE)",
        "",
        "| Model | Overall RMSE | Cliff RMSE | Non-Cliff RMSE | Cliff MAE | Spearman |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for m_name, row in agg_sorted.iterrows():
        lines.append(
            f"| `{m_name}` | {row['rmse_mean']:.4f} | {row['cliff_rmse_mean']:.4f} | "
            f"{row['non_cliff_rmse_mean']:.4f} | {row['cliff_mae_mean']:.4f} | {row['spearman_mean']:.4f} |"
        )

    lines.extend([
        "",
        "## 2. Cross-Target Wilcoxon & 95% Bootstrap CI (`E_Multi_FiLM_OptW` & `PriorN`, N=30)",
        "",
        "### A. Activity Cliff RMSE Comparisons",
        "",
        "| Comparison | Win / Tie / Loss | Mean Diff | Median Diff | 95% Bootstrap CI | Two-Sided p |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for lbl, ca, cb in [
        ("OptW_FiLM vs K_RBF_SVM", "E_Multi_FiLM_OptW_cliff", "K_RBF_SVM_cliff"),
        ("OptW_FiLM vs K_Multi_SVR", "E_Multi_FiLM_OptW_cliff", "K_Multi_SVR_cliff"),
        ("OptW_FiLM vs OptW_DirectMLP", "E_Multi_FiLM_OptW_cliff", "E_Multi_DirectMLP_OptW_cliff"),
        ("OptW_FiLM vs OptW_CatBoost", "E_Multi_FiLM_OptW_cliff", "E_Multi_CatBoost_OptW_cliff"),
        ("OptW_FiLM vs E_Multi_FiLM(0.5)", "E_Multi_FiLM_OptW_cliff", "E_Multi_FiLM_cliff"),
        ("OptW_FiLM vs OptW_A2", "E_Multi_FiLM_OptW_cliff", "E_Multi_A2_OptW_cliff"),
        ("PriorN_FiLM vs K_RBF_SVM", "E_Multi_FiLM_PriorN_cliff", "K_RBF_SVM_cliff"),
        ("PriorN_FiLM vs K_Multi_SVR", "E_Multi_FiLM_PriorN_cliff", "K_Multi_SVR_cliff"),
        ("PriorN_FiLM vs PriorN_DirectMLP", "E_Multi_FiLM_PriorN_cliff", "E_Multi_DirectMLP_PriorN_cliff"),
        ("PriorN_FiLM vs PriorN_CatBoost", "E_Multi_FiLM_PriorN_cliff", "E_Multi_CatBoost_PriorN_cliff"),
    ]:
        st = _cross_target_stats(df_wide, ca, cb)
        lines.append(
            f"| {lbl} | {st['wins']} / {st['ties']} / {st['losses']} | "
            f"{st['mean_diff']:+.5f} | {st['median_diff']:+.5f} | "
            f"[{st['ci95_lo']:+.5f}, {st['ci95_hi']:+.5f}] | `{st['p_value']:.4e}` |"
        )

    lines.extend([
        "",
        "### B. Overall RMSE Comparisons (N=30)",
        "",
        "| Comparison | Win / Tie / Loss | Mean Diff | Median Diff | 95% Bootstrap CI | Two-Sided p |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for lbl, ca, cb in [
        ("OptW_FiLM vs K_RBF_SVM", "E_Multi_FiLM_OptW_rmse", "K_RBF_SVM_rmse"),
        ("OptW_FiLM vs K_Multi_SVR", "E_Multi_FiLM_OptW_rmse", "K_Multi_SVR_rmse"),
        ("OptW_FiLM vs OptW_DirectMLP", "E_Multi_FiLM_OptW_rmse", "E_Multi_DirectMLP_OptW_rmse"),
        ("OptW_FiLM vs OptW_CatBoost", "E_Multi_FiLM_OptW_rmse", "E_Multi_CatBoost_OptW_rmse"),
        ("OptW_FiLM vs E_Multi_FiLM(0.5)", "E_Multi_FiLM_OptW_rmse", "E_Multi_FiLM_rmse"),
        ("PriorN_FiLM vs K_RBF_SVM", "E_Multi_FiLM_PriorN_rmse", "K_RBF_SVM_rmse"),
        ("PriorN_FiLM vs K_Multi_SVR", "E_Multi_FiLM_PriorN_rmse", "K_Multi_SVR_rmse"),
        ("PriorN_FiLM vs PriorN_DirectMLP", "E_Multi_FiLM_PriorN_rmse", "E_Multi_DirectMLP_PriorN_rmse"),
    ]:
        st = _cross_target_stats(df_wide, ca, cb)
        lines.append(
            f"| {lbl} | {st['wins']} / {st['ties']} / {st['losses']} | "
            f"{st['mean_diff']:+.5f} | {st['median_diff']:+.5f} | "
            f"[{st['ci95_lo']:+.5f}, {st['ci95_hi']:+.5f}] | `{st['p_value']:.4e}` |"
        )

    lines.extend([
        "",
        "## 3. Sample-Size Regime Check (`n < 1000` vs `n >= 1000`)",
        "",
        "| Regime | N | Mean w*_FiLM | K_RBF_SVM Cliff | K_Multi_SVR Cliff | OptW_MLP Cliff | E_Multi_FiLM(0.5) Cliff | E_Multi_FiLM_PriorN Cliff | E_Multi_FiLM_OptW Cliff |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r_lbl, sub in [("n < 1000", df_wide[df_wide["n_total"] < 1000]), ("n >= 1000", df_wide[df_wide["n_total"] >= 1000])]:
        lines.append(
            f"| `{r_lbl}` | {len(sub)} | {sub['w_film'].mean():.3f} | "
            f"{sub['K_RBF_SVM_cliff'].mean():.4f} | {sub['K_Multi_SVR_cliff'].mean():.4f} | "
            f"{sub['E_Multi_DirectMLP_OptW_cliff'].mean():.4f} | {sub['E_Multi_FiLM_cliff'].mean():.4f} | "
            f"{sub['E_Multi_FiLM_PriorN_cliff'].mean():.4f} | {sub['E_Multi_FiLM_OptW_cliff'].mean():.4f} |"
        )

    rho_w, p_w = stats.spearmanr(df_wide["n_total"], df_wide["w_film"])
    lines.append(f"\nSpearman correlation (`n_total` vs learned `w*_FiLM`): `rho = {rho_w:+.3f}` (`p = {p_w:.4e}`).")

    lines.extend([
        "",
        "## 4. `|ΔpKi|` Potency-Jump Gradient Anatomy & Cliff-Direction Accuracy",
        "",
        "### A. High-Similarity Structural Analogs (`s_max >= 0.70`) Stratified by `|y_test - y_nearest_train|`",
        "",
        "| `|Δy|` Bin | N Mols (Targets) | RBF_SVM | Multi_SVR | CatBoost | DirectMLP | Multi+MLP_OptW | Multi+FiLM(0.5) | Multi+FiLM_OptW (Pooled / TgtMean) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for row in anatomy["strat_analogs"]:
        lines.append(
            f"| `{row['bin']}` | {row['n_mols']} ({row['n_targets']}) | "
            f"{row['K_RBF_SVM_pooled_rmse']:.4f} | {row['K_Multi_SVR_pooled_rmse']:.4f} | "
            f"{row['B1_CatBoost_pooled_rmse']:.4f} | {row['A1_Direct_MLP_pooled_rmse']:.4f} | "
            f"{row['E_Multi_DirectMLP_OptW_pooled_rmse']:.4f} | {row['E_Multi_FiLM_pooled_rmse']:.4f} | "
            f"{row['E_Multi_FiLM_OptW_pooled_rmse']:.4f} / {row['E_Multi_FiLM_OptW_tgtmean_rmse']:.4f} |"
        )

    lines.extend([
        "",
        "### B. All Test Molecules Stratified by `|y_test - y_nearest_train|`",
        "",
        "| `|Δy|` Bin | N Mols (Targets) | RBF_SVM | Multi_SVR | Multi+MLP_OptW | Multi+Cat_OptW | Multi+FiLM(0.5) | Multi+FiLM_OptW (Pooled / TgtMean) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for row in anatomy["strat_all"]:
        lines.append(
            f"| `{row['bin']}` | {row['n_mols']} ({row['n_targets']}) | "
            f"{row['K_RBF_SVM_pooled_rmse']:.4f} | {row['K_Multi_SVR_pooled_rmse']:.4f} | "
            f"{row['E_Multi_DirectMLP_OptW_pooled_rmse']:.4f} | {row['E_Multi_CatBoost_OptW_pooled_rmse']:.4f} | "
            f"{row['E_Multi_FiLM_pooled_rmse']:.4f} | "
            f"{row['E_Multi_FiLM_OptW_pooled_rmse']:.4f} / {row['E_Multi_FiLM_OptW_tgtmean_rmse']:.4f} |"
        )

    lines.extend([
        "",
        "### C. Cliff-Direction Accuracy (`sign(y_hat - y_anchor) == sign(y_true - y_anchor)`)",
        "",
        "| Subset | N Analogs | RBF_SVM | Multi_SVR | CatBoost | DirectMLP | Multi+MLP_OptW | Multi+Cat_OptW | Multi+FiLM(0.5) | Multi+FiLM_OptW |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for dr in anatomy["dir_rows"]:
        lines.append(
            f"| `{dr['threshold']}` | {dr['n_analogs']} | "
            f"{dr['K_RBF_SVM']:.3%} | {dr['K_Multi_SVR']:.3%} | {dr['B1_CatBoost']:.3%} | "
            f"{dr['A1_Direct_MLP']:.3%} | {dr['E_Multi_DirectMLP_OptW']:.3%} | "
            f"{dr['E_Multi_CatBoost_OptW']:.3%} | {dr['E_Multi_FiLM']:.3%} | {dr['E_Multi_FiLM_OptW']:.3%} |"
        )

    lines.extend([
        "",
        "## 5. Verified Medicinal-Chemistry Case Studies (4 Diverse WIN + 2 Honest n<1000 FAIL)",
        "",
    ])
    for c in cases_out:
        lines.extend([
            f"### {c['role']} — `{c['target']}` (`w*_FiLM = {c['w_film']}`)",
            f"- **Test SMILES**: `{c['test_smiles']}`",
            f"- **Nearest Anchor SMILES** (`s_max = {c['s_max']}`): `{c['anchor_smiles']}`",
            f"- **Ground Truth**: `y_true = {c['y_true']}`, `y_anchor = {c['y_anchor']}` -> `Δy_true = {c['dy_true']:+.3f}`",
            f"- **Local 2D-LFER Substituent Deltas (sum_diff)**: `ΔCrippen_LogP = {c['d_logp']:+.3f}`, `ΔCrippen_MR = {c['d_mr']:+.3f}`, `ΔTPSA = {c['d_tpsa']:+.2f}`",
            f"- **Predictions**: `K_RBF_SVM = {c['K_RBF_SVM']}` | `K_Multi_SVR = {c['K_Multi_SVR']}` | `E_Multi_DirectMLP_OptW = {c['E_Multi_DirectMLP_OptW']}` | `E_Multi_FiLM(0.5) = {c['E_Multi_FiLM']}` | **`E_Multi_FiLM_OptW = {c['E_Multi_FiLM_OptW']}`** (margin vs best ctrl: `{c['margin_vs_best_ctrl']:+.3f}`)",
            "",
        ])

    lines.extend([
        f"Summary CSV: `{csv_path.as_posix()}`",
        f"Wide CSV: `{wide_csv.as_posix()}`",
        f"Figure: `{fig_path.as_posix()}`",
    ])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved {md_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step A OptW + |Delta pKi| Gradient + Case Studies.")
    parser.add_argument("--targets", type=str, default="", help="Optional comma-separated targets.")
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--no-save", action="store_true", default=False)
    args = parser.parse_args()

    target_list = [t.strip() for t in args.targets.split(",") if t.strip()] if args.targets.strip() else list(ALL_30_TARGETS)
    save_cache = not args.no_save

    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] Starting Step A ({len(target_list)} targets, save_cache={save_cache})",
        flush=True,
    )
    t0 = time.time()
    results = [_evaluate_target_step_a(tgt, force=args.force, save_cache=save_cache) for tgt in target_list]
    elapsed = time.time() - t0

    if save_cache and len(results) == len(ALL_30_TARGETS):
        anatomy, cases_out = _compute_potency_gradient_and_cases(results)
        _generate_step_a_report(results, anatomy, cases_out, total_elapsed=elapsed)
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Partial/smoke Step A completed in {elapsed:.1f}s.", flush=True)


if __name__ == "__main__":
    main()

"""Week-1 diagnostic audit (Part A) and physical-constraint / base-architecture probes (Part B)."""

from __future__ import annotations

import copy
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
import torch
import torch.nn as nn
from scipy import optimize, stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from physchem_delta.data import MVP_TARGETS, load_target_dataset
from physchem_delta.featurize import (
    TargetFeatures,
    compute_local_pair_deltas,
    extract_or_load_target_features,
    precompute_mol_env_counters,
)
from physchem_delta.metrics import evaluate_predictions, paired_wilcoxon
from physchem_delta.models import (
    AntisymmetricPairwiseDeltaNet,
    _select_topk_anchors,
    build_tier_matrices,
    compute_tanimoto_matrix,
    get_device,
    make_inner_split,
    set_global_seed,
)

SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)


def _refine_dir() -> Path:
    return ROOT_DIR / "refine-logs"


def _run_dir(target: str) -> Path:
    return _refine_dir() / "runs" / target


def _fig_dir() -> Path:
    d = _refine_dir() / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Part A: Zero-Retrain Diagnostic Analysis on Week-1 Cached Predictions
# ---------------------------------------------------------------------------

def run_part_a_diagnostics(
    datasets: dict[str, Any],
    features: dict[str, TargetFeatures],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, Any]]], pd.DataFrame]:
    """Analyze chemical flexibility profiles, stratified degradation, and concrete cliff failures."""
    profile_rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []
    strat_rows_all: list[dict[str, Any]] = []
    top_cases: dict[str, list[dict[str, Any]]] = {}
    flex_plot_rows: list[dict[str, Any]] = []

    for tgt in MVP_TARGETS:
        ds = datasets[tgt]
        feats = features[tgt]
        rdir = _run_dir(tgt)

        rotb = feats.desc2d[:, 5]
        molwt = feats.desc2d[:, 0]
        fsp3 = feats.desc2d[:, 7]
        tpsa = feats.desc2d[:, 2]
        espan = feats.feat_3d[:, 1]
        chiral_cnt = feats.feat_3d[:, 12]

        sim_te_tr = compute_tanimoto_matrix(feats.ecfp_chiral[ds.test_idx], feats.ecfp_chiral[ds.train_idx])
        s_max_te = np.max(sim_te_tr, axis=1)
        nearest_tr_pos = np.argmax(sim_te_tr, axis=1)
        nearest_tr_global = ds.train_idx[nearest_tr_pos]

        profile_rows.append(
            {
                "target": tgt,
                "n_total": len(ds.smiles),
                "rotb_mean": round(float(np.mean(rotb)), 2),
                "rotb_median": round(float(np.median(rotb)), 1),
                "rotb_gt7_pct": round(100.0 * float(np.mean(rotb >= 8)), 1),
                "molwt_mean": round(float(np.mean(molwt)), 1),
                "fsp3_mean": round(float(np.mean(fsp3)), 3),
                "tpsa_mean": round(float(np.mean(tpsa)), 1),
                "espan_median": round(float(np.median(espan)), 2),
                "chiral_pct": round(100.0 * float(np.mean(chiral_cnt > 0)), 1),
                "smax_test_median": round(float(np.median(s_max_te)), 3),
            }
        )

        # Load Week-1 5-seed predictions for A2, A3, A4, M, B1c_SVM
        seed_frames = []
        for s in SEEDS:
            d_a2 = pd.read_csv(rdir / f"A2_2D_Pairwise_Delta_seed{s}_preds.csv")
            d_a3 = pd.read_csv(rdir / f"A3_2D3D_Pairwise_Delta_seed{s}_preds.csv")
            d_a4 = pd.read_csv(rdir / f"A4_PhysChem_Delta_seed{s}_preds.csv")
            d_m = pd.read_csv(rdir / f"M_AnchorGated_PhysChem_Delta_seed{s}_preds.csv")
            d_sv = pd.read_csv(rdir / f"B1c_ECFP_SVM_seed{s}_preds.csv")

            df_s = pd.DataFrame(
                {
                    "target": tgt,
                    "seed": s,
                    "test_idx": ds.test_idx,
                    "y_true": d_a2["y_true"].to_numpy(),
                    "cliff_mol": d_a2["cliff_mol"].to_numpy(dtype=int),
                    "s_max": s_max_te,
                    "rotb": rotb[ds.test_idx],
                    "molwt": molwt[ds.test_idx],
                    "espan": espan[ds.test_idx],
                    "chiral_cnt": chiral_cnt[ds.test_idx],
                    "pred_a2": d_a2["y_pred"].to_numpy(),
                    "pred_a3": d_a3["y_pred"].to_numpy(),
                    "pred_a4": d_a4["y_pred"].to_numpy(),
                    "pred_m": d_m["y_pred"].to_numpy(),
                    "pred_svm": d_sv["y_pred"].to_numpy(),
                }
            )
            seed_frames.append(df_s)

        pooled = pd.concat(seed_frames, ignore_index=True)
        pooled["abs_a2"] = np.abs(pooled["y_true"] - pooled["pred_a2"])
        pooled["abs_a3"] = np.abs(pooled["y_true"] - pooled["pred_a3"])
        pooled["abs_a4"] = np.abs(pooled["y_true"] - pooled["pred_a4"])
        pooled["abs_m"] = np.abs(pooled["y_true"] - pooled["pred_m"])
        pooled["abs_svm"] = np.abs(pooled["y_true"] - pooled["pred_svm"])
        pooled["delta_a3_a2"] = pooled["abs_a3"] - pooled["abs_a2"]
        pooled["delta_a4_a2"] = pooled["abs_a4"] - pooled["abs_a2"]
        pooled["delta_m_svm"] = pooled["abs_m"] - pooled["abs_svm"]

        cliff = pooled[pooled["cliff_mol"] == 1].copy()

        # Spearman correlations on cliff subset
        for feat_col in ("rotb", "molwt", "espan", "s_max"):
            rho42, p42 = stats.spearmanr(cliff[feat_col], cliff["delta_a4_a2"])
            rho_ms, p_ms = stats.spearmanr(cliff[feat_col], cliff["delta_m_svm"])
            corr_rows.append(
                {
                    "target": tgt,
                    "feature": feat_col,
                    "rho_delta_A4_vs_A2": round(float(rho42), 4),
                    "p_A4_vs_A2": float(f"{p42:.3e}"),
                    "rho_delta_M_vs_SVM": round(float(rho_ms), 4),
                    "p_M_vs_SVM": float(f"{p_ms:.3e}"),
                }
            )

        # Stratify by Rotatable Bonds
        cliff["rot_bin"] = np.where(
            cliff["rotb"] <= 4,
            "<=4 (rigid)",
            np.where(cliff["rotb"] <= 7, "5-7 (moderate)", ">=8 (flexible)"),
        )
        for rb in ("<=4 (rigid)", "5-7 (moderate)", ">=8 (flexible)"):
            sub = cliff[cliff["rot_bin"] == rb]
            if len(sub) == 0:
                continue
            rmse_a2 = float(np.sqrt(np.mean((sub["y_true"] - sub["pred_a2"]) ** 2)))
            rmse_a3 = float(np.sqrt(np.mean((sub["y_true"] - sub["pred_a3"]) ** 2)))
            rmse_a4 = float(np.sqrt(np.mean((sub["y_true"] - sub["pred_a4"]) ** 2)))
            rmse_m = float(np.sqrt(np.mean((sub["y_true"] - sub["pred_m"]) ** 2)))
            rmse_svm = float(np.sqrt(np.mean((sub["y_true"] - sub["pred_svm"]) ** 2)))
            strat_rows_all.append(
                {
                    "target": tgt,
                    "stratum": f"RotBonds: {rb}",
                    "n_obs": int(len(sub)),
                    "rmse_a2": round(rmse_a2, 4),
                    "rmse_a3": round(rmse_a3, 4),
                    "rmse_a4": round(rmse_a4, 4),
                    "rmse_m": round(rmse_m, 4),
                    "rmse_svm": round(rmse_svm, 4),
                    "mean_abs_shift_A4_A2": round(float(sub["delta_a4_a2"].mean()), 4),
                }
            )
            flex_plot_rows.append(
                {
                    "target": tgt,
                    "rot_bin": rb,
                    "n_mols": int(len(sub) // len(SEEDS)),
                    "delta_cliff_rmse_a3_a2": rmse_a3 - rmse_a2,
                    "delta_cliff_rmse_a4_a2": rmse_a4 - rmse_a2,
                    "delta_cliff_rmse_m_svm": rmse_m - rmse_svm,
                }
            )

        # Stratify by e_span (median split), chirality, and s_max
        med_es = float(np.median(espan[ds.test_idx]))
        strat_defs = [
            ("e_span <= med", cliff["espan"] <= med_es),
            ("e_span > med", cliff["espan"] > med_es),
            ("chiral == 0", cliff["chiral_cnt"] == 0),
            ("chiral >= 1", cliff["chiral_cnt"] >= 1),
            ("s_max < 0.70", cliff["s_max"] < 0.70),
            ("s_max [0.70,0.85)", (cliff["s_max"] >= 0.70) & (cliff["s_max"] < 0.85)),
            ("s_max >= 0.85", cliff["s_max"] >= 0.85),
        ]
        for label, mask in strat_defs:
            sub = cliff[mask]
            if len(sub) == 0:
                continue
            strat_rows_all.append(
                {
                    "target": tgt,
                    "stratum": label,
                    "n_obs": int(len(sub)),
                    "rmse_a2": round(float(np.sqrt(np.mean((sub["y_true"] - sub["pred_a2"]) ** 2))), 4),
                    "rmse_a3": round(float(np.sqrt(np.mean((sub["y_true"] - sub["pred_a3"]) ** 2))), 4),
                    "rmse_a4": round(float(np.sqrt(np.mean((sub["y_true"] - sub["pred_a4"]) ** 2))), 4),
                    "rmse_m": round(float(np.sqrt(np.mean((sub["y_true"] - sub["pred_m"]) ** 2))), 4),
                    "rmse_svm": round(float(np.sqrt(np.mean((sub["y_true"] - sub["pred_svm"]) ** 2))), 4),
                    "mean_abs_shift_A4_A2": round(float(sub["delta_a4_a2"].mean()), 4),
                }
            )

        # Top-5 worst A4-vs-A2 cliff regressions (averaged across 5 seeds per test molecule)
        if tgt in ("CHEMBL204_Ki", "CHEMBL2147_Ki"):
            mol_avg = (
                cliff.groupby("test_idx", as_index=False)
                .agg(
                    y_true=("y_true", "first"),
                    rotb=("rotb", "first"),
                    espan=("espan", "first"),
                    s_max=("s_max", "first"),
                    pred_svm=("pred_svm", "mean"),
                    pred_a2=("pred_a2", "mean"),
                    pred_a4=("pred_a4", "mean"),
                    delta_a4_a2=("delta_a4_a2", "mean"),
                )
                .sort_values("delta_a4_a2", ascending=False)
            )
            pos_map = {int(g): pos for pos, g in enumerate(ds.test_idx.tolist())}
            cases = []
            for _, row in mol_avg.head(5).iterrows():
                g_idx = int(row["test_idx"])
                t_pos = pos_map[g_idx]
                anc_g = int(nearest_tr_global[t_pos])
                cases.append(
                    {
                        "test_idx": g_idx,
                        "smiles": ds.smiles[g_idx],
                        "rotb": int(row["rotb"]),
                        "espan": round(float(row["espan"]), 2),
                        "s_max": round(float(row["s_max"]), 3),
                        "y_true": round(float(row["y_true"]), 3),
                        "pred_svm": round(float(row["pred_svm"]), 3),
                        "pred_a2": round(float(row["pred_a2"]), 3),
                        "pred_a4": round(float(row["pred_a4"]), 3),
                        "shift_a4_minus_a2": round(float(row["delta_a4_a2"]), 3),
                        "anchor_y": round(float(ds.y[anc_g]), 3),
                        "anchor_smiles": ds.smiles[anc_g],
                    }
                )
            top_cases[tgt] = cases

    return profile_rows, corr_rows, top_cases, pd.DataFrame(flex_plot_rows), pd.DataFrame(strat_rows_all)


# ---------------------------------------------------------------------------
# Part B: Controlled 5-Seed Probes (Physical Constraints & Kernel Base Swap)
# ---------------------------------------------------------------------------

def _build_local_physchem_diff(
    feats: TargetFeatures,
    bundle_2d_cont: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    env_counters: list[Any],
    rigidity_attenuate: bool = False,
) -> np.ndarray:
    """
    Build compact local-perturbation delta vector WITHOUT global 3D/electronic MLP bloat:
      - desc2d scaled diff (25 dims)
      - 2D added-vs-removed ECFP bit diff (1 dim)
      - local 3D atom perturbation (sum_diff + mean_diff = 6 dims; optionally scaled by rigidity weight)
      - local electronic atom perturbation (sum_diff + mean_diff = 16 dims, unattenuated!)
      - local modified heavy atom count diff (1 dim)
    Total: 49 compact, strictly antisymmetric features.
    """
    desc_diff = (bundle_2d_cont[i_idx] - bundle_2d_cont[j_idx]).astype(np.float32)

    fp_i = feats.ecfp_chiral[i_idx]
    fp_j = feats.ecfp_chiral[j_idx]
    n_add = np.sum((fp_i > 0.5) & (fp_j < 0.5), axis=1, keepdims=True).astype(np.float32)
    n_rem = np.sum((fp_j > 0.5) & (fp_i < 0.5), axis=1, keepdims=True).astype(np.float32)
    fp_diff = n_add - n_rem

    d3 = feats.atom_3d_props.shape[1]
    de = feats.atom_elec_props.shape[1]
    raw_3d = compute_local_pair_deltas(
        i_idx, j_idx, feats.atom_ptr, feats.atom_env_hash, feats.atom_3d_props, env_counters=env_counters
    )
    local_3d = raw_3d[:, : 2 * d3]  # sum_diff + mean_diff
    atom_cnt_diff = raw_3d[:, 3 * d3 : 3 * d3 + 1]

    if rigidity_attenuate:
        rotb = feats.desc2d[:, 5]
        max_rot = np.maximum(rotb[i_idx], rotb[j_idx]).astype(np.float32)
        w_rigid = np.exp(-max_rot / 6.0)[:, None]
        local_3d = (local_3d * w_rigid).astype(np.float32)

    raw_el = compute_local_pair_deltas(
        i_idx, j_idx, feats.atom_ptr, feats.atom_env_hash, feats.atom_elec_props, env_counters=env_counters
    )
    local_el = raw_el[:, : 2 * de]  # sum_diff + mean_diff

    return np.concatenate([desc_diff, fp_diff, local_3d, local_el, atom_cnt_diff], axis=1).astype(np.float32)


def _predict_compact_delta(
    model: AntisymmetricPairwiseDeltaNet,
    feats: TargetFeatures,
    bundle_2d: Any,
    query_idx: np.ndarray,
    anchor_pool_idx: np.ndarray,
    y: np.ndarray,
    y_std: float,
    diff_scale: np.ndarray,
    env_counters: list[Any],
    rigidity_attenuate: bool,
    device: torch.device,
    k_eval: int = 10,
    tau: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    sim_mat = compute_tanimoto_matrix(feats.ecfp_chiral[query_idx], feats.ecfp_chiral[anchor_pool_idx])
    anchor_g_idx, anchor_sims = _select_topk_anchors(sim_mat, anchor_pool_idx, k=k_eval)
    s_max = anchor_sims[:, 0].copy()

    n_q, k_eff = anchor_g_idx.shape
    q_flat = np.repeat(query_idx, k_eff)
    a_flat = anchor_g_idx.reshape(-1)
    s_flat = anchor_sims.reshape(-1)

    raw_diff = _build_local_physchem_diff(
        feats, bundle_2d.x_cont_scaled, q_flat, a_flat, env_counters, rigidity_attenuate=rigidity_attenuate
    )
    scaled_diff = np.clip(raw_diff / diff_scale, -10.0, 10.0).astype(np.float32)

    model.eval()
    preds_scaled = []
    with torch.no_grad():
        for start in range(0, q_flat.size, 2048):
            sl = slice(start, start + 2048)
            out = model(
                torch.tensor(bundle_2d.x_mol[q_flat[sl]], dtype=torch.float32, device=device),
                torch.tensor(bundle_2d.x_mol[a_flat[sl]], dtype=torch.float32, device=device),
                torch.tensor(scaled_diff[sl], dtype=torch.float32, device=device),
                torch.tensor(s_flat[sl], dtype=torch.float32, device=device),
            ).cpu().numpy()
            preds_scaled.append(out)

    delta_pred = (np.concatenate(preds_scaled, axis=0) * y_std).reshape(n_q, k_eff)
    y_anchors = y[anchor_g_idx]
    logits = tau * anchor_sims
    logits = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    y_hat = np.sum(weights * (y_anchors + delta_pred), axis=1).astype(np.float64)
    return y_hat, s_max.astype(np.float64)


def run_compact_delta_probe(
    feats: TargetFeatures,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    rigidity_attenuate: bool = False,
    cycle_lambda: float = 0.0,
    k_train: int = 15,
    k_eval: int = 10,
    tau: float = 10.0,
    max_epochs: int = 80,
    batch_size: int = 512,
    lr: float = 8e-4,
) -> dict[str, Any]:
    """Train P1 (local only) or P2 (local + rigidity attenuation + cycle-closure loss) on 2D mol_encoder."""
    set_global_seed(seed)
    device = get_device()

    inner_tr_rel, inner_va_rel = make_inner_split(y[train_idx], seed=seed, val_frac=0.15)
    inner_tr_idx = train_idx[inner_tr_rel]
    inner_va_idx = train_idx[inner_va_rel]

    # Strictly 2D mol_encoder (no 93-dim global 3D/electronic MLP bloat)
    bundle_2d = build_tier_matrices(feats, tier="2d", fit_idx=inner_tr_idx)
    env_counters = precompute_mol_env_counters(feats.atom_ptr, feats.atom_env_hash)

    sim_tr = compute_tanimoto_matrix(feats.ecfp_chiral[inner_tr_idx], feats.ecfp_chiral[inner_tr_idx])
    nbr_g, nbr_sims = _select_topk_anchors(
        sim_tr, inner_tr_idx, k=k_train, exclude_self_global_idx=inner_tr_idx
    )
    pair_i = np.repeat(inner_tr_idx, k_train)
    pair_j = nbr_g.reshape(-1)
    pair_s = nbr_sims.reshape(-1)

    y_std = float(max(np.std(y[inner_tr_idx]), 1e-6))
    pair_dy = ((y[pair_i] - y[pair_j]) / y_std).astype(np.float32)

    raw_diff_tr = _build_local_physchem_diff(
        feats, bundle_2d.x_cont_scaled, pair_i, pair_j, env_counters, rigidity_attenuate=rigidity_attenuate
    )
    diff_scale = np.std(raw_diff_tr, axis=0)
    diff_scale = np.where(diff_scale < 1e-6, 1.0, diff_scale).astype(np.float32)
    scaled_diff_tr = np.clip(raw_diff_tr / diff_scale, -10.0, 10.0).astype(np.float32)

    # Prebuild cyclic third-anchor triangles (i, j, k) for thermodynamic cycle closure if lambda > 0
    if cycle_lambda > 0.0:
        # Pick k as the next cyclic neighbor of i inside inner_tr_idx
        n_tr_inner = inner_tr_idx.size
        k_cols = (np.arange(k_train) + 1) % k_train
        pair_k = nbr_g[:, k_cols].reshape(-1)
        sim_jk_mat = compute_tanimoto_matrix(feats.ecfp_chiral[inner_tr_idx], feats.ecfp_chiral[inner_tr_idx])
        pos_map = {int(g): pos for pos, g in enumerate(inner_tr_idx.tolist())}
        pos_j = np.asarray([pos_map[int(g)] for g in pair_j.tolist()], dtype=np.int64)
        pos_k = np.asarray([pos_map[int(g)] for g in pair_k.tolist()], dtype=np.int64)
        pos_i = np.repeat(np.arange(n_tr_inner, dtype=np.int64), k_train)

        sim_jk = sim_jk_mat[pos_j, pos_k]
        sim_ki = sim_jk_mat[pos_k, pos_i]

        diff_jk = np.clip(
            _build_local_physchem_diff(
                feats, bundle_2d.x_cont_scaled, pair_j, pair_k, env_counters, rigidity_attenuate=rigidity_attenuate
            ) / diff_scale,
            -10.0,
            10.0,
        ).astype(np.float32)
        diff_ki = np.clip(
            _build_local_physchem_diff(
                feats, bundle_2d.x_cont_scaled, pair_k, pair_i, env_counters, rigidity_attenuate=rigidity_attenuate
            ) / diff_scale,
            -10.0,
            10.0,
        ).astype(np.float32)

        pair_k_t = torch.tensor(pair_k, dtype=torch.long, device=device)
        diff_jk_t = torch.tensor(diff_jk, dtype=torch.float32, device=device)
        diff_ki_t = torch.tensor(diff_ki, dtype=torch.float32, device=device)
        sim_jk_t = torch.tensor(sim_jk, dtype=torch.float32, device=device)
        sim_ki_t = torch.tensor(sim_ki, dtype=torch.float32, device=device)

    model = AntisymmetricPairwiseDeltaNet(
        mol_dim=bundle_2d.x_mol.shape[1],
        pair_diff_dim=scaled_diff_tr.shape[1],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    loss_fn = nn.HuberLoss(delta=1.0)

    x_mol_all_t = torch.tensor(bundle_2d.x_mol, dtype=torch.float32, device=device)
    pair_i_t = torch.tensor(pair_i, dtype=torch.long, device=device)
    pair_j_t = torch.tensor(pair_j, dtype=torch.long, device=device)
    diff_tr_t = torch.tensor(scaled_diff_tr, dtype=torch.float32, device=device)
    sim_tr_t = torch.tensor(pair_s, dtype=torch.float32, device=device)
    dy_tr_t = torch.tensor(pair_dy, dtype=torch.float32, device=device)

    best_val_rmse = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    wait = 0
    patience = 12
    n_pairs = pair_i.size

    for _epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_pairs, device=device)
        for start in range(0, n_pairs, batch_size):
            b = perm[start : start + batch_size]
            opt.zero_grad(set_to_none=True)
            pred_ij = model(x_mol_all_t[pair_i_t[b]], x_mol_all_t[pair_j_t[b]], diff_tr_t[b], sim_tr_t[b])
            loss = loss_fn(pred_ij, dy_tr_t[b])
            if cycle_lambda > 0.0:
                pred_jk = model(x_mol_all_t[pair_j_t[b]], x_mol_all_t[pair_k_t[b]], diff_jk_t[b], sim_jk_t[b])
                pred_ki = model(x_mol_all_t[pair_k_t[b]], x_mol_all_t[pair_i_t[b]], diff_ki_t[b], sim_ki_t[b])
                cycle_res = pred_ij + pred_jk + pred_ki
                loss = loss + cycle_lambda * torch.mean(cycle_res * cycle_res)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

        y_va_pred, _ = _predict_compact_delta(
            model, feats, bundle_2d, inner_va_idx, inner_tr_idx, y, y_std, diff_scale,
            env_counters, rigidity_attenuate, device, k_eval=k_eval, tau=tau
        )
        va_rmse = float(np.sqrt(np.mean((y[inner_va_idx] - y_va_pred) ** 2)))
        if va_rmse < best_val_rmse - 1e-5:
            best_val_rmse = va_rmse
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    y_pred_val, s_max_val = _predict_compact_delta(
        model, feats, bundle_2d, inner_va_idx, inner_tr_idx, y, y_std, diff_scale,
        env_counters, rigidity_attenuate, device, k_eval=k_eval, tau=tau
    )
    y_pred_test, s_max_test = _predict_compact_delta(
        model, feats, bundle_2d, test_idx, train_idx, y, y_std, diff_scale,
        env_counters, rigidity_attenuate, device, k_eval=k_eval, tau=tau
    )
    return {
        "y_pred_val": y_pred_val,
        "s_max_val": s_max_val,
        "inner_tr_idx": inner_tr_idx,
        "inner_va_idx": inner_va_idx,
        "y_pred_test": y_pred_test,
        "s_max_test": s_max_test,
    }


def _fit_tanimoto_svr(
    fp_all: np.ndarray,
    y_all: np.ndarray,
    fit_idx: np.ndarray,
    query_idx: np.ndarray,
    c_val: float = 3.0,
    eps: float = 0.1,
) -> np.ndarray:
    """Exact precomputed Tanimoto-kernel SVR fit strictly on fit_idx."""
    k_tr = compute_tanimoto_matrix(fp_all[fit_idx], fp_all[fit_idx])
    k_q = compute_tanimoto_matrix(fp_all[query_idx], fp_all[fit_idx])
    svr = SVR(kernel="precomputed", C=c_val, epsilon=eps)
    svr.fit(k_tr, y_all[fit_idx])
    return svr.predict(k_q).astype(np.float64)


def run_p4a_tanimoto_gated_delta(
    feats: TargetFeatures,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    p2_out: dict[str, Any],
    p3_test_pred: np.ndarray,
) -> dict[str, Any]:
    """Blend P2 (physical-constraint local delta) with P3 (Tanimoto-kernel SVR) via inner_val gate."""
    inner_tr_idx = p2_out["inner_tr_idx"]
    inner_va_idx = p2_out["inner_va_idx"]

    y_val_svr = _fit_tanimoto_svr(feats.ecfp_chiral, y, inner_tr_idx, inner_va_idx)
    y_val_true = y[inner_va_idx]
    y_val_delta = p2_out["y_pred_val"]
    s_max_val = p2_out["s_max_val"]

    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

    def loss_fn(params: np.ndarray) -> float:
        a, b = float(params[0]), float(params[1])
        alpha = sigmoid(a * (s_max_val - b))
        y_blend = alpha * y_val_delta + (1.0 - alpha) * y_val_svr
        return float(np.mean((y_val_true - y_blend) ** 2)) + 1e-4 * (a ** 2)

    res = optimize.minimize(
        loss_fn,
        x0=np.array([6.0, 0.70], dtype=np.float64),
        bounds=[(0.0, 25.0), (0.20, 0.95)],
        method="L-BFGS-B",
    )
    a_opt, b_opt = float(res.x[0]), float(res.x[1])
    alpha_test = sigmoid(a_opt * (p2_out["s_max_test"] - b_opt))
    y_pred_test = (alpha_test * p2_out["y_pred_test"] + (1.0 - alpha_test) * p3_test_pred).astype(np.float64)
    return {
        "y_pred_test": y_pred_test,
        "gate_a": a_opt,
        "gate_b": b_opt,
        "alpha_test_mean": float(np.mean(alpha_test)),
    }


def _build_residual_features(
    feats: TargetFeatures,
    query_idx: np.ndarray,
    anchor_pool_idx: np.ndarray,
    anchor_residuals: np.ndarray,
    env_counters: list[Any],
    exclude_self: bool,
    k_res: int = 5,
    tau: float = 10.0,
) -> np.ndarray:
    """
    Build 16-dim compact interpretable residual features from top-5 nearest training anchors:
      [weighted_nbr_residual, s_max, s_mean5, rigidity_q,
       weighted_local_elec_sum_diff (8), weighted_local_3d_rigid_sum_diff (3), weighted_atom_cnt_diff]
    """
    sim_mat = compute_tanimoto_matrix(feats.ecfp_chiral[query_idx], feats.ecfp_chiral[anchor_pool_idx])
    anchor_g_idx, anchor_sims = _select_topk_anchors(
        sim_mat, anchor_pool_idx, k=k_res, exclude_self_global_idx=query_idx if exclude_self else None
    )
    pool_pos = {int(g): pos for pos, g in enumerate(anchor_pool_idx.tolist())}
    anchor_rel_pos = np.vectorize(lambda g: pool_pos[int(g)])(anchor_g_idx)

    n_q, k_eff = anchor_g_idx.shape
    logits = tau * anchor_sims
    logits -= np.max(logits, axis=1, keepdims=True)
    w = np.exp(logits)
    w /= np.sum(w, axis=1, keepdims=True)

    r_anchors = anchor_residuals[anchor_rel_pos]
    weighted_res = np.sum(w * r_anchors, axis=1, keepdims=True)
    s_max = anchor_sims[:, 0:1]
    s_mean = np.mean(anchor_sims, axis=1, keepdims=True)

    rotb = feats.desc2d[:, 5]
    rig_q = np.exp(-rotb[query_idx] / 6.0)[:, None].astype(np.float32)

    q_flat = np.repeat(query_idx, k_eff)
    a_flat = anchor_g_idx.reshape(-1)

    de = feats.atom_elec_props.shape[1]
    d3 = feats.atom_3d_props.shape[1]
    raw_el = compute_local_pair_deltas(
        q_flat, a_flat, feats.atom_ptr, feats.atom_env_hash, feats.atom_elec_props, env_counters=env_counters
    )[:, :de].reshape(n_q, k_eff, de)
    raw_3d = compute_local_pair_deltas(
        q_flat, a_flat, feats.atom_ptr, feats.atom_env_hash, feats.atom_3d_props, env_counters=env_counters
    )
    cnt_diff = raw_3d[:, 3 * d3 : 3 * d3 + 1].reshape(n_q, k_eff, 1)
    geom_3d = raw_3d[:, :d3].reshape(n_q, k_eff, d3)

    max_rot = np.maximum(rotb[q_flat], rotb[a_flat]).astype(np.float32).reshape(n_q, k_eff, 1)
    geom_3d = geom_3d * np.exp(-max_rot / 6.0)

    w_exp = w[:, :, None]
    w_el = np.sum(w_exp * raw_el, axis=1)
    w_3d = np.sum(w_exp * geom_3d, axis=1)
    w_cnt = np.sum(w_exp * cnt_diff, axis=1)

    return np.concatenate([weighted_res, s_max, s_mean, rig_q, w_el, w_3d, w_cnt], axis=1).astype(np.float32)


def run_p4b_tanimoto_plus_residual(
    feats: TargetFeatures,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    p3_test_pred: np.ndarray,
) -> dict[str, Any]:
    """5-fold inner OOF Tanimoto-SVR inside train + RidgeCV local PhysChem residual correction."""
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof_svr = np.zeros((train_idx.size,), dtype=np.float64)
    for tr_rel, va_rel in kf.split(train_idx):
        oof_svr[va_rel] = _fit_tanimoto_svr(
            feats.ecfp_chiral, y, train_idx[tr_rel], train_idx[va_rel]
        )

    train_residuals = y[train_idx] - oof_svr
    env_counters = precompute_mol_env_counters(feats.atom_ptr, feats.atom_env_hash)

    x_res_tr = _build_residual_features(
        feats, train_idx, train_idx, train_residuals, env_counters, exclude_self=True
    )
    x_res_te = _build_residual_features(
        feats, test_idx, train_idx, train_residuals, env_counters, exclude_self=False
    )

    scaler = StandardScaler()
    x_res_tr_s = np.clip(scaler.fit_transform(x_res_tr), -8.0, 8.0)
    x_res_te_s = np.clip(scaler.transform(x_res_te), -8.0, 8.0)

    ridge = RidgeCV(alphas=np.logspace(0, 4, 25))
    ridge.fit(x_res_tr_s, train_residuals)
    pred_res_te = ridge.predict(x_res_te_s).astype(np.float64)

    return {
        "y_pred_test": p3_test_pred + pred_res_te,
        "ridge_alpha": float(ridge.alpha_),
        "res_std_pred": float(np.std(pred_res_te)),
        "oof_svr_rmse": float(np.sqrt(np.mean(train_residuals ** 2))),
    }


def _plot_diagnostics(flex_df: pd.DataFrame, summary_df: pd.DataFrame) -> Path:
    """Create 2-panel diagnostic figure (flexibility stratification + probe comparison)."""
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8))

    # Left panel: Delta Cliff RMSE (A4 - A2) by Rotatable Bonds bin
    bins_order = ["<=4 (rigid)", "5-7 (moderate)", ">=8 (flexible)"]
    targets_order = list(MVP_TARGETS)
    colors = ["#0072B2", "#D55E00", "#009E73"]
    bx = np.arange(len(bins_order))
    w = 0.25
    for idx_t, tgt in enumerate(targets_order):
        vals = []
        for rb in bins_order:
            sub = flex_df[(flex_df["target"] == tgt) & (flex_df["rot_bin"] == rb)]
            vals.append(float(sub["delta_cliff_rmse_a4_a2"].iloc[0]) if len(sub) > 0 else 0.0)
        ax1.bar(bx + (idx_t - 1) * w, vals, width=w, label=tgt, color=colors[idx_t], edgecolor="black", linewidth=0.5)
    ax1.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax1.set_xticks(bx)
    ax1.set_xticklabels(bins_order)
    ax1.set_ylabel("Δ Cliff RMSE (A4_PhysChem - A2_2D)\n(<0 means PhysChem helps; >0 hurts)")
    ax1.set_title("Flexibility Attribution of 3D+AIMNet2 Cliff Shift", fontweight="bold")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    # Right panel: Probe comparison normalized or per-target grouped
    probe_order = [
        ("B1c_ECFP_SVM", "B1c RBF-SVM"),
        ("A2_2D_Pairwise_Delta", "A2 2D-Delta"),
        ("A4_PhysChem_Delta", "A4 Week1-Delta"),
        ("P1_LocalPhysChemOnly_Delta", "P1 Local-Only"),
        ("P2_CycleClosure_RigidLocal_Delta", "P2 +Rigid+Cycle"),
        ("P3_Tanimoto_SVR", "P3 Tanimoto-SVR"),
        ("P4a_TanimotoSVR_Gated_P2Delta", "P4a Gate(P2+P3)"),
        ("P4b_TanimotoSVR_Plus_PhysChemResidual", "P4b P3+Resid"),
    ]
    x2 = np.arange(len(probe_order))
    for idx_t, tgt in enumerate(targets_order):
        ys = []
        for m_key, _ in probe_order:
            sub = summary_df[(summary_df["target"] == tgt) & (summary_df["model"] == m_key)]
            ys.append(float(sub["cliff_rmse_mean"].iloc[0]) if len(sub) > 0 else np.nan)
        ax2.plot(x2, ys, marker="o", linewidth=1.8, label=tgt, color=colors[idx_t])
    ax2.set_xticks(x2)
    ax2.set_xticklabels([lbl for _, lbl in probe_order], rotation=28, ha="right", fontsize=8.5)
    ax2.set_ylabel("Activity Cliff RMSE (5-seed mean)")
    ax2.set_title("Physical Constraints & Kernel Base Swap Probes", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    out_path = _fig_dir() / "diag_w1_flexibility_and_probes.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main() -> None:
    t0 = time.time()
    datasets = {tgt: load_target_dataset(tgt) for tgt in MVP_TARGETS}
    features = {
        tgt: extract_or_load_target_features(tgt, datasets[tgt].smiles, prefer_aimnet=True, num_confs=5)
        for tgt in MVP_TARGETS
    }

    print("=== Running Part A: Zero-Retrain Diagnostic Analysis ===")
    profile_rows, corr_rows, top_cases, flex_df, strat_df = run_part_a_diagnostics(datasets, features)

    print("=== Running Part B: 5-Seed Physical Constraint & Base-Architecture Probes ===")
    probe_records: list[dict[str, Any]] = []

    # Load existing Week-1 records for direct side-by-side table
    w1_raw = pd.read_csv(_refine_dir() / "mvp_raw_runs.csv")
    w1_keep = w1_raw[
        w1_raw["model"].isin(
            [
                "B1_ECFP_Chiral_CatBoost",
                "B1c_ECFP_SVM",
                "A2_2D_Pairwise_Delta",
                "A4_PhysChem_Delta",
                "M_AnchorGated_PhysChem_Delta",
            ]
        )
    ]
    for _, r in w1_keep.iterrows():
        probe_records.append(
            {
                "target": r["target"],
                "model": r["model"],
                "seed": int(r["seed"]),
                "rmse": float(r["rmse"]),
                "cliff_rmse": float(r["cliff_rmse"]),
                "non_cliff_rmse": float(r["non_cliff_rmse"]),
                "spearman": float(r["spearman"]),
            }
        )

    probe_preds_store: dict[tuple[str, str, int], np.ndarray] = {}

    for tgt in MVP_TARGETS:
        ds = datasets[tgt]
        feats = features[tgt]
        y_te = ds.y[ds.test_idx]
        c_te = ds.cliff_mol[ds.test_idx]

        # P3_Tanimoto_SVR is deterministic per split -> fit once and evaluate
        p3_pred_test = _fit_tanimoto_svr(feats.ecfp_chiral, ds.y, ds.train_idx, ds.test_idx)
        p3_metrics = evaluate_predictions(y_te, p3_pred_test, c_te)

        for seed in SEEDS:
            # P1: compact local PhysChem only (no global 3D/elec bloat)
            p1_out = run_compact_delta_probe(
                feats, ds.y, ds.train_idx, ds.test_idx, seed=seed, rigidity_attenuate=False, cycle_lambda=0.0
            )
            p1_m = evaluate_predictions(y_te, p1_out["y_pred_test"], c_te)
            probe_records.append({"target": tgt, "model": "P1_LocalPhysChemOnly_Delta", "seed": seed, **p1_m})
            probe_preds_store[(tgt, "P1_LocalPhysChemOnly_Delta", seed)] = p1_out["y_pred_test"]

            # P2: P1 + conformational rigidity attenuation + thermodynamic cycle closure
            p2_out = run_compact_delta_probe(
                feats, ds.y, ds.train_idx, ds.test_idx, seed=seed, rigidity_attenuate=True, cycle_lambda=0.2
            )
            p2_m = evaluate_predictions(y_te, p2_out["y_pred_test"], c_te)
            probe_records.append({"target": tgt, "model": "P2_CycleClosure_RigidLocal_Delta", "seed": seed, **p2_m})
            probe_preds_store[(tgt, "P2_CycleClosure_RigidLocal_Delta", seed)] = p2_out["y_pred_test"]

            # P3: Tanimoto-kernel SVR
            probe_records.append({"target": tgt, "model": "P3_Tanimoto_SVR", "seed": seed, **p3_metrics})
            probe_preds_store[(tgt, "P3_Tanimoto_SVR", seed)] = p3_pred_test

            # P4a: Tanimoto-kernel SVR + OOD gate with P2 physical-constraint delta
            p4a_out = run_p4a_tanimoto_gated_delta(feats, ds.y, ds.train_idx, ds.test_idx, p2_out, p3_pred_test)
            p4a_m = evaluate_predictions(y_te, p4a_out["y_pred_test"], c_te)
            probe_records.append(
                {
                    "target": tgt,
                    "model": "P4a_TanimotoSVR_Gated_P2Delta",
                    "seed": seed,
                    "gate_a": p4a_out["gate_a"],
                    "gate_b": p4a_out["gate_b"],
                    "alpha_mean": p4a_out["alpha_test_mean"],
                    **p4a_m,
                }
            )
            probe_preds_store[(tgt, "P4a_TanimotoSVR_Gated_P2Delta", seed)] = p4a_out["y_pred_test"]

            # P4b: Tanimoto-kernel SVR + RidgeCV local PhysChem residual
            p4b_out = run_p4b_tanimoto_plus_residual(
                feats, ds.y, ds.train_idx, ds.test_idx, seed=seed, p3_test_pred=p3_pred_test
            )
            p4b_m = evaluate_predictions(y_te, p4b_out["y_pred_test"], c_te)
            probe_records.append(
                {
                    "target": tgt,
                    "model": "P4b_TanimotoSVR_Plus_PhysChemResidual",
                    "seed": seed,
                    "ridge_alpha": p4b_out["ridge_alpha"],
                    "res_std": p4b_out["res_std_pred"],
                    **p4b_m,
                }
            )
            probe_preds_store[(tgt, "P4b_TanimotoSVR_Plus_PhysChemResidual", seed)] = p4b_out["y_pred_test"]

            print(
                f"[{tgt} | seed={seed}] "
                f"P1={p1_m['cliff_rmse']:.4f} | P2={p2_m['cliff_rmse']:.4f} | "
                f"P3={p3_metrics['cliff_rmse']:.4f} | P4a={p4a_m['cliff_rmse']:.4f} | P4b={p4b_m['cliff_rmse']:.4f}",
                flush=True,
            )

    rec_df = pd.DataFrame(probe_records)
    model_order = [
        "B1_ECFP_Chiral_CatBoost",
        "B1c_ECFP_SVM",
        "A2_2D_Pairwise_Delta",
        "A4_PhysChem_Delta",
        "M_AnchorGated_PhysChem_Delta",
        "P1_LocalPhysChemOnly_Delta",
        "P2_CycleClosure_RigidLocal_Delta",
        "P3_Tanimoto_SVR",
        "P4a_TanimotoSVR_Gated_P2Delta",
        "P4b_TanimotoSVR_Plus_PhysChemResidual",
    ]

    summary_rows = []
    for tgt in MVP_TARGETS:
        for m in model_order:
            sub = rec_df[(rec_df["target"] == tgt) & (rec_df["model"] == m)]
            if sub.empty:
                continue
            summary_rows.append(
                {
                    "target": tgt,
                    "model": m,
                    "rmse_mean": round(float(sub["rmse"].mean()), 4),
                    "rmse_std": round(float(sub["rmse"].std(ddof=1)), 4),
                    "cliff_rmse_mean": round(float(sub["cliff_rmse"].mean()), 4),
                    "cliff_rmse_std": round(float(sub["cliff_rmse"].std(ddof=1)), 4),
                    "non_cliff_rmse_mean": round(float(sub["non_cliff_rmse"].mean()), 4),
                    "spearman_mean": round(float(sub["spearman"].mean()), 4),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(_refine_dir() / "diag_w1_probes.csv", index=False)

    # Paired Wilcoxon tests for probes vs B1c_ECFP_SVM and vs A2_2D_Pairwise_Delta
    wilcoxon_rows = []
    for tgt in MVP_TARGETS:
        ds = datasets[tgt]
        rdir = _run_dir(tgt)
        y_te = ds.y[ds.test_idx]
        cmask = ds.cliff_mol[ds.test_idx] == 1
        for cand in (
            "P1_LocalPhysChemOnly_Delta",
            "P2_CycleClosure_RigidLocal_Delta",
            "P3_Tanimoto_SVR",
            "P4a_TanimotoSVR_Gated_P2Delta",
            "P4b_TanimotoSVR_Plus_PhysChemResidual",
        ):
            for ref in ("B1c_ECFP_SVM", "A2_2D_Pairwise_Delta"):
                ec_list, er_list = [], []
                for seed in SEEDS:
                    pc = probe_preds_store[(tgt, cand, seed)]
                    dr = pd.read_csv(rdir / f"{ref}_seed{seed}_preds.csv")
                    ec_list.append(np.abs(y_te[cmask] - pc[cmask]))
                    er_list.append(np.abs(y_te[cmask] - dr["y_pred"].to_numpy()[cmask]))
                pval = paired_wilcoxon(np.concatenate(ec_list), np.concatenate(er_list))
                c_val = float(summary_df[(summary_df["target"] == tgt) & (summary_df["model"] == cand)]["cliff_rmse_mean"].iloc[0])
                r_val = float(summary_df[(summary_df["target"] == tgt) & (summary_df["model"] == ref)]["cliff_rmse_mean"].iloc[0])
                wilcoxon_rows.append(
                    {
                        "target": tgt,
                        "candidate": cand,
                        "reference": ref,
                        "delta_cliff_rmse": round(c_val - r_val, 4),
                        "wilcoxon_p": float(f"{pval:.3e}"),
                    }
                )

    fig_path = _plot_diagnostics(flex_df, summary_df)

    # Write DIAGNOSTIC_W1.md
    md: list[str] = [
        "# Week-1 Diagnostic Audit & Physical-Constraint Feasibility Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **Elapsed**: {time.time() - t0:.1f}s",
        "",
        "## Part A.1: Target Chemical Flexibility & 3D Profile",
        "| Target | n_total | RotBonds Mean (Med) | RotBonds >=8 (%) | MolWt Mean | Fsp3 Mean | TPSA Mean | e_span Med | Chiral (%) | Test s_max Med |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in profile_rows:
        md.append(
            f"| {p['target']} | {p['n_total']} | {p['rotb_mean']} ({p['rotb_median']}) | "
            f"{p['rotb_gt7_pct']}% | {p['molwt_mean']} | {p['fsp3_mean']} | {p['tpsa_mean']} | "
            f"{p['espan_median']} | {p['chiral_pct']}% | {p['smax_test_median']} |"
        )

    md.extend(
        [
            "",
            "## Part A.2: Cliff Degradation Correlations (Pooled 5 seeds per target)",
            "| Target | Feature | Spearman rho (|e_A4|-|e_A2|) | p-val | Spearman rho (|e_M|-|e_SVM|) | p-val |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for c in corr_rows:
        md.append(
            f"| {c['target']} | `{c['feature']}` | {c['rho_delta_A4_vs_A2']:+.4f} | {c['p_A4_vs_A2']:.3e} | "
            f"{c['rho_delta_M_vs_SVM']:+.4f} | {c['p_M_vs_SVM']:.3e} |"
        )

    md.extend(
        [
            "",
            "## Part A.3: Stratified Cliff RMSE Breakdown",
            "| Target | Stratum | n_obs (5 seeds) | A2 2D-Delta | A3 2D+3D | A4 PhysChem | M Gate | B1c SVM | Mean |e_A4|-|e_A2| |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, s in strat_df.iterrows():
        md.append(
            f"| {s['target']} | {s['stratum']} | {s['n_obs']} | {s['rmse_a2']:.4f} | {s['rmse_a3']:.4f} | "
            f"{s['rmse_a4']:.4f} | {s['rmse_m']:.4f} | {s['rmse_svm']:.4f} | {s['mean_abs_shift_A4_A2']:+.4f} |"
        )

    md.extend(["", "## Part A.4: Top-5 Worst A4-vs-A2 Cliff Degradation Cases"])
    for tgt, cases in top_cases.items():
        md.append(f"### {tgt}")
        md.append("| test_idx | RotBonds | e_span | s_max | y_true | anchor_y | pred_SVM | pred_A2 | pred_A4 | |e_A4|-|e_A2| | SMILES |")
        md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for cs in cases:
            md.append(
                f"| {cs['test_idx']} | {cs['rotb']} | {cs['espan']} | {cs['s_max']} | "
                f"{cs['y_true']} | {cs['anchor_y']} | {cs['pred_svm']} | {cs['pred_a2']} | {cs['pred_a4']} | "
                f"{cs['shift_a4_minus_a2']:+.3f} | `{cs['smiles'][:60]}` |"
            )

    md.extend(
        [
            "",
            "## Part B.1: 5-Seed Controlled Probes vs Week-1 Baselines",
            "| Target | Model | RMSE | Cliff RMSE | Non-Cliff RMSE | Spearman |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in summary_df.iterrows():
        md.append(
            f"| {row['target']} | `{row['model']}` | "
            f"{row['rmse_mean']:.4f} ± {row['rmse_std']:.4f} | "
            f"**{row['cliff_rmse_mean']:.4f} ± {row['cliff_rmse_std']:.4f}** | "
            f"{row['non_cliff_rmse_mean']:.4f} | {row['spearman_mean']:.4f} |"
        )

    md.extend(
        [
            "",
            "## Part B.2: Paired Wilcoxon Significance on Cliff Subset (Probes vs B1c_SVM & A2_2D_Delta)",
            "| Target | Probe | Reference | Δ Cliff RMSE | p-value |",
            "|---|---|---|---:|---:|",
        ]
    )
    for wr in wilcoxon_rows:
        md.append(
            f"| {wr['target']} | `{wr['candidate']}` | `{wr['reference']}` | "
            f"{wr['delta_cliff_rmse']:+.4f} | {wr['wilcoxon_p']:.3e} |"
        )

    md.extend(["", f"Diagnostic Plot: `{fig_path}`"])
    (_refine_dir() / "DIAGNOSTIC_W1.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Saved DIAGNOSTIC_W1.md, diag_w1_probes.csv, and {fig_path}")


if __name__ == "__main__":
    main()

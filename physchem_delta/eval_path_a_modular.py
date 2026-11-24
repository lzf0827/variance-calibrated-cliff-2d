"""Path A Kill-Test (SVM+Delta vs SVM+MLP/CatBoost) + Modular Architecture Exploration."""

from __future__ import annotations

import argparse
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
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.svm import SVR
import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from physchem_delta.data import load_target_dataset
from physchem_delta.eval_plan_a import SEEDS, _build_pair_diff, _lfer_atom_matrix
from physchem_delta.featurize import (
    TargetFeatures,
    compute_local_pair_deltas,
    extract_or_load_target_features,
    precompute_mol_env_counters,
)
from physchem_delta.metrics import evaluate_predictions
from physchem_delta.models import (
    AntisymmetricPairwiseDeltaNet,
    _select_topk_anchors,
    _train_direct_mlp_with_inner_split,
    build_tier_matrices,
    compute_tanimoto_matrix,
    get_device,
    make_inner_split,
    run_tabular_baseline,
    set_global_seed,
)
from physchem_delta.run_30_targets import WAVE_1_TARGETS

ALL_REPORT_MODELS: tuple[str, ...] = (
    "B1c_RBF_SVM",
    "B1_CatBoost",
    "A1_Direct_MLP",
    "A2_2D_Delta",
    "E_SVM_CatBoost",
    "E_SVM_DirectMLP",
    "E_SVM_A2Delta",
    "G0_RBFSVM_Gate_A2",
    "G2_Prev",
    "KR_2D_Delta",
    "KR_FiLM_2DLFER",
    "KR_FiLM_Gated3DQM",
)


def _out_root() -> Path:
    d = ROOT_DIR / "refine-logs" / "path_a_modular_results"
    (d / "per_target").mkdir(parents=True, exist_ok=True)
    return d


def _fig_dir() -> Path:
    d = ROOT_DIR / "refine-logs" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


class FiLMKernelResidualDeltaNet(nn.Module):
    """Antisymmetric pairwise residual net with zero-initialized FiLM + LFER skip."""

    def __init__(self, mol_dim: int, diff2d_dim: int = 26, lfer2d_dim: int = 7, emb_dim: int = 128, dropout: float = 0.25):
        super().__init__()
        self.mol_encoder = nn.Sequential(
            nn.Linear(mol_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.SiLU(),
        )
        pair_in_dim = emb_dim * 4 + diff2d_dim + 1
        self.pair_trunk = nn.Sequential(
            nn.Linear(pair_in_dim, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Dropout(dropout * 0.8),
        )
        self.film_gamma = nn.Linear(lfer2d_dim, 192)
        self.film_beta = nn.Linear(lfer2d_dim, 192)
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

        self.readout = nn.Sequential(
            nn.Linear(192, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.lfer_skip = nn.Linear(lfer2d_dim, 1, bias=False)
        nn.init.zeros_(self.lfer_skip.weight)

    def _encode_pair_half(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        prod: torch.Tensor,
        diff2d: torch.Tensor,
        sim_col: torch.Tensor,
        p_lfer2d: torch.Tensor,
    ) -> torch.Tensor:
        u = torch.cat([h_a, h_b, h_a - h_b, prod, diff2d, sim_col], dim=-1)
        h_tr = self.pair_trunk(u)
        gamma = 1.0 + torch.tanh(self.film_gamma(p_lfer2d))
        beta = self.film_beta(p_lfer2d)
        h_mod = gamma * h_tr + beta
        return self.readout(h_mod).squeeze(-1) + self.lfer_skip(p_lfer2d).squeeze(-1)

    def forward_base(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        diff2d_ij: torch.Tensor,
        sim_ij: torch.Tensor,
        p_lfer2d_ij: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_i = self.mol_encoder(x_i)
        h_j = self.mol_encoder(x_j)
        prod = h_i * h_j
        sim_col = sim_ij.unsqueeze(-1)
        out_ij = self._encode_pair_half(h_i, h_j, prod, diff2d_ij, sim_col, p_lfer2d_ij)
        out_ji = self._encode_pair_half(h_j, h_i, prod, -diff2d_ij, sim_col, -p_lfer2d_ij)
        return 0.5 * (out_ij - out_ji), h_i, h_j

    def forward(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        diff2d_ij: torch.Tensor,
        sim_ij: torch.Tensor,
        p_lfer2d_ij: torch.Tensor,
    ) -> torch.Tensor:
        out, _, _ = self.forward_base(x_i, x_j, diff2d_ij, sim_ij, p_lfer2d_ij)
        return out


class FiLMGated3DQMResidualNet(FiLMKernelResidualDeltaNet):
    """Extends FiLMKernelResidualDeltaNet with L1-gated 3D/QM group-sparse + ReZero branch."""

    def __init__(
        self,
        mol_dim: int,
        diff2d_dim: int = 26,
        lfer2d_dim: int = 7,
        p3d_dim: int = 8,
        emb_dim: int = 128,
        dropout: float = 0.25,
    ):
        super().__init__(mol_dim=mol_dim, diff2d_dim=diff2d_dim, lfer2d_dim=lfer2d_dim, emb_dim=emb_dim, dropout=dropout)
        self.gate_logits = nn.Parameter(torch.full((3,), -2.5))
        self.beta_3d = nn.Parameter(torch.zeros(()))
        self.branch_3d = nn.Sequential(
            nn.Linear(emb_dim * 2 + p3d_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(64, 1),
        )

    def _expanded_gates(self) -> torch.Tensor:
        g = torch.sigmoid(self.gate_logits)
        return torch.cat([g[0:1].expand(2), g[1:2].expand(4), g[2:3].expand(2)], dim=0)

    def forward(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        diff2d_ij: torch.Tensor,
        sim_ij: torch.Tensor,
        p_lfer2d_ij: torch.Tensor,
        p_3dqm_ij: torch.Tensor,
    ) -> torch.Tensor:
        base_out, h_i, h_j = self.forward_base(x_i, x_j, diff2d_ij, sim_ij, p_lfer2d_ij)
        p3d_gated = p_3dqm_ij * self._expanded_gates()
        res_ij = self.branch_3d(torch.cat([h_i, h_j, p3d_gated], dim=-1)).squeeze(-1)
        res_ji = self.branch_3d(torch.cat([h_j, h_i, -p3d_gated], dim=-1)).squeeze(-1)
        delta_3d = 0.5 * (res_ij - res_ji)
        return base_out + self.beta_3d * delta_3d


def _compute_oof_svr(
    ecfp_chiral: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    seed: int,
    n_splits: int = 5,
) -> np.ndarray:
    """Strictly leak-free K-fold OOF RBF-SVR predictions on fit_idx."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    x_sub = ecfp_chiral[fit_idx]
    y_sub = y[fit_idx]
    oof = np.zeros((fit_idx.size,), dtype=np.float64)
    for tr_rel, va_rel in kf.split(x_sub):
        svr = SVR(kernel="rbf", C=3.0, epsilon=0.1)
        svr.fit(x_sub[tr_rel], y_sub[tr_rel])
        oof[va_rel] = svr.predict(x_sub[va_rel])
    return oof


def _build_modular_pair_features(
    feats: TargetFeatures,
    x_cont_scaled: np.ndarray,
    lfer_atoms: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    env_counters: list[Any],
    e0_kcal: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return strictly antisymmetric:
      - diff2d:   (N, 26) desc2d diff + chiral fp bit diff
      - p_lfer2d: (N, 7)  sum/mean of [crip_logp, crip_mr, tpsa_c] + cnt_diff
      - p_3dqm:   (N, 8)  [AIMNet2_q(2), 3D_eSpan_Geom(4), Chiral_Stereo(2)]
    """
    diff2d = _build_pair_diff(feats, x_cont_scaled, lfer_atoms, i_idx, j_idx, env_counters, mode="2d")
    raw_lfer = compute_local_pair_deltas(
        i_idx, j_idx, feats.atom_ptr, feats.atom_env_hash, lfer_atoms, env_counters=env_counters
    )
    # raw_lfer (N, 22): [0:7]=sum, [7:14]=mean, [21:22]=cnt_diff
    p_lfer2d = raw_lfer[:, [1, 2, 3, 8, 9, 10, 21]].astype(np.float32)

    q_part = raw_lfer[:, [0, 7]]
    e_span = feats.feat_3d[:, 1]
    e_pair = np.maximum(e_span[i_idx], e_span[j_idx]).astype(np.float32)
    c_conf = np.exp(-np.clip(e_pair, 0.0, 100.0) / e0_kcal)[:, None]
    geom_part = raw_lfer[:, [4, 5, 11, 12]] * c_conf
    chiral_part = raw_lfer[:, [6, 13]]
    p_3dqm = np.concatenate([q_part, geom_part, chiral_part], axis=1).astype(np.float32)
    return diff2d, p_lfer2d, p_3dqm


def _predict_kr_backbone(
    model: nn.Module,
    variant: str,
    feats: TargetFeatures,
    bundle_2d: Any,
    lfer_atoms: np.ndarray,
    query_idx: np.ndarray,
    anchor_pool_idx: np.ndarray,
    r_anchor_vec: np.ndarray,
    y_svm_query: np.ndarray,
    r_std: float,
    scales: tuple[np.ndarray, np.ndarray, np.ndarray],
    env_counters: list[Any],
    device: torch.device,
    k_eval: int = 10,
    tau: float = 10.0,
) -> np.ndarray:
    sim_mat = compute_tanimoto_matrix(feats.ecfp_chiral[query_idx], feats.ecfp_chiral[anchor_pool_idx])
    anchor_g_idx, anchor_sims = _select_topk_anchors(sim_mat, anchor_pool_idx, k=k_eval)

    n_q, k_eff = anchor_g_idx.shape
    q_flat = np.repeat(query_idx, k_eff)
    a_flat = anchor_g_idx.reshape(-1)
    s_flat = anchor_sims.reshape(-1)

    diff2d_raw, lfer2d_raw, p3d_raw = _build_modular_pair_features(
        feats, bundle_2d.x_cont_scaled, lfer_atoms, q_flat, a_flat, env_counters
    )
    sc_2d, sc_lfer, sc_3d = scales
    d2d = np.clip(diff2d_raw / sc_2d, -10.0, 10.0).astype(np.float32)
    dlfer = np.clip(lfer2d_raw / sc_lfer, -10.0, 10.0).astype(np.float32)
    d3d = np.clip(p3d_raw / sc_3d, -10.0, 10.0).astype(np.float32)

    model.eval()
    preds_scaled = []
    with torch.no_grad():
        for start in range(0, q_flat.size, 2048):
            sl = slice(start, start + 2048)
            xi_t = torch.tensor(bundle_2d.x_mol[q_flat[sl]], dtype=torch.float32, device=device)
            xj_t = torch.tensor(bundle_2d.x_mol[a_flat[sl]], dtype=torch.float32, device=device)
            s_t = torch.tensor(s_flat[sl], dtype=torch.float32, device=device)
            d2d_t = torch.tensor(d2d[sl], dtype=torch.float32, device=device)
            if variant == "KR_2D_Delta":
                out = model(xi_t, xj_t, d2d_t, s_t).cpu().numpy()
            elif variant == "KR_FiLM_2DLFER":
                dlf_t = torch.tensor(dlfer[sl], dtype=torch.float32, device=device)
                out = model(xi_t, xj_t, d2d_t, s_t, dlf_t).cpu().numpy()
            else:
                dlf_t = torch.tensor(dlfer[sl], dtype=torch.float32, device=device)
                d3d_t = torch.tensor(d3d[sl], dtype=torch.float32, device=device)
                out = model(xi_t, xj_t, d2d_t, s_t, dlf_t, d3d_t).cpu().numpy()
            preds_scaled.append(out)

    delta_r_pred = (np.concatenate(preds_scaled, axis=0) * r_std).reshape(n_q, k_eff)
    r_anchors = r_anchor_vec[anchor_g_idx]

    logits = tau * anchor_sims
    logits = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    weights = weights / np.sum(weights, axis=1, keepdims=True)

    # Smoothly shrinks correction toward 0 via anchor_sims for distant scaffolds
    res_corr = np.sum((weights * anchor_sims) * (r_anchors + delta_r_pred), axis=1)
    return (y_svm_query + res_corr).astype(np.float64)


def _train_kr_backbone(
    variant: str,
    feats: TargetFeatures,
    lfer_atoms: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    inner_tr_idx: np.ndarray,
    inner_va_idx: np.ndarray,
    test_idx: np.ndarray,
    r_inner_vec: np.ndarray,
    y_svm_va: np.ndarray,
    r_train_vec: np.ndarray,
    y_test_svm: np.ndarray,
    seed: int,
    k_train: int = 15,
    k_eval: int = 10,
    tau: float = 10.0,
    max_epochs: int = 80,
    batch_size: int = 512,
    lr: float = 8e-4,
) -> dict[str, Any]:
    set_global_seed(seed)
    device = get_device()

    bundle_2d = build_tier_matrices(feats, tier="2d", fit_idx=inner_tr_idx)
    env_counters = precompute_mol_env_counters(feats.atom_ptr, feats.atom_env_hash)

    sim_tr = compute_tanimoto_matrix(feats.ecfp_chiral[inner_tr_idx], feats.ecfp_chiral[inner_tr_idx])
    nbr_g, nbr_sims = _select_topk_anchors(
        sim_tr, inner_tr_idx, k=k_train, exclude_self_global_idx=inner_tr_idx
    )
    pair_i = np.repeat(inner_tr_idx, k_train)
    pair_j = nbr_g.reshape(-1)
    pair_s = nbr_sims.reshape(-1)

    r_std = float(max(np.std(r_inner_vec[inner_tr_idx]), 1e-6))
    pair_dr = ((r_inner_vec[pair_i] - r_inner_vec[pair_j]) / r_std).astype(np.float32)

    d2d_raw, dlf_raw, d3d_raw = _build_modular_pair_features(
        feats, bundle_2d.x_cont_scaled, lfer_atoms, pair_i, pair_j, env_counters
    )
    sc_2d = np.where(np.std(d2d_raw, axis=0) < 1e-6, 1.0, np.std(d2d_raw, axis=0)).astype(np.float32)
    sc_lf = np.where(np.std(dlf_raw, axis=0) < 1e-6, 1.0, np.std(dlf_raw, axis=0)).astype(np.float32)
    sc_3d = np.where(np.std(d3d_raw, axis=0) < 1e-6, 1.0, np.std(d3d_raw, axis=0)).astype(np.float32)

    d2d_tr = np.clip(d2d_raw / sc_2d, -10.0, 10.0).astype(np.float32)
    dlf_tr = np.clip(dlf_raw / sc_lf, -10.0, 10.0).astype(np.float32)
    d3d_tr = np.clip(d3d_raw / sc_3d, -10.0, 10.0).astype(np.float32)

    mol_dim = bundle_2d.x_mol.shape[1]
    if variant == "KR_2D_Delta":
        model: nn.Module = AntisymmetricPairwiseDeltaNet(mol_dim=mol_dim, pair_diff_dim=d2d_tr.shape[1]).to(device)
    elif variant == "KR_FiLM_2DLFER":
        model = FiLMKernelResidualDeltaNet(mol_dim=mol_dim, diff2d_dim=d2d_tr.shape[1], lfer2d_dim=dlf_tr.shape[1]).to(device)
    elif variant == "KR_FiLM_Gated3DQM":
        model = FiLMGated3DQMResidualNet(
            mol_dim=mol_dim, diff2d_dim=d2d_tr.shape[1], lfer2d_dim=dlf_tr.shape[1], p3d_dim=d3d_tr.shape[1]
        ).to(device)
    else:
        raise ValueError(f"Unknown KR variant: {variant}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    loss_fn = nn.HuberLoss(delta=1.0)

    x_mol_all_t = torch.tensor(bundle_2d.x_mol, dtype=torch.float32, device=device)
    pair_i_t = torch.tensor(pair_i, dtype=torch.long, device=device)
    pair_j_t = torch.tensor(pair_j, dtype=torch.long, device=device)
    d2d_tr_t = torch.tensor(d2d_tr, dtype=torch.float32, device=device)
    dlf_tr_t = torch.tensor(dlf_tr, dtype=torch.float32, device=device)
    d3d_tr_t = torch.tensor(d3d_tr, dtype=torch.float32, device=device)
    sim_tr_t = torch.tensor(pair_s, dtype=torch.float32, device=device)
    dr_tr_t = torch.tensor(pair_dr, dtype=torch.float32, device=device)

    y_va_true = y[inner_va_idx]
    best_val_rmse = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    wait = 0
    patience = 12
    n_pairs = pair_i.size

    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(n_pairs, device=device)
        for start in range(0, n_pairs, batch_size):
            b = perm[start : start + batch_size]
            opt.zero_grad(set_to_none=True)
            xi, xj = x_mol_all_t[pair_i_t[b]], x_mol_all_t[pair_j_t[b]]
            if variant == "KR_2D_Delta":
                pred_ij = model(xi, xj, d2d_tr_t[b], sim_tr_t[b])
                loss = loss_fn(pred_ij, dr_tr_t[b])
            elif variant == "KR_FiLM_2DLFER":
                pred_ij = model(xi, xj, d2d_tr_t[b], sim_tr_t[b], dlf_tr_t[b])
                loss = loss_fn(pred_ij, dr_tr_t[b])
            else:
                pred_ij = model(xi, xj, d2d_tr_t[b], sim_tr_t[b], dlf_tr_t[b], d3d_tr_t[b])
                sparsity = 5e-3 * torch.mean(torch.sigmoid(model.gate_logits))
                loss = loss_fn(pred_ij, dr_tr_t[b]) + sparsity
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

        y_va_pred = _predict_kr_backbone(
            model, variant, feats, bundle_2d, lfer_atoms, inner_va_idx, inner_tr_idx,
            r_inner_vec, y_svm_va, r_std, (sc_2d, sc_lf, sc_3d), env_counters, device,
            k_eval=k_eval, tau=tau,
        )
        va_rmse = float(np.sqrt(np.mean((y_va_true - y_va_pred) ** 2)))
        if va_rmse < best_val_rmse - 1e-5:
            best_val_rmse = va_rmse
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    y_pred_test = _predict_kr_backbone(
        model, variant, feats, bundle_2d, lfer_atoms, test_idx, train_idx,
        r_train_vec, y_test_svm, r_std, (sc_2d, sc_lf, sc_3d), env_counters, device,
        k_eval=k_eval, tau=tau,
    )
    if not np.all(np.isfinite(y_pred_test)):
        raise RuntimeError(f"Non-finite predictions encountered in {variant} (seed={seed})")

    gate_info = {}
    if variant == "KR_FiLM_Gated3DQM":
        with torch.no_grad():
            g = torch.sigmoid(model.gate_logits).detach().cpu().numpy()
            beta = float(torch.abs(model.beta_3d).detach().cpu().item())
        gate_info = {
            "g_aimnet": float(g[0]),
            "g_3dgeom": float(g[1]),
            "g_chiral": float(g[2]),
            "abs_beta_3d": beta,
        }

    return {"y_pred_test": y_pred_test, "gate_info": gate_info, "best_val_rmse": best_val_rmse}


def _evaluate_target(target: str, n_workers: int, force: bool) -> dict[str, Any]:
    out_dir = _out_root() / "per_target"
    json_path = out_dir / f"{target}.json"
    npz_path = out_dir / f"{target}_preds.npz"
    if json_path.exists() and not force:
        cached = json.loads(json_path.read_text(encoding="utf-8"))
        if all(m in cached.get("metrics", {}) for m in ALL_REPORT_MODELS):
            print(f"[CACHE] {target}: loaded existing Path-A modular checkpoint", flush=True)
            return cached

    t0 = time.time()
    ds = load_target_dataset(target)
    feats = extract_or_load_target_features(target, ds.smiles, prefer_aimnet=True, num_confs=5, n_workers=n_workers)
    lfer_atoms = _lfer_atom_matrix(feats)

    w1_npz_path = ROOT_DIR / "refine-logs" / "benchmark30_results" / "per_target" / f"{target}_preds.npz"
    if not w1_npz_path.exists():
        raise FileNotFoundError(f"Missing Wave-1 predictions cache: {w1_npz_path}")
    w1 = np.load(w1_npz_path)

    y_te = ds.y[ds.test_idx]
    c_te = ds.cliff_mol[ds.test_idx]
    assert np.allclose(y_te, w1["y_te"]) and np.array_equal(c_te, w1["c_te"]), f"Split mismatch on {target}"

    seed_metrics: dict[str, list[dict[str, float]]] = {m: [] for m in ALL_REPORT_MODELS}
    seed_preds: dict[str, list[np.ndarray]] = {m: [] for m in ALL_REPORT_MODELS}
    gate_3d_stats: list[dict[str, float]] = []

    for s_idx, seed in enumerate(SEEDS):
        inner_tr_rel, inner_va_rel = make_inner_split(ds.y[ds.train_idx], seed=seed, val_frac=0.15)
        inner_tr_idx = ds.train_idx[inner_tr_rel]
        inner_va_idx = ds.train_idx[inner_va_rel]

        y_svm_te = w1["B1c_RBF_SVM_seeds"][s_idx].astype(np.float64)
        y_a2_te = w1["A2_2D_Delta_seeds"][s_idx].astype(np.float64)
        y_g0_te = w1["G0_RBFSVM_Gate_A2_seeds"][s_idx].astype(np.float64)
        y_g2_te = w1["G2_RBFSVM_PhysGate_L2_seeds"][s_idx].astype(np.float64)

        # Part I Controls
        y_cat_te = run_tabular_baseline("B1_ECFP_Chiral_CatBoost", feats, ds.y, ds.train_idx, ds.test_idx, seed)["y_pred_test"]
        bundle_2d_inner = build_tier_matrices(feats, tier="2d", fit_idx=inner_tr_idx)
        y_mlp_te = _train_direct_mlp_with_inner_split(
            bundle_2d_inner.x_mol, ds.y, ds.train_idx, inner_tr_rel, inner_va_rel, ds.test_idx, seed
        )["y_pred_test"]

        e_svm_cat = 0.5 * y_svm_te + 0.5 * y_cat_te
        e_svm_mlp = 0.5 * y_svm_te + 0.5 * y_mlp_te
        e_svm_a2 = 0.5 * y_svm_te + 0.5 * y_a2_te

        # Part II: Strict Leak-Free OOF Kernel-Residual Formulation (cached once per target/seed)
        oof_inner = _compute_oof_svr(feats.ecfp_chiral, ds.y, inner_tr_idx, seed=seed)
        r_inner_vec = np.zeros_like(ds.y, dtype=np.float64)
        r_inner_vec[inner_tr_idx] = ds.y[inner_tr_idx] - oof_inner

        svr_inner = SVR(kernel="rbf", C=3.0, epsilon=0.1)
        svr_inner.fit(feats.ecfp_chiral[inner_tr_idx], ds.y[inner_tr_idx])
        y_svm_va = svr_inner.predict(feats.ecfp_chiral[inner_va_idx]).astype(np.float64)

        oof_train = _compute_oof_svr(feats.ecfp_chiral, ds.y, ds.train_idx, seed=seed)
        r_train_vec = np.zeros_like(ds.y, dtype=np.float64)
        r_train_vec[ds.train_idx] = ds.y[ds.train_idx] - oof_train

        kr_2d_out = _train_kr_backbone(
            "KR_2D_Delta", feats, lfer_atoms, ds.y, ds.train_idx, inner_tr_idx, inner_va_idx,
            ds.test_idx, r_inner_vec, y_svm_va, r_train_vec, y_svm_te, seed,
        )
        kr_film_out = _train_kr_backbone(
            "KR_FiLM_2DLFER", feats, lfer_atoms, ds.y, ds.train_idx, inner_tr_idx, inner_va_idx,
            ds.test_idx, r_inner_vec, y_svm_va, r_train_vec, y_svm_te, seed,
        )
        kr_3dqm_out = _train_kr_backbone(
            "KR_FiLM_Gated3DQM", feats, lfer_atoms, ds.y, ds.train_idx, inner_tr_idx, inner_va_idx,
            ds.test_idx, r_inner_vec, y_svm_va, r_train_vec, y_svm_te, seed,
        )
        gate_3d_stats.append(kr_3dqm_out["gate_info"])

        preds_map = {
            "B1c_RBF_SVM": y_svm_te,
            "B1_CatBoost": y_cat_te,
            "A1_Direct_MLP": y_mlp_te,
            "A2_2D_Delta": y_a2_te,
            "E_SVM_CatBoost": e_svm_cat,
            "E_SVM_DirectMLP": e_svm_mlp,
            "E_SVM_A2Delta": e_svm_a2,
            "G0_RBFSVM_Gate_A2": y_g0_te,
            "G2_Prev": y_g2_te,
            "KR_2D_Delta": kr_2d_out["y_pred_test"],
            "KR_FiLM_2DLFER": kr_film_out["y_pred_test"],
            "KR_FiLM_Gated3DQM": kr_3dqm_out["y_pred_test"],
        }
        for m_name, yp in preds_map.items():
            seed_metrics[m_name].append(evaluate_predictions(y_te, yp, c_te))
            seed_preds[m_name].append(yp)

    metrics_summary: dict[str, dict[str, float]] = {}
    npz_payload: dict[str, np.ndarray] = {"y_te": y_te, "c_te": c_te}
    for m_name in ALL_REPORT_MODELS:
        rmse_arr = np.asarray([r["rmse"] for r in seed_metrics[m_name]], dtype=np.float64)
        cliff_arr = np.asarray([r["cliff_rmse"] for r in seed_metrics[m_name]], dtype=np.float64)
        ncliff_arr = np.asarray([r["non_cliff_rmse"] for r in seed_metrics[m_name]], dtype=np.float64)
        sp_arr = np.asarray([r["spearman"] for r in seed_metrics[m_name]], dtype=np.float64)
        metrics_summary[m_name] = {
            "rmse_mean": float(np.mean(rmse_arr)),
            "rmse_std": float(np.std(rmse_arr)),
            "cliff_rmse_mean": float(np.mean(cliff_arr)),
            "cliff_rmse_std": float(np.std(cliff_arr)),
            "non_cliff_rmse_mean": float(np.mean(ncliff_arr)),
            "spearman_mean": float(np.mean(sp_arr)),
        }
        stacked = np.stack(seed_preds[m_name], axis=0)
        npz_payload[f"{m_name}_seeds"] = stacked
        npz_payload[f"{m_name}_mean"] = np.mean(stacked, axis=0)

    gate_summary = {
        k: float(np.mean([g[k] for g in gate_3d_stats]))
        for k in ("g_aimnet", "g_3dgeom", "g_chiral", "abs_beta_3d")
    }
    elapsed = round(time.time() - t0, 2)
    result = {
        "target": target,
        "elapsed_sec": elapsed,
        "n_test": int(y_te.size),
        "n_test_cliff": int(np.sum(c_te == 1)),
        "metrics": metrics_summary,
        "gate_3dqm": gate_summary,
    }
    np.savez_compressed(npz_path, **npz_payload)
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    m = metrics_summary
    print(
        f"[DONE {elapsed:.1f}s] {target}: "
        f"SVM={m['B1c_RBF_SVM']['cliff_rmse_mean']:.4f} | "
        f"E_Cat={m['E_SVM_CatBoost']['cliff_rmse_mean']:.4f} | "
        f"E_MLP={m['E_SVM_DirectMLP']['cliff_rmse_mean']:.4f} | "
        f"E_A2={m['E_SVM_A2Delta']['cliff_rmse_mean']:.4f} | "
        f"KR_2D={m['KR_2D_Delta']['cliff_rmse_mean']:.4f} | "
        f"KR_FiLM={m['KR_FiLM_2DLFER']['cliff_rmse_mean']:.4f} | "
        f"KR_3DQM={m['KR_FiLM_Gated3DQM']['cliff_rmse_mean']:.4f} "
        f"(g=[{gate_summary['g_aimnet']:.2f},{gate_summary['g_3dgeom']:.2f},{gate_summary['g_chiral']:.2f}], |beta|={gate_summary['abs_beta_3d']:.3f})",
        flush=True,
    )
    return result


def _wilcoxon_pair(df_wide: pd.DataFrame, col_a: str, col_b: str) -> dict[str, Any]:
    diff = (df_wide[col_a] - df_wide[col_b]).to_numpy(dtype=np.float64)
    wins = int(np.sum(diff < -1e-9))
    ties = int(np.sum(np.abs(diff) <= 1e-9))
    losses = int(np.sum(diff > 1e-9))
    try:
        _, pval = stats.wilcoxon(df_wide[col_a], df_wide[col_b], alternative="two-sided")
        p = float(pval)
    except Exception:
        p = float("nan")
    return {
        "n": int(diff.size),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "mean_diff": float(np.mean(diff)),
        "median_diff": float(np.median(diff)),
        "p_value": p,
    }


def _plot_comparison(df_long: pd.DataFrame, results: list[dict[str, Any]], fig_path: Path) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 5.8))

    bar_models = [
        ("B1c_RBF_SVM", "B1c RBF-SVM", "#888888"),
        ("E_SVM_CatBoost", "0.5*SVM+Cat", "#56B4E9"),
        ("E_SVM_DirectMLP", "0.5*SVM+MLP", "#009E73"),
        ("E_SVM_A2Delta", "0.5*SVM+A2", "#E69F00"),
        ("G0_RBFSVM_Gate_A2", "G0 Gate", "#CC79A7"),
        ("G2_Prev", "G2_Prev", "#999999"),
        ("KR_2D_Delta", "KR 2D", "#0072B2"),
        ("KR_FiLM_2DLFER", "KR FiLM 2D-LFER", "#D55E00"),
        ("KR_FiLM_Gated3DQM", "KR FiLM+3DQM", "#B22222"),
    ]
    agg = df_long.groupby("model")[["cliff_rmse_mean", "rmse_mean"]].mean()
    x = np.arange(len(bar_models))
    w = 0.38
    cliff_vals = [float(agg.loc[m, "cliff_rmse_mean"]) for m, _, _ in bar_models]
    all_vals = [float(agg.loc[m, "rmse_mean"]) for m, _, _ in bar_models]
    cols = [c for _, _, c in bar_models]
    lbls = [l for _, l, _ in bar_models]

    ax1.bar(x - w / 2, cliff_vals, width=w, color=cols, edgecolor="black", linewidth=0.6, label="Cliff RMSE")
    ax1.bar(x + w / 2, all_vals, width=w, color=cols, alpha=0.45, edgecolor="black", linewidth=0.6, hatch="//", label="Overall RMSE")
    for xi, cv, av in zip(x, cliff_vals, all_vals):
        ax1.text(xi - w / 2, cv + 0.002, f"{cv:.4f}", ha="center", va="bottom", fontsize=7, rotation=45)
        ax1.text(xi + w / 2, av + 0.002, f"{av:.4f}", ha="center", va="bottom", fontsize=7, rotation=45)
    ax1.set_xticks(x)
    ax1.set_xticklabels(lbls, rotation=28, ha="right", fontsize=8)
    ax1.set_ylim(0.58, max(cliff_vals) + 0.03)
    ax1.set_ylabel("10-Target Mean RMSE (Lower = Better)")
    ax1.set_title("Path A: Kill-Test & Kernel-Residual Modular Comparison", fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.legend(frameon=False, fontsize=8, loc="upper right")

    # Right panel: Per-target deltas vs E_SVM_DirectMLP & E_SVM_A2Delta + learned 3DQM gates
    rows = []
    for r in results:
        mm = r["metrics"]
        g = r["gate_3dqm"]
        rows.append(
            {
                "target": r["target"],
                "d_film_a2": mm["KR_FiLM_2DLFER"]["cliff_rmse_mean"] - mm["E_SVM_A2Delta"]["cliff_rmse_mean"],
                "d_3d_a2": mm["KR_FiLM_Gated3DQM"]["cliff_rmse_mean"] - mm["E_SVM_A2Delta"]["cliff_rmse_mean"],
                "d_film_mlp": mm["KR_FiLM_2DLFER"]["cliff_rmse_mean"] - mm["E_SVM_DirectMLP"]["cliff_rmse_mean"],
                "gate_lbl": f"g3d={g['g_3dgeom']:.2f}, |b|={g['abs_beta_3d']:.2f}",
            }
        )
    rdf = pd.DataFrame(rows).sort_values("d_film_a2", ascending=True).reset_index(drop=True)
    yp = np.arange(len(rdf))
    bh = 0.27
    ax2.barh(yp - bh, rdf["d_film_a2"], height=bh, color="#D55E00", label="KR_FiLM - E_SVM_A2 (<0 win)")
    ax2.barh(yp, rdf["d_3d_a2"], height=bh, color="#B22222", alpha=0.85, label="KR_3DQM - E_SVM_A2 (<0 win)")
    ax2.barh(yp + bh, rdf["d_film_mlp"], height=bh, color="#009E73", alpha=0.85, label="KR_FiLM - E_SVM_MLP (<0 win)")
    ax2.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax2.set_yticks(yp)
    ax2.set_yticklabels([f"{t} ({g})" for t, g in zip(rdf["target"], rdf["gate_lbl"])], fontsize=7.5)
    ax2.set_xlabel("Per-Target Delta Cliff RMSE (< 0 = Modular KR Better)")
    ax2.set_title("Per-Target KR Gains & Learned 3D/QM Gate Shrinkage", fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.3)
    ax2.legend(frameon=False, fontsize=7.5, loc="lower right")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Path A Kill-Test and Modular Architecture Exploration.")
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--force", action="store_true", default=False)
    args = parser.parse_args()

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Path A Modular Exploration on {len(WAVE_1_TARGETS)} targets", flush=True)
    t0 = time.time()
    results = [_evaluate_target(tgt, n_workers=args.n_workers, force=args.force) for tgt in WAVE_1_TARGETS]
    elapsed = time.time() - t0

    long_rows = []
    wide_rows = []
    for r in results:
        tgt = r["target"]
        wrow: dict[str, Any] = {"target": tgt, **r["gate_3dqm"]}
        for m_name in ALL_REPORT_MODELS:
            mm = r["metrics"][m_name]
            long_rows.append({"target": tgt, "model": m_name, **mm})
            wrow[f"{m_name}_cliff"] = mm["cliff_rmse_mean"]
            wrow[f"{m_name}_rmse"] = mm["rmse_mean"]
        wide_rows.append(wrow)

    df_long = pd.DataFrame(long_rows)
    df_wide = pd.DataFrame(wide_rows)
    csv_path = _out_root() / "path_a_summary.csv"
    df_long.to_csv(csv_path, index=False)

    fig_path = _fig_dir() / "path_a_modular_comparison.png"
    _plot_comparison(df_long, results, fig_path)

    kill_tests = [
        ("Cliff: E_SVM_A2Delta vs B1c_RBF_SVM", "E_SVM_A2Delta_cliff", "B1c_RBF_SVM_cliff"),
        ("Cliff: E_SVM_A2Delta vs E_SVM_DirectMLP", "E_SVM_A2Delta_cliff", "E_SVM_DirectMLP_cliff"),
        ("Cliff: E_SVM_A2Delta vs E_SVM_CatBoost", "E_SVM_A2Delta_cliff", "E_SVM_CatBoost_cliff"),
        ("Overall: E_SVM_A2Delta vs E_SVM_DirectMLP", "E_SVM_A2Delta_rmse", "E_SVM_DirectMLP_rmse"),
        ("Overall: E_SVM_A2Delta vs E_SVM_CatBoost", "E_SVM_A2Delta_rmse", "E_SVM_CatBoost_rmse"),
    ]
    mod_tests = [
        ("Cliff: KR_2D_Delta vs B1c_RBF_SVM", "KR_2D_Delta_cliff", "B1c_RBF_SVM_cliff"),
        ("Cliff: KR_2D_Delta vs E_SVM_A2Delta", "KR_2D_Delta_cliff", "E_SVM_A2Delta_cliff"),
        ("Cliff: KR_FiLM_2DLFER vs B1c_RBF_SVM", "KR_FiLM_2DLFER_cliff", "B1c_RBF_SVM_cliff"),
        ("Cliff: KR_FiLM_2DLFER vs E_SVM_A2Delta", "KR_FiLM_2DLFER_cliff", "E_SVM_A2Delta_cliff"),
        ("Cliff: KR_FiLM_2DLFER vs E_SVM_DirectMLP", "KR_FiLM_2DLFER_cliff", "E_SVM_DirectMLP_cliff"),
        ("Cliff: KR_FiLM_2DLFER vs E_SVM_CatBoost", "KR_FiLM_2DLFER_cliff", "E_SVM_CatBoost_cliff"),
        ("Cliff: KR_FiLM_2DLFER vs KR_2D_Delta", "KR_FiLM_2DLFER_cliff", "KR_2D_Delta_cliff"),
        ("Cliff: KR_FiLM_Gated3DQM vs KR_FiLM_2DLFER", "KR_FiLM_Gated3DQM_cliff", "KR_FiLM_2DLFER_cliff"),
        ("Cliff: KR_FiLM_Gated3DQM vs E_SVM_A2Delta", "KR_FiLM_Gated3DQM_cliff", "E_SVM_A2Delta_cliff"),
        ("Overall: KR_FiLM_2DLFER vs B1c_RBF_SVM", "KR_FiLM_2DLFER_rmse", "B1c_RBF_SVM_rmse"),
        ("Overall: KR_FiLM_2DLFER vs E_SVM_A2Delta", "KR_FiLM_2DLFER_rmse", "E_SVM_A2Delta_rmse"),
        ("Overall: KR_FiLM_2DLFER vs E_SVM_DirectMLP", "KR_FiLM_2DLFER_rmse", "E_SVM_DirectMLP_rmse"),
        ("Overall: KR_FiLM_2DLFER vs E_SVM_CatBoost", "KR_FiLM_2DLFER_rmse", "E_SVM_CatBoost_rmse"),
        ("Overall: KR_FiLM_Gated3DQM vs KR_FiLM_2DLFER", "KR_FiLM_Gated3DQM_rmse", "KR_FiLM_2DLFER_rmse"),
    ]

    agg = df_long.groupby("model")[["rmse_mean", "cliff_rmse_mean", "non_cliff_rmse_mean", "spearman_mean"]].mean()
    md_path = ROOT_DIR / "refine-logs" / "PATH_A_MODULAR_EVAL.md"
    lines = [
        "# Path A: Attribution Kill-Test & Composable Modular Exploration",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **Targets**: {len(results)} | **Elapsed**: {elapsed:.1f}s",
        "",
        "## 1. Path A Kill-Test: Pairwise Delta vs Heterogeneous Ensembles",
        "",
        "### 10-Target Mean Metrics",
        "",
        "| Model | Overall RMSE | Cliff RMSE | Non-Cliff RMSE | Spearman |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for m_name in ("B1c_RBF_SVM", "B1_CatBoost", "A1_Direct_MLP", "A2_2D_Delta", "E_SVM_CatBoost", "E_SVM_DirectMLP", "E_SVM_A2Delta"):
        r = agg.loc[m_name]
        lines.append(f"| `{m_name}` | {r['rmse_mean']:.4f} | {r['cliff_rmse_mean']:.4f} | {r['non_cliff_rmse_mean']:.4f} | {r['spearman_mean']:.4f} |")

    lines.extend([
        "",
        "### Kill-Test Cross-Target Wilcoxon Signed-Rank (N=10)",
        "",
        "| Comparison | Win / Tie / Loss | Mean Diff | Median Diff | Two-Sided p |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    for lbl, ca, cb in kill_tests:
        st = _wilcoxon_pair(df_wide, ca, cb)
        lines.append(f"| {lbl} | {st['wins']} / {st['ties']} / {st['losses']} | {st['mean_diff']:+.5f} | {st['median_diff']:+.5f} | `{st['p_value']:.4e}` |")

    lines.extend([
        "",
        "## 2. Modular Architecture Exploration (Kernel-Residual + FiLM 2D-LFER + Sparse-Gated 3D/QM)",
        "",
        "| Model | Overall RMSE | Cliff RMSE | Non-Cliff RMSE | Spearman |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    for m_name in ("B1c_RBF_SVM", "E_SVM_CatBoost", "E_SVM_DirectMLP", "E_SVM_A2Delta", "G0_RBFSVM_Gate_A2", "G2_Prev", "KR_2D_Delta", "KR_FiLM_2DLFER", "KR_FiLM_Gated3DQM"):
        r = agg.loc[m_name]
        lines.append(f"| `{m_name}` | {r['rmse_mean']:.4f} | {r['cliff_rmse_mean']:.4f} | {r['non_cliff_rmse_mean']:.4f} | {r['spearman_mean']:.4f} |")

    lines.extend([
        "",
        "### Modular Cross-Target Wilcoxon Signed-Rank (N=10)",
        "",
        "| Comparison | Win / Tie / Loss | Mean Diff | Median Diff | Two-Sided p |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    for lbl, ca, cb in mod_tests:
        st = _wilcoxon_pair(df_wide, ca, cb)
        lines.append(f"| {lbl} | {st['wins']} / {st['ties']} / {st['losses']} | {st['mean_diff']:+.5f} | {st['median_diff']:+.5f} | `{st['p_value']:.4e}` |")

    lines.extend([
        "",
        "## 3. Per-Target Cliff RMSE Matrix & Learned 3D/QM Gate Shrinkage",
        "",
        "| Target | B1c_SVM | E_SVM_Cat | E_SVM_MLP | E_SVM_A2 | G2_Prev | KR_2D | KR_FiLM_2DLFER | KR_Gated3DQM | g_aimnet | g_3dgeom | g_chiral | |beta_3d| |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in results:
        mm = r["metrics"]
        g = r["gate_3dqm"]
        lines.append(
            f"| `{r['target']}` | {mm['B1c_RBF_SVM']['cliff_rmse_mean']:.4f} | "
            f"{mm['E_SVM_CatBoost']['cliff_rmse_mean']:.4f} | {mm['E_SVM_DirectMLP']['cliff_rmse_mean']:.4f} | "
            f"{mm['E_SVM_A2Delta']['cliff_rmse_mean']:.4f} | {mm['G2_Prev']['cliff_rmse_mean']:.4f} | "
            f"{mm['KR_2D_Delta']['cliff_rmse_mean']:.4f} | "
            f"{mm['KR_FiLM_2DLFER']['cliff_rmse_mean']:.4f}±{mm['KR_FiLM_2DLFER']['cliff_rmse_std']:.4f} | "
            f"{mm['KR_FiLM_Gated3DQM']['cliff_rmse_mean']:.4f}±{mm['KR_FiLM_Gated3DQM']['cliff_rmse_std']:.4f} | "
            f"{g['g_aimnet']:.3f} | {g['g_3dgeom']:.3f} | {g['g_chiral']:.3f} | {g['abs_beta_3d']:.4f} |"
        )

    lines.extend([
        "",
        f"Summary CSV: `{csv_path.as_posix()}`",
        f"Figure: `{fig_path.as_posix()}`",
    ])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved {md_path}", flush=True)


if __name__ == "__main__":
    main()

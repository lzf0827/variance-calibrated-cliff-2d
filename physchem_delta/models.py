"""Tabular baselines, antisymmetric PairwiseDeltaNet, and similarity-gated OOD blending."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy import optimize
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .featurize import TargetFeatures, compute_local_pair_deltas, precompute_mol_env_counters


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_tanimoto_matrix(fp_a: np.ndarray, fp_b: np.ndarray) -> np.ndarray:
    """Fast exact Tanimoto similarity matrix between binary fingerprint arrays."""
    a = fp_a.astype(np.float32, copy=False)
    b = fp_b.astype(np.float32, copy=False)
    inter = a @ b.T
    sum_a = a.sum(axis=1, keepdims=True)
    sum_b = b.sum(axis=1, keepdims=True).T
    union = sum_a + sum_b - inter
    sim = np.where(union > 0.0, inter / np.maximum(union, 1e-8), 0.0)
    return np.clip(sim, 0.0, 1.0).astype(np.float32)


def make_inner_split(y_train: np.ndarray, seed: int, val_frac: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic quantile-stratified inner train/val split strictly inside train."""
    n = y_train.size
    n_bins = min(5, max(2, n // 40))
    bins = np.zeros((n,), dtype=int)
    if n >= 20:
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(y_train, qs)
        edges[0] -= 1e-6
        edges[-1] += 1e-6
        bins = np.digitize(y_train, edges[1:-1])
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        tr_rel, va_rel = next(sss.split(np.zeros((n, 1)), bins))
        return tr_rel, va_rel
    except Exception:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_val = max(10, int(round(n * val_frac)))
        return perm[n_val:], perm[:n_val]


@dataclass
class PreprocessedFeatureBundle:
    x_mol: np.ndarray        # (N, D_mol)
    x_cont_scaled: np.ndarray  # (N, D_cont) continuous features scaled strictly on fit_idx
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray


def build_tier_matrices(
    feats: TargetFeatures,
    tier: str,
    fit_idx: np.ndarray,
) -> PreprocessedFeatureBundle:
    """
    Assemble molecular feature matrix for a given ablation tier and fit scaler ONLY on fit_idx.
    Tiers:
      - '2d_nochiral': ecfp_nochiral
      - '2d': ecfp_chiral + maccs + desc2d
      - '2d3d': 2d + feat_3d
      - 'physchem': 2d + feat_3d + feat_elec
    """
    if tier == "2d_nochiral":
        x_bin = feats.ecfp_nochiral
        x_cont_raw = np.zeros((feats.ecfp_nochiral.shape[0], 0), dtype=np.float32)
    elif tier == "2d":
        x_bin = np.concatenate([feats.ecfp_chiral, feats.maccs], axis=1)
        x_cont_raw = feats.desc2d
    elif tier == "2d3d":
        x_bin = np.concatenate([feats.ecfp_chiral, feats.maccs], axis=1)
        x_cont_raw = np.concatenate([feats.desc2d, feats.feat_3d], axis=1)
    elif tier == "physchem":
        x_bin = np.concatenate([feats.ecfp_chiral, feats.maccs], axis=1)
        x_cont_raw = np.concatenate([feats.desc2d, feats.feat_3d, feats.feat_elec], axis=1)
    else:
        raise ValueError(f"Unknown feature tier: {tier}")

    if x_cont_raw.shape[1] > 0:
        scaler = StandardScaler()
        scaler.fit(x_cont_raw[fit_idx])
        scale = np.where(scaler.scale_ < 1e-8, 1.0, scaler.scale_)
        x_cont_scaled = ((x_cont_raw - scaler.mean_) / scale).astype(np.float32)
        x_cont_scaled = np.clip(np.nan_to_num(x_cont_scaled, nan=0.0), -10.0, 10.0).astype(np.float32)
        x_mol = np.concatenate([x_bin.astype(np.float32), x_cont_scaled], axis=1)
        return PreprocessedFeatureBundle(
            x_mol=x_mol,
            x_cont_scaled=x_cont_scaled,
            scaler_mean=scaler.mean_.astype(np.float32),
            scaler_scale=scale.astype(np.float32),
        )
    else:
        return PreprocessedFeatureBundle(
            x_mol=x_bin.astype(np.float32),
            x_cont_scaled=x_cont_raw,
            scaler_mean=np.zeros((0,), dtype=np.float32),
            scaler_scale=np.ones((0,), dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# Tabular Baselines (M1)
# ---------------------------------------------------------------------------

def run_tabular_baseline(
    model_name: str,
    feats: TargetFeatures,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    """Run B0, B1, B1b, or B1c strictly fit on train_idx."""
    set_global_seed(seed)
    tier = "2d_nochiral" if model_name == "B0_ECFP_NoChiral_CatBoost" else "2d"
    bundle = build_tier_matrices(feats, tier=tier, fit_idx=train_idx)

    x_tr = bundle.x_mol[train_idx]
    y_tr = y[train_idx]
    x_te = bundle.x_mol[test_idx]

    if model_name in ("B0_ECFP_NoChiral_CatBoost", "B1_ECFP_Chiral_CatBoost"):
        from catboost import CatBoostRegressor

        tr_rel, va_rel = make_inner_split(y_tr, seed=seed, val_frac=0.15)
        model = CatBoostRegressor(
            iterations=1800,
            learning_rate=0.04,
            depth=6,
            l2_leaf_reg=5.0,
            random_seed=seed,
            loss_function="RMSE",
            eval_metric="RMSE",
            verbose=0,
            thread_count=-1,
        )
        model.fit(
            x_tr[tr_rel],
            y_tr[tr_rel],
            eval_set=(x_tr[va_rel], y_tr[va_rel]),
            early_stopping_rounds=100,
            verbose=False,
        )
        y_pred_test = model.predict(x_te).astype(np.float64)
        extra = {"best_iteration": int(model.get_best_iteration() or 0)}

    elif model_name == "B1b_ECFP_Chiral_LGBM":
        import lightgbm as lgb

        tr_rel, va_rel = make_inner_split(y_tr, seed=seed, val_frac=0.15)
        model = lgb.LGBMRegressor(
            n_estimators=1800,
            learning_rate=0.03,
            num_leaves=31,
            subsample_for_bin=50000,
            colsample_bytree=0.7,
            subsample=0.8,
            reg_lambda=2.0,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(
            x_tr[tr_rel],
            y_tr[tr_rel],
            eval_set=[(x_tr[va_rel], y_tr[va_rel])],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        y_pred_test = model.predict(x_te).astype(np.float64)
        extra = {"best_iteration": int(model.best_iteration_ or 0)}

    elif model_name == "B1c_ECFP_SVM":
        # MoleculeACE classic ECFP + RBF SVR
        x_fp_tr = feats.ecfp_chiral[train_idx]
        x_fp_te = feats.ecfp_chiral[test_idx]
        model = SVR(kernel="rbf", C=3.0, epsilon=0.1)
        model.fit(x_fp_tr, y_tr)
        y_pred_test = model.predict(x_fp_te).astype(np.float64)
        extra = {"note": "deterministic RBF SVR on chiral ECFP4"}
    else:
        raise ValueError(f"Unknown tabular model: {model_name}")

    return {
        "y_pred_test": y_pred_test,
        "peak_vram_mb": 0.0,
        "extra": extra,
    }


# ---------------------------------------------------------------------------
# Neural Networks (PyTorch CUDA)
# ---------------------------------------------------------------------------

class DirectMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class AntisymmetricPairwiseDeltaNet(nn.Module):
    """
    Siamese molecular encoder + antisymmetric pairwise delta head.
    Guarantees f_delta(i, j) = -f_delta(j, i) and f_delta(i, i) = 0 identically.
    """

    def __init__(self, mol_dim: int, pair_diff_dim: int, emb_dim: int = 128, dropout: float = 0.25):
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
        # Input to pair_head: [h_i, h_j, h_i - h_j, h_i * h_j, pair_diff_ij, sim_ij]
        pair_in_dim = emb_dim * 4 + pair_diff_dim + 1
        self.pair_head = nn.Sequential(
            nn.Linear(pair_in_dim, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(192, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        diff_ij: torch.Tensor,
        sim_ij: torch.Tensor,
    ) -> torch.Tensor:
        h_i = self.mol_encoder(x_i)
        h_j = self.mol_encoder(x_j)
        prod = h_i * h_j
        sim_col = sim_ij.unsqueeze(-1)

        u_ij = torch.cat([h_i, h_j, h_i - h_j, prod, diff_ij, sim_col], dim=-1)
        u_ji = torch.cat([h_j, h_i, h_j - h_i, prod, -diff_ij, sim_col], dim=-1)

        out_ij = self.pair_head(u_ij).squeeze(-1)
        out_ji = self.pair_head(u_ji).squeeze(-1)
        return 0.5 * (out_ij - out_ji)


def _train_direct_mlp_with_inner_split(
    x_all: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    inner_tr_rel: np.ndarray,
    inner_va_rel: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    max_epochs: int = 120,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> dict[str, Any]:
    """Train DirectMLP strictly on inner_tr, early-stop on inner_val, return inner_val & test preds."""
    set_global_seed(seed)
    device = get_device()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    inner_tr_idx = train_idx[inner_tr_rel]
    inner_va_idx = train_idx[inner_va_rel]

    y_mean = float(np.mean(y[inner_tr_idx]))
    y_std = float(max(np.std(y[inner_tr_idx]), 1e-6))

    x_tr_t = torch.tensor(x_all[inner_tr_idx], dtype=torch.float32, device=device)
    y_tr_t = torch.tensor((y[inner_tr_idx] - y_mean) / y_std, dtype=torch.float32, device=device)
    x_va_t = torch.tensor(x_all[inner_va_idx], dtype=torch.float32, device=device)
    y_va_true = y[inner_va_idx]
    x_te_t = torch.tensor(x_all[test_idx], dtype=torch.float32, device=device)

    model = DirectMLP(in_dim=x_all.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    loss_fn = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience = 18
    wait = 0

    n_tr = x_tr_t.shape[0]
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        for start in range(0, n_tr, batch_size):
            b = perm[start : start + batch_size]
            opt.zero_grad(set_to_none=True)
            pred = model(x_tr_t[b])
            loss = loss_fn(pred, y_tr_t[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            va_pred = model(x_va_t).cpu().numpy() * y_std + y_mean
            va_rmse = float(np.sqrt(np.mean((y_va_true - va_pred) ** 2)))
        if va_rmse < best_val_rmse - 1e-5:
            best_val_rmse = va_rmse
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred_val = (model(x_va_t).cpu().numpy() * y_std + y_mean).astype(np.float64)
        y_pred_test = (model(x_te_t).cpu().numpy() * y_std + y_mean).astype(np.float64)

    peak_vram_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else 0.0
    return {
        "y_pred_val": y_pred_val,
        "y_pred_test": y_pred_test,
        "best_val_rmse": best_val_rmse,
        "peak_vram_mb": peak_vram_mb,
    }


def _build_pair_diff_features(
    feats: TargetFeatures,
    bundle: PreprocessedFeatureBundle,
    tier: str,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    env_counters: list[Any],
) -> np.ndarray:
    """
    Build antisymmetric continuous perturbation vector for pairs (i, j),
    strictly isolated by tier for clean R02 attribution.
    """
    parts: list[np.ndarray] = []
    if bundle.x_cont_scaled.shape[1] > 0:
        parts.append((bundle.x_cont_scaled[i_idx] - bundle.x_cont_scaled[j_idx]).astype(np.float32))

    if tier == "2d":
        # 2D-only local fingerprint bit diff summary: count added vs removed ECFP4 bits (antisymmetric)
        fp_i = feats.ecfp_chiral[i_idx]
        fp_j = feats.ecfp_chiral[j_idx]
        n_added = np.sum((fp_i > 0.5) & (fp_j < 0.5), axis=1, keepdims=True).astype(np.float32)
        n_removed = np.sum((fp_j > 0.5) & (fp_i < 0.5), axis=1, keepdims=True).astype(np.float32)
        parts.append(n_added - n_removed)

    elif tier == "2d3d":
        local_3d = compute_local_pair_deltas(
            i_idx,
            j_idx,
            feats.atom_ptr,
            feats.atom_env_hash,
            feats.atom_3d_props,
            env_counters=env_counters,
        )
        parts.append(local_3d)

    elif tier == "physchem":
        local_3d_elec = compute_local_pair_deltas(
            i_idx,
            j_idx,
            feats.atom_ptr,
            feats.atom_env_hash,
            np.concatenate([feats.atom_3d_props, feats.atom_elec_props], axis=1),
            env_counters=env_counters,
        )
        parts.append(local_3d_elec)
    else:
        raise ValueError(f"Unsupported delta tier: {tier}")

    return np.concatenate(parts, axis=1).astype(np.float32)


def _select_topk_anchors(
    sim_query_to_pool: np.ndarray,
    pool_global_idx: np.ndarray,
    k: int,
    exclude_self_global_idx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (anchor_global_idx [N_q, k], anchor_sims [N_q, k]) without self-matches."""
    sim = sim_query_to_pool.copy()
    if exclude_self_global_idx is not None:
        pool_map = {int(g): pos for pos, g in enumerate(pool_global_idx.tolist())}
        for q_row, q_g in enumerate(exclude_self_global_idx.tolist()):
            if int(q_g) in pool_map:
                sim[q_row, pool_map[int(q_g)]] = -1.0

    k_eff = min(k, pool_global_idx.size)
    top_pos = np.argpartition(-sim, kth=k_eff - 1, axis=1)[:, :k_eff]
    top_sims = np.take_along_axis(sim, top_pos, axis=1)
    order = np.argsort(-top_sims, axis=1)
    top_pos = np.take_along_axis(top_pos, order, axis=1)
    top_sims = np.take_along_axis(top_sims, order, axis=1)
    return pool_global_idx[top_pos], np.clip(top_sims, 0.0, 1.0).astype(np.float32)


def _predict_delta_from_anchors(
    model: AntisymmetricPairwiseDeltaNet,
    feats: TargetFeatures,
    bundle: PreprocessedFeatureBundle,
    tier: str,
    query_idx: np.ndarray,
    anchor_pool_idx: np.ndarray,
    y: np.ndarray,
    y_std: float,
    diff_mean: np.ndarray,
    diff_scale: np.ndarray,
    env_counters: list[Any],
    k_eval: int,
    tau: float,
    device: torch.device,
    exclude_self: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict query labels by retrieving top-K anchors from anchor_pool_idx."""
    sim_mat = compute_tanimoto_matrix(feats.ecfp_chiral[query_idx], feats.ecfp_chiral[anchor_pool_idx])
    anchor_g_idx, anchor_sims = _select_topk_anchors(
        sim_mat,
        anchor_pool_idx,
        k=k_eval,
        exclude_self_global_idx=query_idx if exclude_self else None,
    )
    s_max = anchor_sims[:, 0].copy()

    n_q, k_eff = anchor_g_idx.shape
    q_flat = np.repeat(query_idx, k_eff)
    a_flat = anchor_g_idx.reshape(-1)
    s_flat = anchor_sims.reshape(-1)

    raw_diff = _build_pair_diff_features(feats, bundle, tier, q_flat, a_flat, env_counters)
    scaled_diff = np.clip((raw_diff - diff_mean) / diff_scale, -10.0, 10.0).astype(np.float32)

    model.eval()
    preds_scaled = []
    batch_sz = 2048
    with torch.no_grad():
        for start in range(0, q_flat.size, batch_sz):
            sl = slice(start, start + batch_sz)
            xi_t = torch.tensor(bundle.x_mol[q_flat[sl]], dtype=torch.float32, device=device)
            xj_t = torch.tensor(bundle.x_mol[a_flat[sl]], dtype=torch.float32, device=device)
            df_t = torch.tensor(scaled_diff[sl], dtype=torch.float32, device=device)
            sm_t = torch.tensor(s_flat[sl], dtype=torch.float32, device=device)
            out = model(xi_t, xj_t, df_t, sm_t).cpu().numpy()
            preds_scaled.append(out)

    delta_pred = (np.concatenate(preds_scaled, axis=0) * y_std).reshape(n_q, k_eff)
    y_anchors = y[anchor_g_idx]

    # Temperature-scaled softmax weights over Tanimoto similarity
    logits = tau * anchor_sims
    logits = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    weights = weights / np.sum(weights, axis=1, keepdims=True)

    y_hat = np.sum(weights * (y_anchors + delta_pred), axis=1).astype(np.float64)
    return y_hat, s_max.astype(np.float64)


def run_pairwise_delta_model(
    tier: str,
    feats: TargetFeatures,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    k_train: int = 15,
    k_eval: int = 10,
    tau: float = 10.0,
    max_epochs: int = 80,
    batch_size: int = 512,
    lr: float = 8e-4,
) -> dict[str, Any]:
    """
    Train AntisymmetricPairwiseDeltaNet strictly inside train_idx (inner_tr/inner_val),
    then predict test_idx using train_idx as anchor pool.
    """
    set_global_seed(seed)
    device = get_device()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    inner_tr_rel, inner_va_rel = make_inner_split(y[train_idx], seed=seed, val_frac=0.15)
    inner_tr_idx = train_idx[inner_tr_rel]
    inner_va_idx = train_idx[inner_va_rel]

    # Fit molecular scaler strictly on inner_tr_idx
    bundle = build_tier_matrices(feats, tier=tier, fit_idx=inner_tr_idx)
    env_counters = precompute_mol_env_counters(feats.atom_ptr, feats.atom_env_hash)

    # Build training pairs strictly within inner_tr_idx
    sim_tr = compute_tanimoto_matrix(feats.ecfp_chiral[inner_tr_idx], feats.ecfp_chiral[inner_tr_idx])
    nbr_g, nbr_sims = _select_topk_anchors(
        sim_tr, inner_tr_idx, k=k_train, exclude_self_global_idx=inner_tr_idx
    )
    pair_i = np.repeat(inner_tr_idx, nbr_g.shape[1])
    pair_j = nbr_g.reshape(-1)
    pair_s = nbr_sims.reshape(-1)

    y_std = float(max(np.std(y[inner_tr_idx]), 1e-6))
    pair_dy = ((y[pair_i] - y[pair_j]) / y_std).astype(np.float32)

    raw_diff_tr = _build_pair_diff_features(feats, bundle, tier, pair_i, pair_j, env_counters)
    # Zero-mean antisymmetric scaling: scale by std only (preserves zero at identical pairs!)
    diff_scale = np.std(raw_diff_tr, axis=0)
    diff_scale = np.where(diff_scale < 1e-6, 1.0, diff_scale).astype(np.float32)
    diff_mean = np.zeros_like(diff_scale, dtype=np.float32)
    scaled_diff_tr = np.clip(raw_diff_tr / diff_scale, -10.0, 10.0).astype(np.float32)

    model = AntisymmetricPairwiseDeltaNet(
        mol_dim=bundle.x_mol.shape[1],
        pair_diff_dim=scaled_diff_tr.shape[1],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    loss_fn = nn.HuberLoss(delta=1.0)

    # Pre-move molecular feature matrix to GPU for compact fast gather
    x_mol_all_t = torch.tensor(bundle.x_mol, dtype=torch.float32, device=device)
    pair_i_t = torch.tensor(pair_i, dtype=torch.long, device=device)
    pair_j_t = torch.tensor(pair_j, dtype=torch.long, device=device)
    diff_tr_t = torch.tensor(scaled_diff_tr, dtype=torch.float32, device=device)
    sim_tr_t = torch.tensor(pair_s, dtype=torch.float32, device=device)
    dy_tr_t = torch.tensor(pair_dy, dtype=torch.float32, device=device)

    best_val_rmse = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience = 12
    wait = 0
    n_pairs = pair_i.size

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_pairs, device=device)
        for start in range(0, n_pairs, batch_size):
            b = perm[start : start + batch_size]
            opt.zero_grad(set_to_none=True)
            pred_dy = model(
                x_mol_all_t[pair_i_t[b]],
                x_mol_all_t[pair_j_t[b]],
                diff_tr_t[b],
                sim_tr_t[b],
            )
            loss = loss_fn(pred_dy, dy_tr_t[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

        # Evaluate on inner_val using inner_tr strictly as anchors
        y_va_pred, _ = _predict_delta_from_anchors(
            model=model,
            feats=feats,
            bundle=bundle,
            tier=tier,
            query_idx=inner_va_idx,
            anchor_pool_idx=inner_tr_idx,
            y=y,
            y_std=y_std,
            diff_mean=diff_mean,
            diff_scale=diff_scale,
            env_counters=env_counters,
            k_eval=k_eval,
            tau=tau,
            device=device,
            exclude_self=False,
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
    # Final inner_val predictions (strictly using inner_tr anchors) for OOD gate fitting
    y_pred_val, s_max_val = _predict_delta_from_anchors(
        model=model,
        feats=feats,
        bundle=bundle,
        tier=tier,
        query_idx=inner_va_idx,
        anchor_pool_idx=inner_tr_idx,
        y=y,
        y_std=y_std,
        diff_mean=diff_mean,
        diff_scale=diff_scale,
        env_counters=env_counters,
        k_eval=k_eval,
        tau=tau,
        device=device,
        exclude_self=False,
    )
    # Final test predictions using all train_idx as anchor pool (zero overlap with test_idx)
    y_pred_test, s_max_test = _predict_delta_from_anchors(
        model=model,
        feats=feats,
        bundle=bundle,
        tier=tier,
        query_idx=test_idx,
        anchor_pool_idx=train_idx,
        y=y,
        y_std=y_std,
        diff_mean=diff_mean,
        diff_scale=diff_scale,
        env_counters=env_counters,
        k_eval=k_eval,
        tau=tau,
        device=device,
        exclude_self=False,
    )

    peak_vram_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else 0.0
    return {
        "y_pred_val": y_pred_val,
        "s_max_val": s_max_val,
        "inner_va_idx": inner_va_idx,
        "inner_tr_idx": inner_tr_idx,
        "y_pred_test": y_pred_test,
        "s_max_test": s_max_test,
        "best_val_rmse": best_val_rmse,
        "peak_vram_mb": peak_vram_mb,
        "bundle": bundle,
    }


def run_direct_neural_model(
    tier: str,
    feats: TargetFeatures,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    """Train DirectMLP on the specified tier using the exact same inner split."""
    inner_tr_rel, inner_va_rel = make_inner_split(y[train_idx], seed=seed, val_frac=0.15)
    inner_tr_idx = train_idx[inner_tr_rel]
    bundle = build_tier_matrices(feats, tier=tier, fit_idx=inner_tr_idx)
    res = _train_direct_mlp_with_inner_split(
        x_all=bundle.x_mol,
        y=y,
        train_idx=train_idx,
        inner_tr_rel=inner_tr_rel,
        inner_va_rel=inner_va_rel,
        test_idx=test_idx,
        seed=seed,
    )
    res["inner_tr_rel"] = inner_tr_rel
    res["inner_va_rel"] = inner_va_rel
    return res


def fit_similarity_gate_and_blend(
    y_val_true: np.ndarray,
    y_val_delta: np.ndarray,
    y_val_direct: np.ndarray,
    s_max_val: np.ndarray,
    y_test_delta: np.ndarray,
    y_test_direct: np.ndarray,
    s_max_test: np.ndarray,
) -> dict[str, Any]:
    """
    Fit 2-parameter OOD sigmoid gate alpha(s_max) = sigmoid(a * (s_max - b))
    strictly on inner_val predictions inside train, then blend test predictions.
    """
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

    def loss_fn(params: np.ndarray) -> float:
        a, b = float(params[0]), float(params[1])
        alpha = sigmoid(a * (s_max_val - b))
        y_blend = alpha * y_val_delta + (1.0 - alpha) * y_val_direct
        mse = float(np.mean((y_val_true - y_blend) ** 2))
        # Tiny L2 stabilization on slope `a` to avoid extreme step-function collapse
        return mse + 1e-4 * (a ** 2)

    res = optimize.minimize(
        loss_fn,
        x0=np.array([6.0, 0.55], dtype=np.float64),
        bounds=[(0.0, 25.0), (0.15, 0.90)],
        method="L-BFGS-B",
    )
    a_opt, b_opt = float(res.x[0]), float(res.x[1])
    alpha_val = sigmoid(a_opt * (s_max_val - b_opt))
    alpha_test = sigmoid(a_opt * (s_max_test - b_opt))

    y_pred_test = (alpha_test * y_test_delta + (1.0 - alpha_test) * y_test_direct).astype(np.float64)
    y_pred_val = (alpha_val * y_val_delta + (1.0 - alpha_val) * y_val_direct).astype(np.float64)

    return {
        "y_pred_test": y_pred_test,
        "y_pred_val": y_pred_val,
        "alpha_test": alpha_test.astype(np.float64),
        "gate_a": a_opt,
        "gate_b": b_opt,
        "alpha_test_mean": float(np.mean(alpha_test)),
    }

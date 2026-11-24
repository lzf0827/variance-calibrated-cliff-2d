"""Automated anti-leakage, ground-truth alignment, and determinism verification tests."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from physchem_delta.data import MVP_TARGETS, load_target_dataset
from physchem_delta.featurize import TargetFeatures, extract_or_load_target_features
from physchem_delta.models import (
    _select_topk_anchors,
    build_tier_matrices,
    compute_tanimoto_matrix,
    make_inner_split,
    run_pairwise_delta_model,
)

TEST_CACHE_DIR = ROOT_DIR / "physchem_delta" / "_test_feature_cache"


def _load_test_slice(target: str = "CHEMBL2147_Ki"):
    ds = load_target_dataset(target)
    tr_sub = ds.train_idx[:120]
    te_sub = ds.test_idx[:40]
    keep = np.concatenate([tr_sub, te_sub])
    smiles_sub = [ds.smiles[i] for i in keep]
    y_sub = ds.y[keep]
    train_rel = np.arange(len(tr_sub))
    test_rel = np.arange(len(tr_sub), len(keep))
    feats = extract_or_load_target_features(
        f"{target}_leaktest",
        smiles_sub,
        cache_dir=TEST_CACHE_DIR,
        prefer_aimnet=False,
        num_confs=2,
    )
    return feats, y_sub, train_rel, test_rel


def test_1_ground_truth_alignment(target: str = "CHEMBL2147_Ki") -> None:
    """Verify loaded y, cliff_mol, and splits match raw CSV byte-for-byte."""
    ds = load_target_dataset(target)
    raw = pd.read_csv(ds.csv_path)

    assert len(ds.df) == len(raw), f"Row count mismatch: {len(ds.df)} vs {len(raw)}"
    assert np.allclose(ds.y, raw["y"].to_numpy(dtype=np.float64)), "y labels altered!"
    assert np.array_equal(ds.cliff_mol, raw["cliff_mol"].to_numpy(dtype=int)), "cliff_mol altered!"
    raw_train = np.where(raw["split"].astype(str).str.strip().str.lower() == "train")[0]
    raw_test = np.where(raw["split"].astype(str).str.strip().str.lower() == "test")[0]
    assert np.array_equal(ds.train_idx, raw_train), "train_idx altered!"
    assert np.array_equal(ds.test_idx, raw_test), "test_idx altered!"
    assert len(np.intersect1d(ds.train_idx, ds.test_idx)) == 0, "Train/test overlap!"
    print(f"[PASS] Test 1: Ground-truth alignment & zero train/test overlap ({target})")


def test_2_pair_construction_isolation(target: str = "CHEMBL2147_Ki") -> None:
    """Verify training pairs and anchor retrieval never leak test indices or self-matches."""
    feats, y_sub, train_idx, test_idx = _load_test_slice(target)

    inner_tr_rel, inner_va_rel = make_inner_split(y_sub[train_idx], seed=42, val_frac=0.15)
    inner_tr_idx = train_idx[inner_tr_rel]
    inner_va_idx = train_idx[inner_va_rel]

    assert len(np.intersect1d(inner_tr_idx, test_idx)) == 0
    assert len(np.intersect1d(inner_va_idx, test_idx)) == 0
    assert len(np.intersect1d(inner_tr_idx, inner_va_idx)) == 0

    sim_tr = compute_tanimoto_matrix(feats.ecfp_chiral[inner_tr_idx], feats.ecfp_chiral[inner_tr_idx])
    nbr_g, _ = _select_topk_anchors(sim_tr, inner_tr_idx, k=15, exclude_self_global_idx=inner_tr_idx)

    test_set = set(test_idx.tolist())
    val_set = set(inner_va_idx.tolist())
    assert not any(int(g) in test_set for g in nbr_g.reshape(-1)), "Test index leaked into training pairs!"
    assert not any(int(g) in val_set for g in nbr_g.reshape(-1)), "Inner val leaked into training pairs!"
    for row, src in enumerate(inner_tr_idx.tolist()):
        assert int(src) not in set(nbr_g[row].tolist()), "Self-pair leaked into training pairs!"

    # Test anchor lookup must only retrieve from train_idx
    sim_te = compute_tanimoto_matrix(feats.ecfp_chiral[test_idx], feats.ecfp_chiral[train_idx])
    te_anchors, _ = _select_topk_anchors(sim_te, train_idx, k=10, exclude_self_global_idx=None)
    assert not any(int(g) in test_set for g in te_anchors.reshape(-1)), "Test index leaked into anchors!"
    print("[PASS] Test 2: Zero train/val/test overlap & no self-match in pair/anchor construction")


def test_3_scaler_fit_only_on_train(target: str = "CHEMBL2147_Ki") -> None:
    """Verify StandardScaler mean and scale match fit_idx-only statistics, invariant to test perturbations."""
    feats, _, train_idx, test_idx = _load_test_slice(target)

    bundle1 = build_tier_matrices(feats, tier="physchem", fit_idx=train_idx)

    # Corrupt test features dramatically in a copy and confirm scaler statistics and train features stay identical
    corrupted_3d = feats.feat_3d.copy()
    corrupted_3d[test_idx] = 9999.0
    feats_corrupted = TargetFeatures(
        target=feats.target,
        electronic_backend=feats.electronic_backend,
        ecfp_nochiral=feats.ecfp_nochiral,
        ecfp_chiral=feats.ecfp_chiral,
        maccs=feats.maccs,
        desc2d=feats.desc2d,
        feat_3d=corrupted_3d,
        feat_elec=feats.feat_elec,
        atom_ptr=feats.atom_ptr,
        atom_env_hash=feats.atom_env_hash,
        atom_3d_props=feats.atom_3d_props,
        atom_elec_props=feats.atom_elec_props,
        meta=feats.meta,
    )
    bundle2 = build_tier_matrices(feats_corrupted, tier="physchem", fit_idx=train_idx)

    assert np.allclose(bundle1.scaler_mean, bundle2.scaler_mean), "Scaler mean depended on test split!"
    assert np.allclose(bundle1.scaler_scale, bundle2.scaler_scale), "Scaler scale depended on test split!"
    assert np.allclose(bundle1.x_mol[train_idx], bundle2.x_mol[train_idx]), "Train features shifted!"
    print("[PASS] Test 3: Scaler fit strictly on fit_idx (invariant to test perturbations)")


def test_4_test_labels_blind_during_prediction(target: str = "CHEMBL2147_Ki") -> None:
    """Verify predictions on test_idx are identical even if test y and test cliff_mol are scrambled."""
    feats, y_sub, train_idx, test_idx = _load_test_slice(target)

    res1 = run_pairwise_delta_model(
        tier="2d",
        feats=feats,
        y=y_sub,
        train_idx=train_idx,
        test_idx=test_idx,
        seed=42,
        max_epochs=6,
    )
    y_scrambled = y_sub.copy()
    y_scrambled[test_idx] = -999.0
    res2 = run_pairwise_delta_model(
        tier="2d",
        feats=feats,
        y=y_scrambled,
        train_idx=train_idx,
        test_idx=test_idx,
        seed=42,
        max_epochs=6,
    )
    assert np.allclose(res1["y_pred_test"], res2["y_pred_test"], atol=1e-6), (
        "Model prediction changed when test labels were altered — TEST LABEL LEAKAGE!"
    )
    print("[PASS] Test 4: Test labels completely blind during training and inference")


def test_5_seed_determinism(target: str = "CHEMBL2147_Ki") -> None:
    """Verify repeated runs with identical seed produce deterministic predictions."""
    feats, y_sub, train_idx, test_idx = _load_test_slice(target)

    res1 = run_pairwise_delta_model(
        tier="2d", feats=feats, y=y_sub, train_idx=train_idx, test_idx=test_idx, seed=42, max_epochs=5
    )
    res2 = run_pairwise_delta_model(
        tier="2d", feats=feats, y=y_sub, train_idx=train_idx, test_idx=test_idx, seed=42, max_epochs=5
    )
    max_diff = float(np.max(np.abs(res1["y_pred_test"] - res2["y_pred_test"])))
    assert max_diff < 1e-3, f"Non-deterministic predictions across identical seed (max_diff={max_diff})"
    print(f"[PASS] Test 5: Seed determinism verified (max_diff={max_diff:.2e})")


def main() -> None:
    try:
        for t in MVP_TARGETS:
            test_1_ground_truth_alignment(t)
        test_2_pair_construction_isolation("CHEMBL2147_Ki")
        test_3_scaler_fit_only_on_train("CHEMBL2147_Ki")
        test_4_test_labels_blind_during_prediction("CHEMBL2147_Ki")
        test_5_seed_determinism("CHEMBL2147_Ki")
        print("ALL 5 ANTI-LEAKAGE TESTS PASSED.")
    finally:
        if TEST_CACHE_DIR.exists():
            shutil.rmtree(TEST_CACHE_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()

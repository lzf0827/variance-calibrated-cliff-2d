"""Download, cache, and validate official MoleculeACE benchmark datasets."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

MVP_TARGETS: tuple[str, ...] = (
    "CHEMBL204_Ki",   # Thrombin (protease)
    "CHEMBL2147_Ki",  # PIM1 (kinase)
    "CHEMBL237_Ki",   # Kappa opioid receptor (GPCR)
)

GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/molML/MoleculeACE/main/"
    "MoleculeACE/Data/benchmark_data"
)
JSDELIVR_RAW_BASE = (
    "https://cdn.jsdelivr.net/gh/molML/MoleculeACE@main/"
    "MoleculeACE/Data/benchmark_data"
)
GITHUB_API_DIR = (
    "https://api.github.com/repos/molML/MoleculeACE/contents/"
    "MoleculeACE/Data/benchmark_data"
)

REQUIRED_COLUMNS = ("smiles", "y", "cliff_mol", "split")


@dataclass(frozen=True)
class TargetDataset:
    target: str
    csv_path: Path
    df: pd.DataFrame
    train_idx: np.ndarray
    test_idx: np.ndarray
    smiles: list[str]
    y: np.ndarray
    cliff_mol: np.ndarray


def get_default_data_dir() -> Path:
    return Path(__file__).resolve().parent / "data_cache"


def download_target_csv(target: str, data_dir: Path | None = None, timeout: int = 30) -> Path:
    """Download official MoleculeACE CSV with retry and CDN mirror fallback if not already cached."""
    data_dir = data_dir or get_default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / f"{target}.csv"
    if out_path.exists() and out_path.stat().st_size > 100:
        return out_path

    urls = (
        f"{GITHUB_RAW_BASE}/{target}.csv",
        f"{JSDELIVR_RAW_BASE}/{target}.csv",
    )
    backoffs = (1, 2, 4, 8, 12, 16)
    last_err: Exception | None = None

    for attempt, wait_sec in enumerate(backoffs):
        url = urls[attempt % len(urls)]
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200 and len(resp.content) > 100:
                tmp_path = out_path.with_suffix(".csv.tmp")
                tmp_path.write_bytes(resp.content)
                tmp_path.replace(out_path)
                return out_path
            last_err = RuntimeError(f"HTTP {resp.status_code} (len={len(resp.content)}) from {url}")
        except requests.RequestException as exc:
            last_err = exc

        if attempt + 1 < len(backoffs):
            time.sleep(wait_sec)

    raise RuntimeError(f"Failed to download {target}.csv after {len(backoffs)} attempts: {last_err}")


def validate_moleculeace_df(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Validate schema and integrity of MoleculeACE dataframe without mutating labels/splits."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"[{target}] Missing required columns: {missing}. Found: {list(df.columns)}")

    df = df.copy()
    split_vals = df["split"].astype(str).str.strip().str.lower()
    if not set(split_vals.unique()).issuperset({"train", "test"}):
        raise ValueError(f"[{target}] Unexpected split values: {split_vals.unique().tolist()}")
    df["split"] = split_vals

    if df["smiles"].isna().any():
        raise ValueError(f"[{target}] Found NaN SMILES entries.")
    if not np.all(np.isfinite(df["y"].to_numpy(dtype=np.float64))):
        raise ValueError(f"[{target}] Found non-finite target y values.")

    cliff_vals = df["cliff_mol"].to_numpy(dtype=int)
    if not set(np.unique(cliff_vals)).issubset({0, 1}):
        raise ValueError(f"[{target}] Unexpected cliff_mol values: {np.unique(cliff_vals)}")

    n_test_cliff = int(np.sum((df["split"] == "test") & (df["cliff_mol"] == 1)))
    if n_test_cliff == 0:
        raise ValueError(f"[{target}] No cliff_mol == 1 samples found in test split.")
    return df


def load_target_dataset(target: str, data_dir: Path | None = None) -> TargetDataset:
    """Load and validate a MoleculeACE target dataset."""
    csv_path = download_target_csv(target, data_dir=data_dir)
    raw_df = pd.read_csv(csv_path)
    df = validate_moleculeace_df(raw_df, target)

    train_idx = np.where(df["split"].to_numpy() == "train")[0]
    test_idx = np.where(df["split"].to_numpy() == "test")[0]
    if len(np.intersect1d(train_idx, test_idx)) != 0:
        raise RuntimeError(f"[{target}] Train/test index overlap detected!")

    return TargetDataset(
        target=target,
        csv_path=csv_path,
        df=df,
        train_idx=train_idx,
        test_idx=test_idx,
        smiles=df["smiles"].astype(str).tolist(),
        y=df["y"].to_numpy(dtype=np.float64),
        cliff_mol=df["cliff_mol"].to_numpy(dtype=int),
    )


def ensure_targets_downloaded(targets: Iterable[str] = MVP_TARGETS, data_dir: Path | None = None) -> dict[str, dict[str, int]]:
    """Download all targets and return dataset summary stats."""
    summary: dict[str, dict[str, int]] = {}
    for target in targets:
        ds = load_target_dataset(target, data_dir=data_dir)
        n_train_cliff = int(np.sum(ds.cliff_mol[ds.train_idx] == 1))
        n_test_cliff = int(np.sum(ds.cliff_mol[ds.test_idx] == 1))
        summary[target] = {
            "n_total": len(ds.df),
            "n_train": int(ds.train_idx.size),
            "n_test": int(ds.test_idx.size),
            "n_train_cliff": n_train_cliff,
            "n_test_cliff": n_test_cliff,
        }
    return summary

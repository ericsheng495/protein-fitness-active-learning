from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, Ridge, RidgeCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.preprocessing import StandardScaler


AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {a: i for i, a in enumerate(AA_LIST)}
KD_HYDRO = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}


@dataclass
class Config:
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Optional[Path] = None
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    holdout_index_path: Path = field(default_factory=lambda: Path("cache/holdout_indices.npy"))
    query_results_glob: str = "query_round_*_results.csv"
    active_plm: str = "esm2_8m"
    esm2_checkpoint: str = "esm2_t6_8M_UR50D"
    esm1v_model_idx: int = 1
    use_blosum: bool = True
    use_zero_shot: bool = True
    structure_csv: Optional[Path] = None
    n_folds: int = 5
    holdout_frac: float = 0.12
    random_state: int = 42
    use_smote: bool = True
    smote_tail_quantile: float = 0.9
    embedding_noise_std: float = 0.02
    plm_pca_components: Optional[int] = 128
    use_xgb: bool = True
    use_rf: bool = True
    use_ridge: bool = True
    use_elastic_net: bool = True
    use_knn: bool = True
    use_mlp: bool = True
    xgb_n_estimators: int = 400
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_reg_lambda: float = 1.0
    xgb_n_jobs: int = 1
    rf_n_estimators: int = 300
    rf_max_depth: Optional[int] = None
    rf_min_samples_leaf: int = 2
    rf_n_jobs: int = 1
    ridge_alphas: Optional[Tuple[float, ...]] = None
    elastic_net_alpha_min: float = 1e-4
    elastic_net_alpha_max: float = 1.0
    elastic_net_n_alphas: int = 20
    elastic_net_l1_ratios: Tuple[float, ...] = (0.1, 0.5, 0.9)
    elastic_net_cv: int = 3
    elastic_net_max_iter: int = 8000
    knn_neighbors: int = 8
    knn_weights: str = "distance"
    knn_leaf_size: int = 30
    knn_p: int = 2
    mlp_hidden_dim: int = 256
    mlp_num_hidden_layers: int = 2
    mlp_dropout: float = 0.2
    mlp_lr: float = 1e-3
    mlp_weight_decay: float = 1e-4
    mlp_epochs: int = 200
    mlp_patience: int = 30
    mlp_batch_size: int = 64
    mlp_optimizer: str = "adamw"
    meta_learner: str = "spearman_ridge"
    meta_log_alpha_min: float = -4.0
    meta_log_alpha_max: float = 4.0
    meta_n_alphas: int = 24
    repeated_holdout_n: int = 5
    use_optuna: bool = False
    optuna_n_trials_xgb: int = 20
    optuna_n_trials_mlp: int = 15
    optuna_xgb_n_jobs: int = 1
    device: Optional[str] = None
    clip_predictions: bool = True
    query_budget: int = 100
    query_destination_min: Optional[int] = None
    query_destination_max: Optional[int] = None
    hybrid_alpha: float = 0.5
    ucb_beta: float = 1.0
    diversity_lambda: float = 0.3
    round2_fractions: Tuple[float, float, float, float] = (0.28, 0.28, 0.24, 0.20)
    round3_neighbor_fraction: float = 0.05
    round3_neighbor_k: int = 32
    round3_budgets: Tuple[int, int, int, int] = (35, 28, 22, 15)
    round3_hybrid_alpha: float = 0.66
    round3_ucb_beta: float = 0.62
    round3_diversity_lambda: float = 1.22
    round3_destination_min: Optional[int] = 2
    round3_destination_max: Optional[int] = 7
    diagnostic_weight_min_n: int = 3
    diagnostic_weight_clip: Tuple[float, float] = (0.5, 2.0)
    diagnostic_weight_power: float = 1.0
    query_txt_name: str = "query.txt"
    plm_batch_size: int = 8
    prott5_batch_size: int = 4
    xgb_overrides: Dict[str, Any] = field(default_factory=dict)

    def project_root(self) -> Path:
        return Path(self.data_dir).resolve().parent

    def output_path(self) -> Path:
        path = Path(self.output_dir) if self.output_dir is not None else self.project_root() / "results" / "generated"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def query_txt_path(self) -> Path:
        return self.output_path() / self.query_txt_name

    def effective_device(self) -> str:
        raw = None if self.device is None else str(self.device).strip().lower()
        if raw and raw not in {"auto", "gpu"}:
            return str(self.device)
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"


@dataclass
class FeatureLayout:
    traditional_cols: Tuple[int, int]
    plm_cols: Tuple[int, int]
    trailing_cols: Tuple[int, int]

    def with_plm_width(self, width: int) -> "FeatureLayout":
        t0, t1 = self.traditional_cols
        r0, r1 = self.trailing_cols
        tw = r1 - r0
        p0 = t1
        return FeatureLayout((t0, t1), (p0, p0 + width), (p0 + width, p0 + width + tw))

    def traditional(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.traditional_cols[0] : self.traditional_cols[1]]

    def plm(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.plm_cols[0] : self.plm_cols[1]]

    def trailing(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.trailing_cols[0] : self.trailing_cols[1]]

    def no_plm(self, X: np.ndarray) -> np.ndarray:
        t = self.traditional(X)
        r = self.trailing(X)
        return np.hstack([t, r]) if r.shape[1] else t

    def distance_repr(self, X: np.ndarray) -> np.ndarray:
        p = self.plm(X)
        return p if p.shape[1] else X

    def base_view(self, name: str, X: np.ndarray) -> np.ndarray:
        if name == "xgb":
            return self.traditional(X)
        if name == "elastic_net":
            return self.no_plm(X)
        return X


@dataclass
class StackModel:
    base_names: List[str]
    base_models: Dict[str, Any]
    base_scalers: Dict[str, StandardScaler]
    meta_model: Any
    meta_scaler: StandardScaler
    layout_in: FeatureLayout
    layout_work: FeatureLayout
    plm_pca: Optional[PCA]
    base_layouts: Optional[Dict[str, FeatureLayout]] = None
    base_pcas: Optional[Dict[str, Optional[PCA]]] = None


@dataclass
class RepeatedHoldoutResult:
    stack_spearmans: List[float]
    base_spearmans: List[Dict[str, float]]
    mean_stack: float
    std_stack: float
    mean_bases: Dict[str, float]
    std_bases: Dict[str, float]


@dataclass
class PipelineResults:
    cfg: Config
    df_train: pd.DataFrame
    df_test: pd.DataFrame
    wt_sequence: str
    original_train_mutants: Set[str]
    X_train: np.ndarray
    X_test: np.ndarray
    feature_layout: FeatureLayout
    holdout_spearman: float
    holdout_base_spearmans: Dict[str, float]
    repeated_holdout: Optional[RepeatedHoldoutResult]
    stack: StackModel
    y_train_pred_oof: np.ndarray
    y_test_pred: np.ndarray
    predictions: pd.DataFrame


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "manual_seed"):
        try:
            mps.manual_seed(seed)
        except Exception:
            pass


def spearman_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if len(yt) < 2 or len(yp) < 2 or len(yt) != len(yp):
        return 0.0
    if float(np.nanstd(yt)) < 1e-12 or float(np.nanstd(yp)) < 1e-12:
        return 0.0
    rho, _ = spearmanr(yt, yp)
    return 0.0 if np.isnan(rho) else float(rho)


def parse_mutant(mutant: str) -> Tuple[str, int, str]:
    m = re.match(r"^([A-Z])(\d+)([A-Z])$", str(mutant).strip())
    if m is None:
        raise ValueError(f"invalid mutant: {mutant}")
    return m.group(1), int(m.group(2)), m.group(3)


def get_mutated_sequence(mutant: str, wt_sequence: str) -> str:
    wt, pos, mt = parse_mutant(mutant)
    if pos < 0 or pos >= len(wt_sequence):
        raise ValueError(f"position out of range: {mutant}")
    if wt_sequence[pos] != wt:
        raise ValueError(f"wild-type mismatch for {mutant}")
    return wt_sequence[:pos] + mt + wt_sequence[pos + 1 :]


def load_wild_type_sequence(path: Path) -> str:
    lines = path.read_text().splitlines()
    return "".join(line.strip() for line in lines if line.strip() and not line.startswith(">"))


def load_train_test(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, str, Set[str]]:
    wt_sequence = load_wild_type_sequence(Path(cfg.data_dir) / "sequence.fasta")
    df_train = pd.read_csv(Path(cfg.data_dir) / "train.csv").copy()
    df_test = pd.read_csv(Path(cfg.data_dir) / "test.csv").copy()
    original_train_mutants = set(df_train["mutant"].astype(str))
    df_train["sequence"] = df_train["mutant"].map(lambda m: get_mutated_sequence(m, wt_sequence))
    df_test["sequence"] = df_test["mutant"].map(lambda m: get_mutated_sequence(m, wt_sequence))
    return df_train, df_test, wt_sequence, original_train_mutants


def merge_query_results(df_train: pd.DataFrame, wt_sequence: str, cfg: Config) -> pd.DataFrame:
    out = df_train.copy()
    for path in sorted(Path(cfg.data_dir).glob(cfg.query_results_glob)):
        q = pd.read_csv(path)
        if not {"mutant", "DMS_score"}.issubset(q.columns):
            continue
        q = q[["mutant", "DMS_score"]].copy()
        q["sequence"] = q["mutant"].map(lambda m: get_mutated_sequence(m, wt_sequence))
        out = pd.concat([out, q], ignore_index=True)
    return out.drop_duplicates(subset=["mutant"], keep="last").reset_index(drop=True)


def read_query_txt_mutants(path: Path) -> Set[str]:
    path = Path(path)
    if not path.is_file():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def collect_prior_query_mutants(cfg: Config) -> Set[str]:
    data_dir = Path(cfg.data_dir)
    out: Set[str] = set()
    for path in sorted(data_dir.glob("query_round_*.txt")):
        out |= read_query_txt_mutants(path)
    return out


def _hash_mutants(mutants: Sequence[str]) -> str:
    h = hashlib.sha256()
    for mutant in mutants:
        h.update(str(mutant).encode())
    return h.hexdigest()[:16]


def make_saved_holdout_indices(n_samples: int, mutants: Sequence[str], cfg: Config) -> np.ndarray:
    path = Path(cfg.holdout_index_path)
    meta_path = path.with_suffix(".meta.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    n_hold = max(1, int(round(n_samples * cfg.holdout_frac)))
    if path.exists() and meta_path.exists():
        try:
            idx = np.load(path)
            meta = json.loads(meta_path.read_text())
            if (
                len(idx) == n_hold
                and int(meta.get("n_samples", -1)) == int(n_samples)
                and str(meta.get("mutants_hash", "")) == _hash_mutants(mutants)
            ):
                return np.sort(idx.astype(int))
        except Exception:
            pass
    rng = np.random.default_rng(cfg.random_state)
    idx = np.sort(rng.choice(n_samples, size=n_hold, replace=False))
    np.save(path, idx)
    meta_path.write_text(json.dumps({"n_samples": int(n_samples), "mutants_hash": _hash_mutants(mutants)}))
    return idx


def lpo_splits(df_train: pd.DataFrame, n_folds: int, random_state: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    groups = df_train["mutant"].map(lambda m: parse_mutant(m)[1]).to_numpy()
    unique_groups = np.unique(groups)
    n_splits = min(int(n_folds), len(unique_groups))
    if n_splits >= 2:
        gkf = GroupKFold(n_splits=n_splits)
        y_dummy = np.zeros(len(df_train), dtype=np.float32)
        for tr, va in gkf.split(np.arange(len(df_train)), y_dummy, groups):
            yield tr, va
        return
    kf = KFold(n_splits=min(2, len(df_train)), shuffle=True, random_state=random_state)
    for tr, va in kf.split(np.arange(len(df_train))):
        yield tr, va


def _blosum62_matrix() -> Dict[Tuple[str, str], float]:
    try:
        from Bio.Align import substitution_matrices

        mat = substitution_matrices.load("BLOSUM62")
        out: Dict[Tuple[str, str], float] = {}
        for a in AA_LIST:
            for b in AA_LIST:
                try:
                    out[(a, b)] = float(mat[a, b])
                except Exception:
                    out[(a, b)] = 0.0
        return out
    except Exception:
        return {(a, b): (4.0 if a == b else -1.0) for a in AA_LIST for b in AA_LIST}


_BLOSUM62: Optional[Dict[Tuple[str, str], float]] = None


def blosum_scores(mutants: Sequence[str]) -> np.ndarray:
    global _BLOSUM62
    if _BLOSUM62 is None:
        _BLOSUM62 = _blosum62_matrix()
    out = np.zeros((len(mutants), 1), dtype=np.float32)
    for i, mutant in enumerate(mutants):
        wt, _, mt = parse_mutant(mutant)
        out[i, 0] = float(_BLOSUM62.get((wt, mt), _BLOSUM62.get((mt, wt), 0.0))) / 11.0
    return out


def compact_mutation_features(mutants: Sequence[str], seq_len: int) -> np.ndarray:
    X = np.zeros((len(mutants), 41), dtype=np.float32)
    for i, mutant in enumerate(mutants):
        wt, pos, mt = parse_mutant(mutant)
        X[i, AA_TO_IDX[mt]] = 1.0
        X[i, 20] = pos / max(seq_len - 1, 1)
        X[i, 21 + AA_TO_IDX[wt]] = 1.0
    return X


def mutation_delta_features(mutants: Sequence[str]) -> np.ndarray:
    charged = set("DEKRH")
    X = np.zeros((len(mutants), 4), dtype=np.float32)
    for i, mutant in enumerate(mutants):
        wt, _, mt = parse_mutant(mutant)
        X[i, 0] = KD_HYDRO.get(mt, 0.0) - KD_HYDRO.get(wt, 0.0)
        X[i, 1] = float(mt in charged) - float(wt in charged)
        X[i, 2] = KD_HYDRO.get(mt, 0.0)
        X[i, 3] = KD_HYDRO.get(wt, 0.0)
    return X


def protparam_vector(sequence: str) -> np.ndarray:
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis

        p = ProteinAnalysis(sequence)
        return np.array(
            [
                p.molecular_weight() / 1e5,
                p.isoelectric_point() / 14.0,
                p.aromaticity(),
                p.instability_index() / 100.0,
                (p.gravy() + 4.5) / 9.0,
            ],
            dtype=np.float32,
        )
    except Exception:
        return np.zeros(5, dtype=np.float32)


def global_sequence_features(sequences: Sequence[str]) -> np.ndarray:
    return np.stack([protparam_vector(seq) for seq in sequences], axis=0)


def _normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def load_structural_features(mutants: Sequence[str], cfg: Config) -> np.ndarray:
    if cfg.structure_csv is None:
        return np.zeros((len(mutants), 2), dtype=np.float32)
    path = Path(cfg.structure_csv)
    if not path.is_file():
        raise ValueError(f"structural feature file not found: {path}")
    df = pd.read_csv(path)
    if "mutant" not in df.columns:
        raise ValueError("structural feature CSV must contain a 'mutant' column")
    numeric_cols = [c for c in df.columns if c != "mutant" and pd.api.types.is_numeric_dtype(df[c])]
    sasa_cols = [col for col in numeric_cols if "sasa" in _normalize_column_name(col)]
    bfactor_cols = [col for col in numeric_cols if "bfactor" in _normalize_column_name(col)]
    if len(sasa_cols) != 1:
        raise ValueError("structural feature CSV must contain exactly one SASA column")
    if len(bfactor_cols) != 1:
        raise ValueError("structural feature CSV must contain exactly one B-factor column")
    allowed = {sasa_cols[0], bfactor_cols[0]}
    extra_numeric = [col for col in numeric_cols if col not in allowed]
    if extra_numeric:
        raise ValueError(f"structural feature CSV contains unsupported numeric columns: {extra_numeric}")
    feature_cols = [sasa_cols[0], bfactor_cols[0]]
    value_map = {
        str(row["mutant"]): row[feature_cols].to_numpy(dtype=np.float32)
        for _, row in df[["mutant", *feature_cols]].iterrows()
    }
    width = len(feature_cols)
    return np.stack([value_map.get(str(m), np.zeros(width, dtype=np.float32)) for m in mutants], axis=0)


def _cache_key(cfg: Config, split_label: str, mutants: Sequence[str]) -> str:
    payload = {
        "plm": cfg.active_plm,
        "esm2": cfg.esm2_checkpoint,
        "n": len(mutants),
        "mutants_hash": _hash_mutants(mutants),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def _plm_batch_starts(n: int, batch_size: int) -> Iterable[int]:
    for i in range(0, n, batch_size):
        yield i


def ensure_esm_weights(model_name: str) -> None:
    hub = Path(torch.hub.get_dir()) / "checkpoints"
    hub.mkdir(parents=True, exist_ok=True)
    main_name = f"{model_name}.pt"
    main_dest = hub / main_name
    if not main_dest.is_file():
        torch.hub.download_url_to_file(
            f"https://dl.fbaipublicfiles.com/fair-esm/models/{main_name}",
            str(main_dest),
            progress=False,
        )
    if "esm1v" not in model_name:
        reg_name = f"{model_name}-contact-regression.pt"
        reg_dest = hub / reg_name
        if not reg_dest.is_file():
            torch.hub.download_url_to_file(
                f"https://dl.fbaipublicfiles.com/fair-esm/regression/{reg_name}",
                str(reg_dest),
                progress=False,
            )


def load_esm_model(cfg: Config) -> Tuple[Any, Any, int]:
    import esm

    if cfg.active_plm == "esm1v":
        names = [
            "esm1v_t33_650M_UR90S_1",
            "esm1v_t33_650M_UR90S_2",
            "esm1v_t33_650M_UR90S_3",
            "esm1v_t33_650M_UR90S_4",
            "esm1v_t33_650M_UR90S_5",
        ]
        loaders = [
            esm.pretrained.esm1v_t33_650M_UR90S_1,
            esm.pretrained.esm1v_t33_650M_UR90S_2,
            esm.pretrained.esm1v_t33_650M_UR90S_3,
            esm.pretrained.esm1v_t33_650M_UR90S_4,
            esm.pretrained.esm1v_t33_650M_UR90S_5,
        ]
        idx = max(0, min(int(cfg.esm1v_model_idx) - 1, 4))
        ensure_esm_weights(names[idx])
        model, alphabet = loaders[idx]()
        return model, alphabet, 33
    name = cfg.esm2_checkpoint
    if cfg.active_plm == "esm2_35m":
        name = "esm2_t12_35M_UR50D"
    elif cfg.active_plm == "esm2_150m":
        name = "esm2_t30_150M_UR50D"
    elif cfg.active_plm == "esm2_650m":
        name = "esm2_t33_650M_UR50D"
    else:
        name = cfg.esm2_checkpoint or "esm2_t6_8M_UR50D"
    ensure_esm_weights(name)
    fn = getattr(esm.pretrained, name)
    model, alphabet = fn()
    return model, alphabet, int(model.num_layers)


def compute_esm_embeddings(sequences: Sequence[str], cfg: Config) -> np.ndarray:
    model, alphabet, layer = load_esm_model(cfg)
    device = torch.device(cfg.effective_device())
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    out: List[np.ndarray] = []
    with torch.no_grad():
        for i in _plm_batch_starts(len(sequences), int(cfg.plm_batch_size)):
            batch = list(sequences[i : i + int(cfg.plm_batch_size)])
            _, _, tokens = batch_converter([("p", seq) for seq in batch])
            tokens = tokens.to(device)
            res = model(tokens, repr_layers=[layer])
            reps = res["representations"][layer][:, 1:-1, :].mean(dim=1)
            out.append(reps.float().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def compute_prott5_embeddings(sequences: Sequence[str], cfg: Config) -> np.ndarray:
    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False)
    t5 = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    device = torch.device(cfg.effective_device())
    model = cast(nn.Module, t5).to(device).eval()
    out: List[np.ndarray] = []
    with torch.no_grad():
        for i in _plm_batch_starts(len(sequences), int(cfg.prott5_batch_size)):
            batch = list(sequences[i : i + int(cfg.prott5_batch_size)])
            toks = tokenizer(batch, add_special_tokens=True, padding=True, return_tensors="pt")
            toks = {k: v.to(device) for k, v in toks.items()}
            reps = model(**toks).last_hidden_state
            mask = toks["attention_mask"].unsqueeze(-1)
            pooled = (reps * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(pooled.float().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def compute_plm_embeddings(mutants: Sequence[str], sequences: Sequence[str], split_label: str, cfg: Config) -> np.ndarray:
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(cfg.cache_dir) / f"embeddings_{split_label}_{_cache_key(cfg, split_label, mutants)}.npy"
    if cache_path.is_file():
        return np.load(cache_path)
    if cfg.active_plm == "prott5":
        emb = compute_prott5_embeddings(sequences, cfg)
    else:
        emb = compute_esm_embeddings(sequences, cfg)
    np.save(cache_path, emb)
    return emb


def compute_zero_shot_scores(mutants: Sequence[str], wt_sequence: str, cfg: Config) -> np.ndarray:
    if not cfg.use_zero_shot:
        return np.zeros((len(mutants), 1), dtype=np.float32)
    try:
        import esm
    except Exception:
        return np.zeros((len(mutants), 1), dtype=np.float32)
    ensure_esm_weights("esm2_t6_8M_UR50D")
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    device = torch.device(cfg.effective_device())
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    aa_to_idx = {aa: alphabet.get_idx(aa) for aa in AA_LIST if aa in alphabet.all_toks}
    mask_idx = getattr(alphabet, "mask_idx", None)
    if mask_idx is None:
        try:
            mask_idx = alphabet.get_idx("<mask>")
        except Exception:
            return np.zeros((len(mutants), 1), dtype=np.float32)
    by_pos: Dict[int, List[Tuple[int, str, str]]] = {}
    for i, mutant in enumerate(mutants):
        wt, pos, mt = parse_mutant(mutant)
        by_pos.setdefault(pos, []).append((i, wt, mt))
    scores = np.zeros(len(mutants), dtype=np.float32)
    _, _, wt_tokens = batch_converter([("wt", wt_sequence)])
    wt_tokens = wt_tokens.to(device)
    positions = sorted(by_pos)
    with torch.no_grad():
        for start in _plm_batch_starts(len(positions), int(cfg.plm_batch_size)):
            batch_positions = positions[start : start + int(cfg.plm_batch_size)]
            tokens = wt_tokens.repeat(len(batch_positions), 1)
            for row, pos in enumerate(batch_positions):
                tokens[row, pos + 1] = int(mask_idx)
            logits = model(tokens)["logits"]
            log_probs = torch.log_softmax(logits, dim=-1)
            for row, pos in enumerate(batch_positions):
                pos_log_probs = log_probs[row, pos + 1]
                for idx, wt, mt in by_pos[pos]:
                    mt_lp = float(pos_log_probs[aa_to_idx[mt]].item()) if mt in aa_to_idx else -20.0
                    wt_lp = float(pos_log_probs[aa_to_idx[wt]].item()) if wt in aa_to_idx else -20.0
                    scores[idx] = mt_lp - wt_lp
    return scores.reshape(-1, 1)


def build_feature_matrix(df: pd.DataFrame, wt_sequence: str, split_label: str, cfg: Config) -> Tuple[np.ndarray, FeatureLayout]:
    mutants = df["mutant"].astype(str).tolist()
    sequences = df["sequence"].astype(str).tolist()
    parts: List[np.ndarray] = []
    parts.append(compact_mutation_features(mutants, len(wt_sequence)))
    parts.append(mutation_delta_features(mutants))
    parts.append(global_sequence_features(sequences))
    if cfg.use_blosum:
        parts.append(blosum_scores(mutants))
    traditional_end = sum(part.shape[1] for part in parts)
    parts.append(compute_plm_embeddings(mutants, sequences, split_label, cfg))
    plm_end = sum(part.shape[1] for part in parts)
    parts.append(load_structural_features(mutants, cfg))
    if cfg.use_zero_shot:
        parts.append(compute_zero_shot_scores(mutants, wt_sequence, cfg))
    X = np.hstack(parts).astype(np.float32)
    layout = FeatureLayout((0, traditional_end), (traditional_end, plm_end), (plm_end, X.shape[1]))
    return X, layout


def apply_plm_pca_fold(
    X_train: np.ndarray,
    X_val: np.ndarray,
    layout: FeatureLayout,
    n_components: Optional[int],
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, FeatureLayout, Optional[PCA]]:
    if not n_components or int(n_components) < 1:
        return X_train, X_val, layout, None
    p0, p1 = layout.plm_cols
    if p1 <= p0:
        return X_train, X_val, layout, None
    width = p1 - p0
    k = min(int(n_components), width, max(1, X_train.shape[0] - 1))
    if k < 1:
        return X_train, X_val, layout, None
    pca = PCA(n_components=k, random_state=random_state)
    tr_plm = pca.fit_transform(X_train[:, p0:p1]).astype(np.float32)
    va_plm = pca.transform(X_val[:, p0:p1]).astype(np.float32)
    X_train_new = np.hstack([X_train[:, :p0], tr_plm, X_train[:, p1:]])
    X_val_new = np.hstack([X_val[:, :p0], va_plm, X_val[:, p1:]])
    return X_train_new, X_val_new, layout.with_plm_width(tr_plm.shape[1]), pca


def fit_dev_plm_pca(X: np.ndarray, layout: FeatureLayout, cfg: Config) -> Tuple[np.ndarray, FeatureLayout, Optional[PCA]]:
    if not cfg.plm_pca_components or int(cfg.plm_pca_components) < 1:
        return X, layout, None
    p0, p1 = layout.plm_cols
    if p1 <= p0:
        return X, layout, None
    k = min(int(cfg.plm_pca_components), p1 - p0, max(1, X.shape[0] - 1))
    pca = PCA(n_components=k, random_state=cfg.random_state)
    emb = pca.fit_transform(X[:, p0:p1]).astype(np.float32)
    X_new = np.hstack([X[:, :p0], emb, X[:, p1:]])
    return X_new, layout.with_plm_width(emb.shape[1]), pca


def transform_plm_pca(X: np.ndarray, layout: FeatureLayout, pca: Optional[PCA]) -> Tuple[np.ndarray, FeatureLayout]:
    if pca is None:
        return X, layout
    p0, p1 = layout.plm_cols
    emb = pca.transform(X[:, p0:p1]).astype(np.float32)
    X_new = np.hstack([X[:, :p0], emb, X[:, p1:]])
    return X_new, layout.with_plm_width(emb.shape[1])


def smote_tail_oversample(X: np.ndarray, y: np.ndarray, cfg: Config, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    if not cfg.use_smote:
        return X, y
    threshold = np.quantile(y, cfg.smote_tail_quantile)
    tail_mask = y >= threshold
    if int(tail_mask.sum()) < 3:
        return X, y
    try:
        from imblearn.over_sampling import SMOTER

        smoter = SMOTER(random_state=random_state, k_neighbors=min(5, int(tail_mask.sum()) - 1))
        X_new, y_new = smoter.fit_resample(X, y)
        return X_new.astype(np.float32), y_new.astype(np.float32)
    except Exception:
        rng = np.random.default_rng(random_state)
        X_tail = X[tail_mask]
        y_tail = y[tail_mask]
        n_add = min(len(X), max(50, len(X_tail) * 2))
        pick = rng.choice(len(X_tail), size=n_add, replace=True)
        noise = rng.normal(0.0, 0.01, size=(n_add, X.shape[1])).astype(np.float32)
        return np.vstack([X, X_tail[pick] + noise]), np.concatenate([y, y_tail[pick]])


def add_embedding_noise(X: np.ndarray, layout: FeatureLayout, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    if float(cfg.embedding_noise_std) <= 0:
        return X
    p0, p1 = layout.plm_cols
    if p1 <= p0:
        return X
    X_new = X.astype(np.float32, copy=True)
    X_new[:, p0:p1] += rng.normal(0.0, float(cfg.embedding_noise_std), size=(X.shape[0], p1 - p0)).astype(np.float32)
    return X_new


def _sklearn_booster_only() -> bool:
    val = os.environ.get("HACKATHON_FORCE_SKLEARN_XGB", "").strip().lower()
    return val in {"1", "true", "yes"} or sys.version_info >= (3, 14)


def make_xgb(cfg: Config, **overrides: Any) -> Any:
    params = {
        "n_estimators": int(cfg.xgb_n_estimators),
        "max_depth": int(cfg.xgb_max_depth),
        "learning_rate": float(cfg.xgb_learning_rate),
        "subsample": float(cfg.xgb_subsample),
        "colsample_bytree": float(cfg.xgb_colsample_bytree),
        "reg_lambda": float(cfg.xgb_reg_lambda),
        "random_state": int(cfg.random_state),
        "n_jobs": int(cfg.xgb_n_jobs),
        "verbosity": 0,
    }
    params.update(cfg.xgb_overrides)
    params.update(overrides)
    if _sklearn_booster_only():
        return HistGradientBoostingRegressor(
            max_iter=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            l2_regularization=float(params["reg_lambda"]),
            max_features=float(params["colsample_bytree"]),
            random_state=int(params["random_state"]),
        )
    try:
        from xgboost import XGBRegressor

        return XGBRegressor(**params)
    except Exception:
        return HistGradientBoostingRegressor(
            max_iter=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            l2_regularization=float(params["reg_lambda"]),
            max_features=float(params["colsample_bytree"]),
            random_state=int(params["random_state"]),
        )


def make_rf(cfg: Config) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=int(cfg.rf_n_estimators),
        max_depth=cfg.rf_max_depth,
        min_samples_leaf=int(cfg.rf_min_samples_leaf),
        random_state=int(cfg.random_state),
        n_jobs=int(cfg.rf_n_jobs),
    )


def make_ridge(cfg: Config) -> RidgeCV:
    if cfg.ridge_alphas:
        alphas = np.asarray(cfg.ridge_alphas, dtype=np.float64)
    else:
        alphas = np.logspace(-4, 3, 20)
    return RidgeCV(alphas=alphas)


def make_elastic_net(cfg: Config, n_train: int) -> ElasticNetCV:
    inner = max(2, min(int(cfg.elastic_net_cv), max(2, n_train // 10)))
    alphas = np.logspace(np.log10(cfg.elastic_net_alpha_min), np.log10(cfg.elastic_net_alpha_max), int(cfg.elastic_net_n_alphas))
    return ElasticNetCV(
        alphas=alphas,
        l1_ratio=tuple(float(x) for x in cfg.elastic_net_l1_ratios),
        cv=inner,
        random_state=int(cfg.random_state),
        max_iter=int(cfg.elastic_net_max_iter),
        n_jobs=-1,
    )


def make_knn(cfg: Config, n_train: int) -> KNeighborsRegressor:
    k = max(1, min(int(cfg.knn_neighbors), max(1, n_train - 1)))
    return KNeighborsRegressor(
        n_neighbors=k,
        weights=str(cfg.knn_weights),
        algorithm="auto",
        leaf_size=int(cfg.knn_leaf_size),
        p=int(cfg.knn_p),
        n_jobs=-1,
    )


class RegressionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_hidden_layers: int, dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        width = int(hidden_dim)
        d = int(in_dim)
        for _ in range(int(num_hidden_layers)):
            layers.extend([nn.Linear(d, width), nn.ReLU(), nn.Dropout(float(dropout))])
            d = width
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _make_optimizer(params: Any, cfg: Config) -> torch.optim.Optimizer:
    name = str(cfg.mlp_optimizer).lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=float(cfg.mlp_lr), weight_decay=float(cfg.mlp_weight_decay))
    if name == "sgd":
        return torch.optim.SGD(params, lr=float(cfg.mlp_lr), weight_decay=float(cfg.mlp_weight_decay), momentum=0.9, nesterov=True)
    return torch.optim.AdamW(params, lr=float(cfg.mlp_lr), weight_decay=float(cfg.mlp_weight_decay))


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    cfg: Config,
    hidden_dim: Optional[int] = None,
    dropout: Optional[float] = None,
    fixed_epochs: Optional[int] = None,
) -> Tuple[RegressionMLP, StandardScaler, int]:
    seed_everything(cfg.random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    device = torch.device(cfg.effective_device())
    model = RegressionMLP(
        X_train_scaled.shape[1],
        int(hidden_dim if hidden_dim is not None else cfg.mlp_hidden_dim),
        int(cfg.mlp_num_hidden_layers),
        float(dropout if dropout is not None else cfg.mlp_dropout),
    ).to(device)
    optimizer = _make_optimizer(model.parameters(), cfg)
    X_tr_t = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    batch_size = max(8, min(int(cfg.mlp_batch_size), len(X_tr_t)))
    if fixed_epochs is not None:
        epochs = max(1, int(fixed_epochs))
        for _ in range(epochs):
            model.train()
            perm = torch.randperm(len(X_tr_t), device=device)
            for i in range(0, len(perm), batch_size):
                idx = perm[i : i + batch_size]
                optimizer.zero_grad()
                pred = model(X_tr_t[idx])
                loss = torch.mean((pred - y_tr_t[idx]) ** 2)
                loss.backward()
                optimizer.step()
        return model, scaler, epochs
    if X_val is None or y_val is None:
        raise ValueError("X_val and y_val are required when fixed_epochs is not set")
    X_val_scaled = scaler.transform(X_val)
    X_va_t = torch.tensor(X_val_scaled, dtype=torch.float32, device=device)
    y_va_t = torch.tensor(y_val, dtype=torch.float32, device=device)
    best_state = None
    best_loss = float("inf")
    best_epoch = 1
    bad = 0
    for epoch in range(int(cfg.mlp_epochs)):
        model.train()
        perm = torch.randperm(len(X_tr_t), device=device)
        for i in range(0, len(perm), batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            pred = model(X_tr_t[idx])
            loss = torch.mean((pred - y_tr_t[idx]) ** 2)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(torch.mean((model(X_va_t) - y_va_t) ** 2).item())
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_epoch = epoch + 1
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(cfg.mlp_patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler, best_epoch


def predict_mlp(model: RegressionMLP, scaler: StandardScaler, X: np.ndarray, cfg: Config) -> np.ndarray:
    device = torch.device(cfg.effective_device())
    model = model.to(device).eval()
    with torch.no_grad():
        X_t = torch.tensor(scaler.transform(X), dtype=torch.float32, device=device)
        return model(X_t).detach().cpu().numpy()


def enabled_bases(cfg: Config) -> List[str]:
    names: List[str] = []
    if cfg.use_xgb:
        names.append("xgb")
    if cfg.use_rf:
        names.append("rf")
    if cfg.use_ridge:
        names.append("ridge")
    if cfg.use_elastic_net:
        names.append("elastic_net")
    if cfg.use_knn:
        names.append("knn")
    if cfg.use_mlp:
        names.append("mlp")
    if not names:
        raise ValueError("at least one base model must be enabled")
    return names


def fit_base_predict(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    cfg: Config,
) -> Tuple[Any, StandardScaler, np.ndarray]:
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    X_va_scaled = scaler.transform(X_val)
    if name == "xgb":
        model = make_xgb(cfg)
        model.fit(X_tr_scaled, y_train)
        return model, scaler, np.asarray(model.predict(X_va_scaled), dtype=np.float32)
    if name == "rf":
        model = make_rf(cfg)
        model.fit(X_tr_scaled, y_train)
        return model, scaler, np.asarray(model.predict(X_va_scaled), dtype=np.float32)
    if name == "ridge":
        model = make_ridge(cfg)
        model.fit(X_tr_scaled, y_train)
        return model, scaler, np.asarray(model.predict(X_va_scaled), dtype=np.float32)
    if name == "elastic_net":
        model = make_elastic_net(cfg, len(X_train))
        model.fit(X_tr_scaled, y_train)
        return model, scaler, np.asarray(model.predict(X_va_scaled), dtype=np.float32)
    if name == "knn":
        model = make_knn(cfg, len(X_train))
        model.fit(X_tr_scaled, y_train)
        return model, scaler, np.asarray(model.predict(X_va_scaled), dtype=np.float32)
    if name == "mlp":
        raise ValueError("mlp should be trained via the dedicated MLP path")
    raise ValueError(name)


def fit_base(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: Config,
) -> Tuple[Any, StandardScaler]:
    if name == "mlp":
        n_val = max(1, min(len(X_train) // 5, max(2, len(X_train) // 4)))
        rng = np.random.default_rng(cfg.random_state)
        order = rng.permutation(len(X_train))
        va_idx = np.sort(order[:n_val])
        tr_idx = np.sort(order[n_val:])
        X_dev = X_train[tr_idx]
        y_dev = y_train[tr_idx]
        X_val = X_train[va_idx]
        y_val = y_train[va_idx]
        if len(X_dev) < 2:
            X_dev = X_train
            y_dev = y_train
            X_val = X_train
            y_val = y_train
        model, scaler, _ = train_mlp(X_dev, y_dev, X_val, y_val, cfg)
        return model, scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    if name == "xgb":
        model = make_xgb(cfg)
    elif name == "rf":
        model = make_rf(cfg)
    elif name == "ridge":
        model = make_ridge(cfg)
    elif name == "elastic_net":
        model = make_elastic_net(cfg, len(X_train))
    elif name == "knn":
        model = make_knn(cfg, len(X_train))
    else:
        raise ValueError(name)
    sklearn_est: Any = model
    sklearn_est.fit(X_scaled, y_train)
    return sklearn_est, scaler


def predict_base(name: str, model: Any, scaler: StandardScaler, X: np.ndarray, cfg: Config) -> np.ndarray:
    if name == "mlp":
        return np.asarray(predict_mlp(model, scaler, X, cfg), dtype=np.float32)
    return np.asarray(model.predict(scaler.transform(X)), dtype=np.float32)


def fit_meta_model(Z: np.ndarray, y: np.ndarray, cfg: Config) -> Tuple[Any, StandardScaler]:
    scaler = StandardScaler()
    Z_scaled = scaler.fit_transform(Z)
    if str(cfg.meta_learner).lower() == "ridgecv":
        alphas = np.logspace(float(cfg.meta_log_alpha_min), float(cfg.meta_log_alpha_max), int(cfg.meta_n_alphas))
        model = RidgeCV(alphas=alphas)
        model.fit(Z_scaled, y)
        return model, scaler
    alphas = np.logspace(float(cfg.meta_log_alpha_min), float(cfg.meta_log_alpha_max), int(cfg.meta_n_alphas))
    kf = KFold(n_splits=min(5, max(2, len(y) // 25)), shuffle=True, random_state=cfg.random_state)
    best_alpha = float(alphas[0])
    best_score = -np.inf
    for alpha in alphas:
        scores: List[float] = []
        for tr, va in kf.split(Z_scaled):
            ridge = Ridge(alpha=float(alpha), random_state=cfg.random_state)
            ridge.fit(Z_scaled[tr], y[tr])
            scores.append(spearman_metric(y[va], ridge.predict(Z_scaled[va])))
        score = float(np.mean(scores))
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
    model = Ridge(alpha=best_alpha, random_state=cfg.random_state)
    model.fit(Z_scaled, y)
    return model, scaler


def build_meta_oof_predictions(Z: np.ndarray, y: np.ndarray, df_train: pd.DataFrame, cfg: Config) -> np.ndarray:
    pred = np.zeros(len(y), dtype=np.float32)
    for tr_idx, va_idx in lpo_splits(df_train, cfg.n_folds, cfg.random_state):
        model, scaler = fit_meta_model(Z[tr_idx], y[tr_idx], cfg)
        pred[va_idx] = np.asarray(model.predict(scaler.transform(Z[va_idx])), dtype=np.float32)
    return pred


def build_oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    df_train: pd.DataFrame,
    layout: FeatureLayout,
    cfg: Config,
) -> Tuple[np.ndarray, Dict[str, float]]:
    rng = np.random.default_rng(cfg.random_state)
    names = enabled_bases(cfg)
    oof = np.zeros((len(y), len(names)), dtype=np.float32)
    scores: Dict[str, List[float]] = {name: [] for name in names}
    for tr_idx, va_idx in lpo_splits(df_train, cfg.n_folds, cfg.random_state):
        X_tr = X[tr_idx]
        y_tr = y[tr_idx]
        X_va = X[va_idx]
        y_va = y[va_idx]
        X_tr, y_tr = smote_tail_oversample(X_tr, y_tr, cfg, cfg.random_state + int(tr_idx[0]))
        X_tr = add_embedding_noise(X_tr, layout, cfg, rng)
        X_tr, X_va, fold_layout, _ = apply_plm_pca_fold(X_tr, X_va, layout, cfg.plm_pca_components, cfg.random_state)
        for j, name in enumerate(names):
            Xb_tr = fold_layout.base_view(name, X_tr)
            Xb_va = fold_layout.base_view(name, X_va)
            if name == "mlp":
                model, scaler, _ = train_mlp(Xb_tr, y_tr, Xb_va, y_va, cfg)
                pred = predict_mlp(model, scaler, Xb_va, cfg)
            else:
                model, scaler, pred = fit_base_predict(name, Xb_tr, y_tr, Xb_va, cfg)
            pred = np.asarray(pred, dtype=np.float32).ravel()
            oof[va_idx, j] = pred
            scores[name].append(spearman_metric(y_va, pred))
    return oof, {name: float(np.mean(vals)) for name, vals in scores.items()}


def _select_final_mlp_epoch(
    X: np.ndarray,
    y: np.ndarray,
    df_train: pd.DataFrame,
    layout: FeatureLayout,
    cfg: Config,
) -> int:
    splits = list(lpo_splits(df_train, cfg.n_folds, cfg.random_state))
    if not splits:
        return max(1, int(cfg.mlp_epochs))
    tr_idx, va_idx = splits[0]
    X_tr = X[tr_idx]
    y_tr = y[tr_idx]
    X_va = X[va_idx]
    y_va = y[va_idx]
    if len(X_tr) < 2 or len(X_va) < 1:
        return max(1, int(cfg.mlp_epochs))
    rng = np.random.default_rng(cfg.random_state + 29)
    X_tr, y_tr = smote_tail_oversample(X_tr, y_tr, cfg, cfg.random_state + 29)
    X_tr = add_embedding_noise(X_tr, layout, cfg, rng)
    X_tr, X_va, fold_layout, _ = apply_plm_pca_fold(X_tr, X_va, layout, cfg.plm_pca_components, cfg.random_state)
    Xb_tr = fold_layout.base_view("mlp", X_tr)
    Xb_va = fold_layout.base_view("mlp", X_va)
    _, _, best_epoch = train_mlp(Xb_tr, y_tr, Xb_va, y_va, cfg)
    return best_epoch


def _fit_final_mlp(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    layout_fit: FeatureLayout,
    pca_fit: Optional[PCA],
    best_epoch: int,
    cfg: Config,
) -> Tuple[RegressionMLP, StandardScaler, FeatureLayout, Optional[PCA]]:
    Xb_fit = layout_fit.base_view("mlp", X_fit)
    model, scaler, _ = train_mlp(Xb_fit, y_fit, None, None, cfg, fixed_epochs=best_epoch)
    return model, scaler, layout_fit, pca_fit


def fit_bases_on_all(
    X: np.ndarray,
    y: np.ndarray,
    df_train: pd.DataFrame,
    layout: FeatureLayout,
    cfg: Config,
) -> Tuple[Dict[str, Any], Dict[str, StandardScaler], Dict[str, FeatureLayout], Dict[str, Optional[PCA]]]:
    rng = np.random.default_rng(cfg.random_state + 17)
    X_fit, y_fit = smote_tail_oversample(X, y, cfg, cfg.random_state)
    X_fit = add_embedding_noise(X_fit, layout, cfg, rng)
    X_fit, layout_work, pca = fit_dev_plm_pca(X_fit, layout, cfg)
    mlp_best_epoch = _select_final_mlp_epoch(X, y, df_train, layout, cfg) if cfg.use_mlp else max(1, int(cfg.mlp_epochs))
    models: Dict[str, Any] = {}
    scalers: Dict[str, StandardScaler] = {}
    base_layouts: Dict[str, FeatureLayout] = {}
    base_pcas: Dict[str, Optional[PCA]] = {}
    for name in enabled_bases(cfg):
        if name == "mlp":
            model, scaler, mlp_layout, mlp_pca = _fit_final_mlp(X_fit, y_fit, layout_work, pca, mlp_best_epoch, cfg)
            models[name] = model
            scalers[name] = scaler
            base_layouts[name] = mlp_layout
            base_pcas[name] = mlp_pca
            continue
        Xb = layout_work.base_view(name, X_fit)
        model, scaler = fit_base(name, Xb, y_fit, cfg)
        models[name] = model
        scalers[name] = scaler
        base_layouts[name] = layout_work
        base_pcas[name] = pca
    return models, scalers, base_layouts, base_pcas


def predict_base_matrix(stack: StackModel, X: np.ndarray, cfg: Config) -> np.ndarray:
    cols: List[np.ndarray] = []
    transform_cache: Dict[Any, Tuple[np.ndarray, FeatureLayout]] = {}
    for name in stack.base_names:
        base_pca = stack.plm_pca
        if stack.base_pcas is not None:
            base_pca = stack.base_pcas.get(name)
        cache_key = id(base_pca) if base_pca is not None else "none"
        if cache_key not in transform_cache:
            transform_cache[cache_key] = transform_plm_pca(X, stack.layout_in, base_pca)
        X_work, layout_work = transform_cache[cache_key]
        if stack.base_layouts is not None and name in stack.base_layouts:
            layout_work = stack.base_layouts[name]
        elif stack.layout_work is not None:
            layout_work = stack.layout_work
        Xb = layout_work.base_view(name, X_work)
        cols.append(predict_base(name, stack.base_models[name], stack.base_scalers[name], Xb, cfg))
    return np.column_stack(cols).astype(np.float32)


def predict_stack(stack: StackModel, X: np.ndarray, cfg: Config) -> np.ndarray:
    H = predict_base_matrix(stack, X, cfg)
    return np.asarray(stack.meta_model.predict(stack.meta_scaler.transform(H)), dtype=np.float32)


def fit_stack_with_holdout(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    df_dev: pd.DataFrame,
    X_hold: np.ndarray,
    y_hold: np.ndarray,
    layout: FeatureLayout,
    cfg: Config,
) -> Tuple[StackModel, float, Dict[str, float]]:
    Z_oof, _ = build_oof_predictions(X_dev, y_dev, df_dev, layout, cfg)
    meta_model, meta_scaler = fit_meta_model(Z_oof, y_dev, cfg)
    base_models, base_scalers, base_layouts, base_pcas = fit_bases_on_all(X_dev, y_dev, df_dev, layout, cfg)
    stack = StackModel(
        enabled_bases(cfg),
        base_models,
        base_scalers,
        meta_model,
        meta_scaler,
        layout,
        layout,
        None,
        base_layouts=base_layouts,
        base_pcas=base_pcas,
    )
    H_hold = predict_base_matrix(stack, X_hold, cfg)
    y_hold_pred = np.asarray(stack.meta_model.predict(stack.meta_scaler.transform(H_hold)), dtype=np.float32)
    holdout_rho = spearman_metric(y_hold, y_hold_pred)
    base_rhos = {name: spearman_metric(y_hold, H_hold[:, i]) for i, name in enumerate(stack.base_names)}
    return stack, holdout_rho, base_rhos


def repeated_holdout_stack_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    df_train: pd.DataFrame,
    layout: FeatureLayout,
    cfg: Config,
) -> RepeatedHoldoutResult:
    n = len(y)
    n_hold = max(1, int(round(n * cfg.holdout_frac)))
    stack_scores: List[float] = []
    base_scores: List[Dict[str, float]] = []
    for repeat in range(int(cfg.repeated_holdout_n)):
        rng = np.random.default_rng(cfg.random_state + repeat * 10007)
        hold_idx = np.sort(rng.choice(n, size=n_hold, replace=False))
        mask = np.ones(n, dtype=bool)
        mask[hold_idx] = False
        train_idx = np.flatnonzero(mask)
        stack, rho, rho_bases = fit_stack_with_holdout(
            X[train_idx],
            y[train_idx],
            df_train.iloc[train_idx].reset_index(drop=True),
            X[hold_idx],
            y[hold_idx],
            layout,
            cfg,
        )
        _ = stack
        stack_scores.append(float(rho))
        base_scores.append(rho_bases)
    names = sorted(set().union(*(row.keys() for row in base_scores)))
    mean_bases = {name: float(np.nanmean([row.get(name, np.nan) for row in base_scores])) for name in names}
    std_bases = {name: float(np.nanstd([row.get(name, np.nan) for row in base_scores])) for name in names}
    return RepeatedHoldoutResult(stack_scores, base_scores, float(np.mean(stack_scores)), float(np.std(stack_scores)), mean_bases, std_bases)


def predictions_dataframe(df_test: pd.DataFrame, y_pred: np.ndarray, cfg: Config) -> pd.DataFrame:
    out = pd.DataFrame({"mutant": df_test["mutant"].astype(str), "DMS_score_predicted": np.asarray(y_pred, dtype=np.float32)})
    if cfg.clip_predictions:
        out["DMS_score_predicted"] = out["DMS_score_predicted"].clip(0.0, 1.0)
    return out


def top10_mutants(df_test: pd.DataFrame, y_pred: np.ndarray, original_train_mutants: Set[str], cfg: Config) -> List[str]:
    pred_df = predictions_dataframe(df_test, y_pred, cfg)
    pred_df = pred_df.loc[~pred_df["mutant"].isin(original_train_mutants)].sort_values("DMS_score_predicted", ascending=False)
    return pred_df.head(10)["mutant"].astype(str).tolist()


def save_predictions_csv(pred_df: pd.DataFrame, path: Path) -> None:
    pred_df[["mutant", "DMS_score_predicted"]].to_csv(path, index=False)


def save_top10(mutants: Sequence[str], path: Path) -> None:
    path.write_text("\n".join(str(m) for m in mutants))


def write_query_txt(mutants: Sequence[str], path: Path) -> None:
    path.write_text("\n".join(str(m) for m in mutants))


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def _expected_improvement(mean: np.ndarray, std: np.ndarray, best: float) -> np.ndarray:
    std = np.maximum(np.asarray(std, dtype=np.float64), 1e-8)
    mean = np.asarray(mean, dtype=np.float64)
    z = (mean - best) / std
    return (mean - best) * norm.cdf(z) + std * norm.pdf(z)


def per_substitution_errors(df_train: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, min_group_n: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    groups: Dict[Tuple[str, str], List[int]] = {}
    for i, mutant in enumerate(df_train["mutant"].astype(str)):
        wt, _, mt = parse_mutant(mutant)
        groups.setdefault((wt, mt), []).append(i)
    for (wt, mt), idx in sorted(groups.items()):
        if len(idx) < int(min_group_n):
            continue
        idx_arr = np.asarray(idx, dtype=int)
        yt = np.asarray(y_true)[idx_arr]
        yp = np.asarray(y_pred)[idx_arr]
        rows.append(
            {
                "wt": wt,
                "mt": mt,
                "n": int(len(idx)),
                "mae": float(np.mean(np.abs(yt - yp))),
                "spearman": spearman_metric(yt, yp),
            }
        )
    return pd.DataFrame(rows)


def build_substitution_weight_map(df_train: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, cfg: Config) -> Dict[Tuple[str, str], float]:
    diag = per_substitution_errors(df_train, y_true, y_pred, int(cfg.diagnostic_weight_min_n))
    if diag.empty:
        return {}
    signal = diag["mae"].to_numpy(dtype=np.float64)
    signal = _normalize(signal) ** float(cfg.diagnostic_weight_power)
    lo, hi = float(cfg.diagnostic_weight_clip[0]), float(cfg.diagnostic_weight_clip[1])
    weights = lo + (hi - lo) * signal
    return {(str(row["wt"]), str(row["mt"])): float(weight) for (_, row), weight in zip(diag.iterrows(), weights)}


def diagnostic_test_weights(df_train: pd.DataFrame, y_true: np.ndarray, y_pred_labeled: np.ndarray, df_test: pd.DataFrame, cfg: Config) -> np.ndarray:
    weight_map = build_substitution_weight_map(df_train, y_true, y_pred_labeled, cfg)
    weights = np.ones(len(df_test), dtype=np.float64)
    for i, mutant in enumerate(df_test["mutant"].astype(str)):
        wt, _, mt = parse_mutant(mutant)
        weights[i] = weight_map.get((wt, mt), 1.0)
    return weights


def _largest_remainder_counts(total: int, fractions: Sequence[float]) -> List[int]:
    fr = np.asarray(fractions, dtype=np.float64)
    fr = fr / fr.sum()
    exact = fr * total
    floors = np.floor(exact).astype(int)
    rem = int(total - floors.sum())
    order = np.argsort(-(exact - floors))
    for i in range(rem):
        floors[order[i]] += 1
    return floors.tolist()


def _diverse_pick(ranked_idx: np.ndarray, score: np.ndarray, repr_test: np.ndarray, budget: int, lam: float) -> List[int]:
    if budget <= 0:
        return []
    if lam <= 0:
        return ranked_idx[:budget].tolist()
    chosen: List[int] = []
    remaining = list(ranked_idx.tolist())
    while remaining and len(chosen) < budget:
        best_idx = remaining[0]
        best_score = -np.inf
        for idx in remaining:
            if not chosen:
                best_idx = idx
                break
            sims = []
            for other in chosen:
                denom = np.linalg.norm(repr_test[idx]) * np.linalg.norm(repr_test[other]) + 1e-8
                sims.append(float(np.dot(repr_test[idx], repr_test[other]) / denom))
            adjusted = float(score[idx]) - float(lam) * max(sims)
            if adjusted > best_score:
                best_score = adjusted
                best_idx = idx
        chosen.append(best_idx)
        remaining.remove(best_idx)
    return chosen


def _top_neighbor_pool(seed_idx: Sequence[int], repr_test: np.ndarray, k: int) -> List[int]:
    if len(seed_idx) == 0 or len(repr_test) == 0:
        return []
    nn = NearestNeighbors(n_neighbors=min(max(1, int(k)), len(repr_test)), metric="cosine")
    nn.fit(repr_test)
    pool: Set[int] = set()
    for idx in seed_idx:
        _, inds = nn.kneighbors(repr_test[int(idx) : int(idx) + 1], return_distance=True)
        pool.update(int(i) for i in inds.ravel())
    return sorted(pool)


def _round_query_params(query_round: str, cfg: Config) -> Dict[str, Any]:
    qr = str(query_round).lower()
    if qr == "round3":
        return {
            "hybrid_alpha": float(cfg.round3_hybrid_alpha),
            "ucb_beta": float(cfg.round3_ucb_beta),
            "diversity_lambda": float(cfg.round3_diversity_lambda),
            "destination_min": cfg.round3_destination_min,
            "destination_max": cfg.round3_destination_max,
        }
    return {
        "hybrid_alpha": float(cfg.hybrid_alpha),
        "ucb_beta": float(cfg.ucb_beta),
        "diversity_lambda": float(cfg.diversity_lambda),
        "destination_min": cfg.query_destination_min,
        "destination_max": cfg.query_destination_max,
    }


def _apply_destination_constraints(
    selected: List[str],
    eligible_order: List[str],
    cap: int,
    min_per_destination: Optional[int],
    max_per_destination: Optional[int],
) -> List[str]:
    eligible_set = set(eligible_order)
    dest_map = {mutant: parse_mutant(mutant)[2] for mutant in eligible_order}
    rank = {mutant: i for i, mutant in enumerate(eligible_order)}
    out: List[str] = []
    seen: Set[str] = set()
    for mutant in selected:
        if mutant in seen or mutant not in eligible_set:
            continue
        seen.add(mutant)
        out.append(mutant)
        if len(out) >= cap:
            break

    dest_to_candidates: Dict[str, List[str]] = {}
    for mutant in eligible_order:
        dest_to_candidates.setdefault(dest_map[mutant], []).append(mutant)

    max_limit = int(max_per_destination) if max_per_destination else None
    raw_need = int(min_per_destination) if min_per_destination else 0

    def required_count(destination: str) -> int:
        if raw_need <= 0:
            return 0
        available = len(dest_to_candidates.get(destination, ()))
        need = min(raw_need, available)
        if max_limit is not None:
            need = min(need, max_limit)
        return need

    def count_dest(mutants: Sequence[str]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for mutant in mutants:
            mt = dest_map[mutant]
            counts[mt] = counts.get(mt, 0) + 1
        return counts

    counts = count_dest(out)
    if max_limit is not None:
        drop: Set[str] = set()
        by_dest: Dict[str, List[str]] = {}
        for mutant in out:
            by_dest.setdefault(dest_map[mutant], []).append(mutant)
        for mutants in by_dest.values():
            if len(mutants) <= max_limit:
                continue
            keep = set(sorted(mutants, key=lambda mutant: rank[mutant])[:max_limit])
            drop |= {mutant for mutant in mutants if mutant not in keep}
        if drop:
            out = [mutant for mutant in out if mutant not in drop]
            counts = count_dest(out)

    out_set = set(out)
    if raw_need > 0:
        for destination in sorted(dest_to_candidates):
            need = required_count(destination)
            while counts.get(destination, 0) < need:
                candidate = next((mutant for mutant in dest_to_candidates[destination] if mutant not in out_set), None)
                if candidate is None:
                    break
                if len(out) >= cap:
                    removable = [
                        mutant
                        for mutant in out
                        if counts.get(dest_map[mutant], 0) > required_count(dest_map[mutant])
                    ]
                    if not removable:
                        break
                    evict = max(removable, key=lambda mutant: (rank[mutant], out.index(mutant)))
                    out.remove(evict)
                    out_set.remove(evict)
                    counts[dest_map[evict]] -= 1
                if max_limit is not None and counts.get(destination, 0) >= max_limit:
                    break
                out.append(candidate)
                out_set.add(candidate)
                counts[destination] = counts.get(destination, 0) + 1

    for mutant in eligible_order:
        if len(out) >= cap:
            break
        if mutant in out_set:
            continue
        destination = dest_map[mutant]
        if max_limit is not None and counts.get(destination, 0) >= max_limit:
            continue
        out.append(mutant)
        out_set.add(mutant)
        counts[destination] = counts.get(destination, 0) + 1
    return out[:cap]


def _query_plan(query_round: str, cfg: Config) -> List[Tuple[str, int]]:
    cap = min(int(cfg.query_budget), 100)
    qr = str(query_round).lower()
    if qr == "round1":
        return [("hybrid", cap)]
    if qr == "round2":
        budgets = _largest_remainder_counts(cap, cfg.round2_fractions)
        return list(zip(["ucb_ei", "greedy", "hybrid", "random"], budgets))
    if qr == "round3":
        return list(zip(["greedy", "ucb", "hybrid", "random"], cfg.round3_budgets))
    raise ValueError("query_round must be 'round1', 'round2', or 'round3'")


def select_queries(
    query_round: str,
    cfg: Config,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    X_train: np.ndarray,
    X_test: np.ndarray,
    layout: FeatureLayout,
    y_pred_test: np.ndarray,
    base_test_pred: np.ndarray,
    y_pred_labeled: Optional[np.ndarray] = None,
) -> List[str]:
    cap = min(int(cfg.query_budget), 100)
    round_params = _round_query_params(query_round, cfg)
    blocked_baseline = set(df_train["mutant"].astype(str)) | collect_prior_query_mutants(cfg)
    blocked = set(blocked_baseline)
    test_mutants = df_test["mutant"].astype(str).tolist()
    repr_train = layout.distance_repr(X_train)
    repr_test = layout.distance_repr(X_test)
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(repr_train)
    dist = nn.kneighbors(repr_test, return_distance=True)[0].ravel()
    dist_norm = _normalize(dist)
    mean = np.asarray(y_pred_test, dtype=np.float64).ravel()
    if base_test_pred.shape[1] >= 2:
        std = np.std(base_test_pred, axis=1, ddof=0).astype(np.float64)
    else:
        std = np.full(len(df_test), max(1e-6, float(np.std(mean) * 0.05)), dtype=np.float64)
    weights = np.ones(len(df_test), dtype=np.float64)
    if str(query_round).lower() == "round2" and y_pred_labeled is not None:
        weights = diagnostic_test_weights(
            df_train,
            df_train["DMS_score"].to_numpy(dtype=np.float32),
            np.asarray(y_pred_labeled, dtype=np.float32),
            df_test,
            cfg,
        )
    selected: List[str] = []
    eligible_order = [test_mutants[int(i)] for i in np.argsort(-(mean * weights)) if test_mutants[int(i)] not in blocked_baseline]
    best_so_far = float(np.max(df_train["DMS_score"].to_numpy(dtype=np.float64)))
    if str(query_round).lower() == "round3":
        neighbor_take = min(cap, max(0, int(round(cap * float(cfg.round3_neighbor_fraction)))))
        if neighbor_take > 0:
            seed_idx = np.array([i for i in np.argsort(-mean) if test_mutants[i] not in blocked], dtype=int)[:10]
            pool_idx = _top_neighbor_pool(seed_idx.tolist(), repr_test, int(cfg.round3_neighbor_k))
            pool_ranked = sorted(pool_idx, key=lambda i: -mean[int(i)])
            for idx in pool_ranked:
                mutant = test_mutants[int(idx)]
                if mutant in blocked:
                    continue
                blocked.add(mutant)
                selected.append(mutant)
                if len(selected) >= neighbor_take:
                    break
    for strategy, budget in _query_plan(query_round, cfg):
        if len(selected) >= cap:
            break
        remaining = cap - len(selected)
        take = min(int(budget), remaining)
        eligible_idx = np.array([i for i, mutant in enumerate(test_mutants) if mutant not in blocked], dtype=int)
        if len(eligible_idx) == 0 or take <= 0:
            continue
        if strategy == "random":
            rng = np.random.default_rng(cfg.random_state + len(selected))
            p = weights[eligible_idx].astype(np.float64)
            p = p / p.sum() if float(p.sum()) > 0 else None
            chosen = rng.choice(eligible_idx, size=min(take, len(eligible_idx)), replace=False, p=p)
            ranked = chosen.tolist()
        elif strategy == "greedy":
            score = mean * weights
            ranked = eligible_idx[np.argsort(-score[eligible_idx])][:take].tolist()
        elif strategy == "hybrid":
            score = (
                float(round_params["hybrid_alpha"]) * dist_norm
                + (1.0 - float(round_params["hybrid_alpha"])) * _normalize(mean * weights)
            )
            ranked = eligible_idx[np.argsort(-score[eligible_idx])][:take].tolist()
        elif strategy == "ucb":
            score = mean + float(round_params["ucb_beta"]) * std
            score = score * weights
            ordered = eligible_idx[np.argsort(-score[eligible_idx])]
            ranked = _diverse_pick(ordered, score, repr_test, take, float(round_params["diversity_lambda"]))
        elif strategy == "ucb_ei":
            ucb = mean + float(round_params["ucb_beta"]) * std
            ei = _expected_improvement(mean, std, best_so_far)
            score = 0.5 * _normalize(ucb) + 0.5 * _normalize(ei)
            score = score * weights
            ordered = eligible_idx[np.argsort(-score[eligible_idx])]
            ranked = _diverse_pick(ordered, score, repr_test, take, float(round_params["diversity_lambda"]))
        else:
            raise ValueError(strategy)
        for idx in ranked:
            mutant = test_mutants[int(idx)]
            if mutant in blocked:
                continue
            blocked.add(mutant)
            selected.append(mutant)
            if len(selected) >= cap:
                break
    return _apply_destination_constraints(
        selected,
        eligible_order,
        cap,
        round_params["destination_min"],
        round_params["destination_max"],
    )


def tune_xgb(X: np.ndarray, y: np.ndarray, df_train: pd.DataFrame, layout: FeatureLayout, cfg: Config) -> Dict[str, Any]:
    if not cfg.use_optuna:
        return {}
    import optuna

    X_base = layout.traditional(X)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        }
        scores: List[float] = []
        for tr, va in lpo_splits(df_train, cfg.n_folds, cfg.random_state):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_base[tr])
            X_va = scaler.transform(X_base[va])
            model = make_xgb(cfg, n_jobs=int(cfg.optuna_xgb_n_jobs), **params)
            model.fit(X_tr, y[tr])
            scores.append(spearman_metric(y[va], model.predict(X_va)))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.random_state))
    study.optimize(objective, n_trials=int(cfg.optuna_n_trials_xgb), show_progress_bar=False, gc_after_trial=True)
    return dict(study.best_params)


def tune_mlp(X: np.ndarray, y: np.ndarray, df_train: pd.DataFrame, layout: FeatureLayout, cfg: Config) -> Dict[str, Any]:
    if not cfg.use_optuna or not cfg.use_mlp:
        return {}
    import optuna

    def objective(trial: optuna.Trial) -> float:
        hidden = trial.suggest_categorical("hidden_dim", [128, 256, 384])
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        scores: List[float] = []
        for tr, va in lpo_splits(df_train, cfg.n_folds, cfg.random_state):
            X_tr = X[tr]
            X_va = X[va]
            X_tr, X_va, fold_layout, _ = apply_plm_pca_fold(X_tr, X_va, layout, cfg.plm_pca_components, cfg.random_state)
            Xb_tr = fold_layout.base_view("mlp", X_tr)
            Xb_va = fold_layout.base_view("mlp", X_va)
            model, scaler, _ = train_mlp(Xb_tr, y[tr], Xb_va, y[va], cfg, hidden_dim=hidden, dropout=dropout)
            pred = predict_mlp(model, scaler, Xb_va, cfg)
            scores.append(spearman_metric(y[va], pred))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.random_state))
    study.optimize(objective, n_trials=int(cfg.optuna_n_trials_mlp), show_progress_bar=False, gc_after_trial=True)
    return dict(study.best_params)


def run_pipeline(cfg: Config, run_optuna: bool = False) -> PipelineResults:
    seed_everything(cfg.random_state)
    df_train, df_test, wt_sequence, original_train_mutants = load_train_test(cfg)
    df_train = merge_query_results(df_train, wt_sequence, cfg)
    X_train, layout = build_feature_matrix(df_train, wt_sequence, "train", cfg)
    X_test, _ = build_feature_matrix(df_test, wt_sequence, "test", cfg)
    y_train = df_train["DMS_score"].to_numpy(dtype=np.float32)
    if run_optuna:
        cfg.use_optuna = True
        xgb_best = tune_xgb(X_train, y_train, df_train, layout, cfg)
        if xgb_best:
            cfg.xgb_overrides.update(xgb_best)
        mlp_best = tune_mlp(X_train, y_train, df_train, layout, cfg)
        if mlp_best:
            cfg.mlp_hidden_dim = int(mlp_best.get("hidden_dim", cfg.mlp_hidden_dim))
            cfg.mlp_dropout = float(mlp_best.get("dropout", cfg.mlp_dropout))
    hold_idx = make_saved_holdout_indices(len(df_train), df_train["mutant"].astype(str).tolist(), cfg)
    dev_mask = np.ones(len(df_train), dtype=bool)
    dev_mask[hold_idx] = False
    dev_idx = np.flatnonzero(dev_mask)
    stack_dev, holdout_spearman, holdout_base_spearmans = fit_stack_with_holdout(
        X_train[dev_idx],
        y_train[dev_idx],
        df_train.iloc[dev_idx].reset_index(drop=True),
        X_train[hold_idx],
        y_train[hold_idx],
        layout,
        cfg,
    )
    _ = stack_dev
    repeated = None
    if int(cfg.repeated_holdout_n) > 1:
        repeated = repeated_holdout_stack_evaluation(X_train, y_train, df_train, layout, cfg)
    Z_oof, _ = build_oof_predictions(X_train, y_train, df_train, layout, cfg)
    y_train_pred_oof = build_meta_oof_predictions(Z_oof, y_train, df_train, cfg)
    meta_model, meta_scaler = fit_meta_model(Z_oof, y_train, cfg)
    base_models, base_scalers, base_layouts, base_pcas = fit_bases_on_all(X_train, y_train, df_train, layout, cfg)
    stack = StackModel(
        enabled_bases(cfg),
        base_models,
        base_scalers,
        meta_model,
        meta_scaler,
        layout,
        layout,
        None,
        base_layouts=base_layouts,
        base_pcas=base_pcas,
    )
    y_test_pred = predict_stack(stack, X_test, cfg)
    pred_df = predictions_dataframe(df_test, y_test_pred, cfg)
    return PipelineResults(
        cfg=cfg,
        df_train=df_train,
        df_test=df_test,
        wt_sequence=wt_sequence,
        original_train_mutants=original_train_mutants,
        X_train=X_train,
        X_test=X_test,
        feature_layout=layout,
        holdout_spearman=float(holdout_spearman),
        holdout_base_spearmans=holdout_base_spearmans,
        repeated_holdout=repeated,
        stack=stack,
        y_train_pred_oof=y_train_pred_oof,
        y_test_pred=y_test_pred,
        predictions=pred_df,
    )


def export_submission_files(results: PipelineResults) -> Dict[str, Path]:
    root = results.cfg.output_path()
    pred_path = root / "predictions.csv"
    top10_path = root / "top10.txt"
    save_predictions_csv(results.predictions, pred_path)
    save_top10(top10_mutants(results.df_test, results.y_test_pred, results.original_train_mutants, results.cfg), top10_path)
    return {"predictions": pred_path, "top10": top10_path}


def export_query_file(results: PipelineResults, query_round: str) -> List[str]:
    base_test_pred = predict_base_matrix(results.stack, results.X_test, results.cfg)
    query = select_queries(
        query_round,
        results.cfg,
        results.df_train,
        results.df_test,
        results.X_train,
        results.X_test,
        results.feature_layout,
        results.y_test_pred,
        base_test_pred,
        y_pred_labeled=results.y_train_pred_oof,
    )
    write_query_txt(query, results.cfg.query_txt_path())
    return query

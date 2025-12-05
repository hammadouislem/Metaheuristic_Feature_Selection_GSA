import os
import time
import json
import random
import logging
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any, Optional
from functools import lru_cache
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gsa_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ----------------------------
# ========== CONFIG ==========
# ----------------------------
# Map dataset filenames to their metadata you provided.
# If any mapping is wrong, edit this dict.
datasets_meta = {
    "arrhythmia.csv": {
        "target_column_name": "class",
        "target_column_index": 279,
        "has_header": True,
        "numeric_only": True,
        "missing_values": True
    },
    "colon_cancer.csv": {
        "target_column_name": "class",
        "target_column_index": 2000,
        "has_header": True,
        "numeric_only": True,
        "missing_values": False
    },
    # Assuming this is dermatology (you gave a block that likely maps)
    "darmatology.csv": {
        "target_column_name": "class",
        "target_column_index": 34,
        "has_header": True,
        "numeric_only": True,
        "missing_values": False
    },
    # Assuming this is heart_stat.csv (the provided block with index 13, non-numeric features)
    "heart_stat.csv": {
        "target_column_name": "class",
        "target_column_index": 13,
        "has_header": True,
        "numeric_only": False,
        "missing_values": False
    },
    "vehicule.csv": {
        "target_column_name": "Class",
        "target_column_index": 18,
        "has_header": True,
        "numeric_only": False,
        "missing_values": False
    }
}

# Experiment hyperparameters (you can tune these)
N_RUNS = 10                # number of independent runs per dataset
N_AGENTS = 12              # population size (lowered for speed)
MAX_ITER = 25              # iterations (balance between quality/time)
K_NEIGHBORS = 3            # k for KNN
CV_FOLDS = 3               # cross-validation folds for fitness
LOWER = -6.0               # lower bound for continuous position
UPPER = 6.0                # upper bound
G0 = 50.0                  # initial gravitational constant
RANDOM_SEED = 123          # reproducibility seed
OUTPUT_DIR = "gsa_results" # where to save reports

# make output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# ========== UTIL =============
# ----------------------------
def S4(x: np.ndarray) -> np.ndarray:
    """S4 transfer function (vectorized)."""
    return 1.0 / (1.0 + np.exp(-x / 3.0))

def binarize_position(position: np.ndarray, rng: np.random.RandomState = None) -> np.ndarray:
    """Convert continuous position vector -> binary mask using S4."""
    if rng is None:
        rng = np.random.RandomState()
    probs = S4(position)
    rand = rng.rand(*probs.shape)
    return (rand < probs).astype(int)

def knn_fitness(mask: np.ndarray, X: np.ndarray, y: np.ndarray, k: int = 3, cv: int = 3) -> float:
    """Evaluate binary mask with KNN and cross-validated accuracy."""
    if mask.sum() == 0:
        return 0.0
    
    # Check minimum class size to determine appropriate CV folds
    class_counts = Counter(y)
    min_class_size = min(class_counts.values())
    
    # Adjust CV folds to not exceed minimum class size
    effective_cv = min(cv, min_class_size, len(y))
    
    if effective_cv < cv:
        logger.debug(f"Reduced CV folds from {cv} to {effective_cv} (min class size: {min_class_size})")
    
    Xs = X[:, mask.astype(bool)]
    clf = KNeighborsClassifier(n_neighbors=min(k, len(y) - 1))
    try:
        # Use StratifiedKFold for better handling of imbalanced classes
        if effective_cv >= 2:
            cv_splitter = StratifiedKFold(n_splits=effective_cv, shuffle=True, random_state=42)
            scores = cross_val_score(clf, Xs, y, cv=cv_splitter, scoring='accuracy', n_jobs=1)
        else:
            # Fallback: if only 1 fold possible, use simple train/test split
            X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.3, random_state=42, stratify=y)
            clf.fit(X_train, y_train)
            scores = [clf.score(X_test, y_test)]
        return float(scores.mean())
    except Exception as e:
        logger.warning(f"cross_val_score failed: {e}")
        return 0.0

# ----------------------------
# ========== GSA =============
# ----------------------------
def gravitational_search_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    n_agents: int = N_AGENTS,
    max_iter: int = MAX_ITER,
    k: int = K_NEIGHBORS,
    cv: int = CV_FOLDS,
    dim: int = None,
    lower: float = LOWER,
    upper: float = UPPER,
    G0_param: float = G0,
    rng: np.random.RandomState = None,
    early_stopping_patience: int = None
) -> Dict[str, Any]:
    """
    Binary GSA for feature selection returning best mask and history.
    
    Args:
        early_stopping_patience: Stop if no improvement for N iterations (None = disabled)
    """
    if rng is None:
        rng = np.random.RandomState()
    if dim is None:
        dim = X.shape[1]
    
    # Validate inputs
    if X.shape[0] < cv:
        raise ValueError(f"Not enough samples ({X.shape[0]}) for {cv}-fold CV")
    if X.shape[0] == 0 or dim == 0:
        raise ValueError("Empty dataset or no features")

    # initialize continuous positions and velocities
    positions = rng.uniform(lower, upper, size=(n_agents, dim))
    velocities = np.zeros_like(positions)

    best_score = -1.0
    best_mask = np.zeros(dim, dtype=int)
    history = []
    no_improvement_count = 0
    iteration_best_before = best_score

    for t in range(max_iter):
        accuracies = np.zeros(n_agents)

        # evaluate all agents
        for i in range(n_agents):
            mask = binarize_position(positions[i], rng=rng)
            accuracies[i] = knn_fitness(mask, X, y, k=k, cv=cv)
            if accuracies[i] > best_score:
                best_score = float(accuracies[i])
                best_mask = mask.copy()
        
        # Check if iteration improved best score
        if best_score > iteration_best_before:
            no_improvement_count = 0
            iteration_best_before = best_score
        else:
            no_improvement_count += 1

        costs = 1.0 - accuracies  # convert to minimization
        worst = costs.max()
        best = costs.min()

        # compute masses
        if worst == best:
            masses = np.ones(n_agents) / n_agents
        else:
            inv = (costs - worst) / (best - worst + 1e-12)
            # stabilize
            inv = np.abs(inv)
            if inv.sum() == 0:
                masses = np.ones(n_agents) / n_agents
            else:
                masses = inv / inv.sum()

        # gravitational constant decays
        G = G0_param * (1.0 - (t / float(max_iter)))

        # Vectorized force calculation (more efficient)
        for i in range(n_agents):
            # Compute distances to all other agents
            diffs = positions - positions[i]  # (n_agents, dim)
            dists = np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-12  # (n_agents, 1)
            dists[i] = np.inf  # exclude self
            
            # Compute forces from all agents
            rand_vecs = rng.rand(n_agents, dim)
            forces = rand_vecs * G * masses[:, np.newaxis] * diffs / dists
            force = forces.sum(axis=0)  # sum over all agents
            
            acc = force / (masses[i] + 1e-12)
            velocities[i] = rng.rand() * velocities[i] + acc
            positions[i] = positions[i] + velocities[i]
            positions[i] = np.clip(positions[i], lower, upper)

        history.append(best_score)
        
        # Early stopping
        if early_stopping_patience and no_improvement_count >= early_stopping_patience:
            logger.info(f"Early stopping at iteration {t+1} (no improvement for {early_stopping_patience} iterations)")
            break

    return {
        "best_mask": best_mask,
        "best_score": best_score,
        "history": history
    }

# ----------------------------
# ========== PREPROCESS ==========
# ----------------------------
def load_and_preprocess(path: str, meta: dict):
    """
    Load csv, preprocess:
    - If missing values: impute with median for numeric
    - If non-numeric features exist: pd.get_dummies
    - Label encode target column
    - Standard scale features
    Returns: X (np.ndarray), y (np.ndarray), feature_names (list)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    # read CSV
    header = 0 if meta.get("has_header", True) else None
    try:
        df = pd.read_csv(path, header=header)
    except Exception as e:
        raise ValueError(f"Failed to read CSV {path}: {e}")
    
    if df.empty:
        raise ValueError(f"Dataset {path} is empty")

    # If target specified by name use it, else by index
    target_name = meta.get("target_column_name")
    if target_name and target_name in df.columns:
        target_col = target_name
    else:
        idx = meta.get("target_column_index")
        if idx is not None and idx < df.shape[1]:
            target_col = df.columns[idx]
        else:
            # fallback: assume last column
            target_col = df.columns[-1]

    # Separate X/y
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])

    # Handle missing values: numeric median impute; for categorical, mode
    if meta.get("missing_values", False):
        for c in X.columns:
            if pd.api.types.is_numeric_dtype(X[c]):
                med = X[c].median()
                X[c] = X[c].fillna(med)
            else:
                mode = X[c].mode()
                X[c] = X[c].fillna(mode.iloc[0] if len(mode) > 0 else "")

    # Convert non-numeric features with one-hot encoding (pd.get_dummies)
    # This allows KNN to operate on numeric matrix
    X = pd.get_dummies(X, drop_first=True)

    # encode target
    if not pd.api.types.is_numeric_dtype(y):
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        # still encode if numeric but not 0/1 labels
        # convert to integers if needed
        try:
            y = y.astype(int).values
        except Exception:
            le = LabelEncoder()
            y = le.fit_transform(y)

    # final conversion and scaling
    feature_names = X.columns.tolist()
    X_values = X.values.astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_values)

    return X_scaled, y, feature_names

# ----------------------------
# ========== RUNNER ==========
# ----------------------------
def run_experiments(data_dir: str = ".", meta_map: Dict[str, dict] = datasets_meta):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    summary_rows = []

    for filename, meta in meta_map.items():
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            logger.warning(f"{filename} not found at {path}. Skipping.")
            continue

        logger.info(f"\n=== Running dataset: {filename} ===")
        try:
            X, y, feature_names = load_and_preprocess(path, meta)
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}. Skipping.")
            continue
        n_features = X.shape[1]
        logger.info(f"Features after encoding: {n_features}, Samples: {X.shape[0]}")

        dataset_results = []
        dataset_dir = os.path.join(OUTPUT_DIR, filename.replace('.','_'))
        os.makedirs(dataset_dir, exist_ok=True)

        for run_idx in tqdm(range(1, N_RUNS + 1), desc=f"{filename} runs"):
            seed_run = RANDOM_SEED + run_idx
            rng = np.random.RandomState(seed_run)
            start_time = time.time()
            res = gravitational_search_feature_selection(
                X, y,
                n_agents=N_AGENTS,
                max_iter=MAX_ITER,
                k=K_NEIGHBORS,
                cv=CV_FOLDS,
                dim=n_features,
                lower=LOWER,
                upper=UPPER,
                G0_param=G0,
                rng=rng
            )
            elapsed = time.time() - start_time
            selected_mask = res["best_mask"]
            selected_count = int(selected_mask.sum())
            selected_indices = [int(i) for i in np.where(selected_mask == 1)[0]]

            # Get actual feature names for selected features
            selected_feature_names = [feature_names[i] for i in selected_indices]
            
            row = {
                "dataset": filename,
                "run": run_idx,
                "best_accuracy": float(res["best_score"]),
                "selected_features_count": selected_count,
                "selected_feature_indices": json.dumps(selected_indices),
                "selected_feature_names": json.dumps(selected_feature_names),
                "time_seconds": elapsed
            }
            dataset_results.append(row)
            # optionally save run history plot data
            pd.DataFrame({"iteration": list(range(1, len(res["history"]) + 1)), "best_accuracy": res["history"]}) \
                .to_csv(os.path.join(dataset_dir, f"run_{run_idx}_history.csv"), index=False)

            # lightweight runner print
            tqdm.write(f"Run {run_idx}: acc={res['best_score']:.4f}, selected={selected_count}, time={elapsed:.1f}s")

        # save dataset results to CSV
        df_runs = pd.DataFrame(dataset_results)
        df_runs.to_csv(os.path.join(dataset_dir, f"{filename}_gsa_runs_summary.csv"), index=False)

        # aggregate summary row
        summary_rows.append({
            "dataset": filename,
            "n_features_after_encoding": n_features,
            "runs": N_RUNS,
            "mean_accuracy": float(df_runs["best_accuracy"].mean()),
            "std_accuracy": float(df_runs["best_accuracy"].std()),
            "mean_selected": float(df_runs["selected_features_count"].mean()),
            "std_selected": float(df_runs["selected_features_count"].std()),
            "mean_time_s": float(df_runs["time_seconds"].mean())
        })

    # global summary
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(os.path.join(OUTPUT_DIR, "global_summary.csv"), index=False)
        logger.info(f"\nAll experiments completed. Summary saved to: {OUTPUT_DIR}")
        logger.info(f"\nSummary:\n{df_summary.to_string()}")
    else:
        logger.warning("No experiments completed. Check dataset paths and configuration.")

if __name__ == "__main__":
    # default data_dir is current directory; change if your CSVs are elsewhere
    run_experiments(data_dir=".")

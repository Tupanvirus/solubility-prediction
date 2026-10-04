"""
train.py

Trains a Random Forest model to predict aqueous solubility (log S, mol/L)
from RDKit molecular descriptors, using the Delaney/ESOL benchmark dataset.

Usage:
    python src/train.py \
        --data data/delaney_raw.csv \
        --out-model results/model.joblib \
        --out-metrics results/metrics.json \
        --out-plot results/predicted_vs_actual.png

The script:
    1. Loads the raw CSV (Compound ID, measured log solubility, SMILES, ...).
    2. Computes RDKit descriptors for every molecule (see src/featurize.py).
    3. Splits into train/test sets (80/20, fixed random seed for
       reproducibility).
    4. Trains a RandomForestRegressor.
    5. Reports R^2, RMSE and MAE on the held-out test set.
    6. Saves the trained model, a metrics JSON file, and a
       predicted-vs-actual scatter plot.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # no display available in headless/CI environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from featurize import DESCRIPTOR_COLUMNS, featurize_dataframe

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
TARGET_COLUMN = "measured log solubility in mols per litre"
SMILES_COLUMN = "smiles"


def load_and_featurize(data_path: Path) -> pd.DataFrame:
    """Load the raw CSV and compute descriptors for every molecule."""
    df = pd.read_csv(data_path)
    logger.info("Loaded %d rows from %s", len(df), data_path)

    df = featurize_dataframe(df, smiles_col=SMILES_COLUMN)
    logger.info("Featurized %d rows (after dropping unparseable SMILES)", len(df))
    return df


def train_and_evaluate(
    df: pd.DataFrame,
) -> tuple[RandomForestRegressor, dict, np.ndarray, np.ndarray]:
    """Train a RandomForestRegressor and evaluate it on a held-out test set."""
    X = df[DESCRIPTOR_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    feature_importances = dict(
        zip(DESCRIPTOR_COLUMNS, model.feature_importances_.round(4).tolist())
    )
    metrics["feature_importances"] = feature_importances

    return model, metrics, y_test.to_numpy(), y_pred


def plot_predicted_vs_actual(y_test: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    """Scatter plot of predicted vs. measured solubility, with a y=x line."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.6, edgecolor="k", linewidth=0.3)

    lims = [min(y_test.min(), y_pred.min()) - 0.5, max(y_test.max(), y_pred.max()) + 0.5]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction (y = x)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("Measured log solubility (mol/L)")
    ax.set_ylabel("Predicted log solubility (mol/L)")
    ax.set_title("Random Forest: predicted vs. measured aqueous solubility")
    ax.legend()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved plot to %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/delaney_raw.csv"))
    parser.add_argument("--out-model", type=Path, default=Path("results/model.joblib"))
    parser.add_argument("--out-metrics", type=Path, default=Path("results/metrics.json"))
    parser.add_argument(
        "--out-plot", type=Path, default=Path("results/predicted_vs_actual.png")
    )
    args = parser.parse_args()

    df = load_and_featurize(args.data)
    model, metrics, y_test, y_pred = train_and_evaluate(df)

    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out_model)
    logger.info("Saved trained model to %s", args.out_model)

    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", args.out_metrics)

    plot_predicted_vs_actual(y_test, y_pred, args.out_plot)

    logger.info(
        "Test set performance: R^2=%.3f  RMSE=%.3f  MAE=%.3f (n_test=%d)",
        metrics["r2"],
        metrics["rmse"],
        metrics["mae"],
        metrics["n_test"],
    )


if __name__ == "__main__":
    main()

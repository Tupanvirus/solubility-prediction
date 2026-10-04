"""
predict.py

Loads a trained model and predicts aqueous solubility for one or more
new molecules given as SMILES strings.

Usage:
    python src/predict.py --model results/model.joblib "CCO" "c1ccccc1" "CC(=O)OC1=CC=CC=C1C(=O)O"

    (The third example is aspirin.)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from featurize import DESCRIPTOR_COLUMNS, compute_descriptors


def predict_solubility(model, smiles_list: list[str]) -> pd.DataFrame:
    """Predict log solubility for a list of SMILES strings.

    Molecules that fail to parse are reported with a predicted value of
    NaN rather than silently skipped, so the caller always gets one row
    per input SMILES.
    """
    rows = []
    for smiles in smiles_list:
        desc = compute_descriptors(smiles)
        if desc is None:
            rows.append({"smiles": smiles, "predicted_logS": float("nan")})
            continue
        X = pd.DataFrame([desc])[DESCRIPTOR_COLUMNS]
        pred = model.predict(X)[0]
        rows.append({"smiles": smiles, "predicted_logS": round(float(pred), 3)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("results/model.joblib"))
    parser.add_argument("smiles", nargs="+", help="One or more SMILES strings")
    args = parser.parse_args()

    if not args.model.exists():
        parser.error(
            f"Model file not found: {args.model}\n"
            f"Train it first with: python src/train.py"
        )

    model = joblib.load(args.model)
    result = predict_solubility(model, args.smiles)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

"""
FasalSetu — NPK Model Training Script
Usage: python scripts/train_npk_model.py --data npk_iot_processed.csv
Outputs 6 model files to models/:
  FINAL_nitrogen_regressor.pkl
  FINAL_phosphorus_regressor.pkl
  FINAL_potassium_regressor.pkl
  FINAL_sqi_classifier.pkl       (binary: SQI >= 0.9 = Good)
  FINAL_label_encoder.pkl
  FINAL_model_metadata.json
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

REAL_FEATURES = [
    "soil_conductivity",
    "soil_humidity",
    "soil_pH",
    "soil_temperature",
    "hour",
    "day_of_year",
]

OPTIONAL_FEATURES = [
    "moisture_7d_avg",
    "temp_7d_avg",
    "temp_trend",
    "gdd_daily",
]

TARGETS = ["nitrogen", "phosphorus", "potassium"]


def load_data(csv_path: str) -> tuple[pd.DataFrame, list[str]]:
    print(f"\n[1/4] Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Shape: {df.shape}")

    # Derive time features from unix timestamp or datetime column
    if "datetime" in df.columns:
        ts = pd.to_datetime(df["datetime"], unit="s", errors="coerce")
        if ts.isna().all():
            ts = pd.to_datetime(df["datetime"], errors="coerce")
        df["hour"] = ts.dt.hour.fillna(12).astype(int)
        df["day_of_year"] = ts.dt.dayofyear.fillna(180).astype(int)

    features = [f for f in REAL_FEATURES if f in df.columns]
    for col in OPTIONAL_FEATURES:
        if col in df.columns and df[col].notna().sum() > len(df) * 0.9:
            features.append(col)
            print(f"  Optional feature included: {col}")

    required_cols = features + TARGETS + ["SQI"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    df_clean = df[required_cols].dropna()
    print(f"  Features used: {features}")
    print(f"  Clean rows: {len(df_clean)}")
    return df_clean, features


def train_regressors(df: pd.DataFrame, features: list[str]) -> dict:
    print("\n[2/4] Training NPK regressors...")
    X = df[features]
    results = {}

    for target in TARGETS:
        y = df[target]
        target_range = float(y.max() - y.min())
        target_std = float(y.std())

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=20,
            random_state=42,
        )
        model.fit(X_tr, y_tr)

        preds = model.predict(X_te)
        r2 = r2_score(y_te, preds)
        mae = mean_absolute_error(y_te, preds)
        mae_pct = (mae / target_range * 100) if target_range > 0 else float("nan")

        # Cross-val R²
        cv_r2 = cross_val_score(model, X, y, cv=5, scoring="r2")

        print(f"\n  {target.upper()}")
        print(f"    R²:             {r2:.4f}")
        print(f"    CV R² (5-fold): {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
        print(f"    MAE:            {mae:.4f} units")
        print(f"    MAE % of range: {mae_pct:.1f}%  ← primary metric")
        print(f"    Target range:   {target_range:.4f}")
        print(f"    Target std:     {target_std:.4f}")

        fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        print(f"    Top features:")
        for fname, fval in fi.head(4).items():
            print(f"      {fname:<28} {fval:.4f}")

        out_path = MODEL_DIR / f"FINAL_{target}_regressor.pkl"
        joblib.dump(model, out_path)
        print(f"    Saved → {out_path}")

        results[target] = {
            "r2": round(r2, 4),
            "cv_r2_mean": round(float(cv_r2.mean()), 4),
            "cv_r2_std": round(float(cv_r2.std()), 4),
            "mae": round(mae, 4),
            "mae_pct_range": round(mae_pct, 1),
            "target_range": round(target_range, 4),
            "top_feature": fi.index[0],
        }

    return results


def train_sqi_classifier(df: pd.DataFrame, features: list[str]) -> dict:
    """
    Binary classifier: SQI >= 0.9 → 'Good', else 'Poor'.
    More trainable than the 3-class Fuzzy cascade that failed.
    """
    print("\n[3/4] Training SQI binary classifier...")

    threshold = 0.9
    y_bin = (df["SQI"] >= threshold).astype(int)
    counts = y_bin.value_counts()
    print(f"  Class distribution: Good={counts.get(1,0)}, Poor={counts.get(0,0)}")

    X = df[features]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_bin, test_size=0.2,
                                               stratify=y_bin, random_state=42)

    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
    )
    clf.fit(X_tr, y_tr)

    preds = clf.predict(X_te)
    macro_f1 = f1_score(y_te, preds, average="macro")
    print(f"\n{classification_report(y_te, preds, target_names=['Poor', 'Good'])}")
    print(f"  Macro F1: {macro_f1:.3f}")

    fi = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
    print(f"  Top features:")
    for fname, fval in fi.head(4).items():
        print(f"    {fname:<28} {fval:.4f}")

    joblib.dump(clf, MODEL_DIR / "FINAL_sqi_classifier.pkl")
    print(f"  Saved → {MODEL_DIR / 'FINAL_sqi_classifier.pkl'}")

    le = LabelEncoder()
    le.fit(["Poor", "Good"])
    joblib.dump(le, MODEL_DIR / "FINAL_label_encoder.pkl")

    return {"macro_f1": round(macro_f1, 3), "threshold": threshold}


def save_metadata(features: list[str], regression_results: dict, sqi_results: dict):
    print("\n[4/4] Saving model metadata...")
    metadata = {
        "features": features,
        "targets": TARGETS,
        "regression": regression_results,
        "sqi_classifier": sqi_results,
        "note": (
            "R² is low by design — the NPK sensor values in this single-field "
            "IoT dataset have <0.4 unit range. MAE % of range is the meaningful metric. "
            "SQI binary classifier is the recommended downstream task."
        ),
    }
    out = MODEL_DIR / "FINAL_model_metadata.json"
    with open(out, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved → {out}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Train FasalSetu NPK models")
    parser.add_argument("--data", default=r"C:\Users\riddh\fasal-setu-28-03\soil_data_final.csv", help="Path to soil_data_final.csv")
    args = parser.parse_args()

    print("=" * 60)
    print("  FASALSETU — NPK MODEL TRAINING")
    print("=" * 60)

    df, features = load_data(args.data)
    reg_results = train_regressors(df, features)
    sqi_results = train_sqi_classifier(df, features)
    metadata = save_metadata(features, reg_results, sqi_results)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Models saved to: {MODEL_DIR}/")
    print(f"  Files created:")
    for f in sorted(MODEL_DIR.glob("FINAL_*")):
        print(f"    {f.name}")
    print(f"\n  Phosphorus MAE: {reg_results['phosphorus']['mae_pct_range']}% of range")
    print(f"  SQI classifier Macro F1: {sqi_results['macro_f1']}")
    print("\n  Next step: python scripts/ingest_schemes.py --pdf-dir data/schemes/ --verify")


if __name__ == "__main__":
    main()

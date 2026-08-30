"""Per-record SHAP explanations for both the prediction model (XGBoost) and
the anomaly model (GRU+MLP autoencoder), not just global importance plots.

Prediction model: shap.TreeExplainer is exact and cheap for gradient-boosted
trees -- no approximation, no background sampling needed.

Anomaly model: the "prediction" being explained is the model's own
reconstruction-error anomaly score, produced by a hybrid GRU (sequence) +
MLP (static) autoencoder. shap.GradientExplainer supports multi-input
PyTorch models (a list of tensors in, one tensor out), so we wrap the
autoencoder in a tiny module whose forward() *is* the scalar anomaly score,
then explain that directly against a background sample from the train
split -- this gives real per-timestep, per-channel and per-static-feature
attributions rather than a hand-rolled reconstruction-error heuristic.

Run directly: `python -m loan_intelligence.explain.shap_explain`
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn

from loan_intelligence import config as cfg
from loan_intelligence.models.anomaly import STATIC_COLS as ANOMALY_STATIC_COLS
from loan_intelligence.models.anomaly import GRUMLPAutoencoder, _prepare as _prepare_anomaly_inputs
from loan_intelligence.models.predict import CATEGORICAL_COLS, get_feature_cols, _as_xgb_frame

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"


def _native(v):
    if isinstance(v, np.floating):
        return round(float(v), 4)
    if isinstance(v, np.integer):
        return int(v)
    return v


def _top_k(names: list[str], values: np.ndarray, raw_values: list, k: int) -> list[dict]:
    order = np.argsort(-np.abs(values))[:k]
    return [
        {"feature": names[i], "shap_value": round(float(values[i]), 5), "value": raw_values[i]}
        for i in order
    ]


# ---------------------------------------------------------------------------
# Prediction model (XGBoost) -- exact TreeExplainer SHAP
# ---------------------------------------------------------------------------

class PredictionExplainer:
    def __init__(self):
        import xgboost as xgb

        self.feature_cols = joblib.load(MODEL_DIR / "xgboost_feature_cols.joblib")
        self.model = xgb.XGBClassifier()
        self.model.load_model(MODEL_DIR / "xgboost_model.json")
        self.explainer = shap.TreeExplainer(self.model)

    def explain(self, df: pd.DataFrame, top_k: int = 6) -> list[list[dict]]:
        X = _as_xgb_frame(df, self.feature_cols)
        shap_values = self.explainer(X)
        out = []
        for row_idx in range(len(df)):
            raw_values = [_native(X.iloc[row_idx][c]) for c in self.feature_cols]
            out.append(_top_k(self.feature_cols, shap_values.values[row_idx], raw_values, top_k))
        return out

    def explain_loan(self, loan_id: str, features_df: pd.DataFrame, top_k: int = 6) -> list[dict]:
        row = features_df[features_df["loan_id"] == loan_id]
        if row.empty:
            raise ValueError(f"loan_id {loan_id} not found in provided feature table")
        return self.explain(row, top_k=top_k)[0]


# ---------------------------------------------------------------------------
# Anomaly model (GRU+MLP autoencoder) -- GradientExplainer over a wrapper
# module whose forward() *is* the scalar reconstruction-error anomaly score
# ---------------------------------------------------------------------------

class _AnomalyScorer(nn.Module):
    def __init__(self, autoencoder: GRUMLPAutoencoder):
        super().__init__()
        self.ae = autoencoder

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        seq_recon, static_recon, _ = self.ae(x_seq, x_static)
        seq_err = ((seq_recon - x_seq) ** 2).mean(dim=(1, 2))
        static_err = ((static_recon - x_static) ** 2).mean(dim=1)
        return (seq_err + static_err).unsqueeze(-1)


class AnomalyExplainer:
    def __init__(self, background_size: int = 40):
        meta = json.load(open(MODEL_DIR / "anomaly_meta.json"))
        self.seq_channels = ["dti_snapshot", "income_snapshot", "payment_ratio", "missed_payment_flag", "days_past_due", "delinquency_flag"]
        self.static_cols = ANOMALY_STATIC_COLS
        self.seq_len = meta["seq_len"]

        autoencoder = GRUMLPAutoencoder(n_seq_channels=meta["n_seq_channels"], n_static=len(self.static_cols), seq_len=self.seq_len)
        autoencoder.load_state_dict(torch.load(MODEL_DIR / "anomaly_gru_mlp.pt"))
        autoencoder.eval()
        self.scorer = _AnomalyScorer(autoencoder)

        self.seq_scaler = joblib.load(MODEL_DIR / "anomaly_seq_scaler.joblib")
        self.static_scaler = joblib.load(MODEL_DIR / "anomaly_static_scaler.joblib")

        train_df = pd.read_csv(OUTPUT_DIR / "train.csv").sample(n=background_size, random_state=cfg.RANDOM_SEED)
        bg_seq, bg_static = _prepare_anomaly_inputs(train_df, self.seq_scaler, self.static_scaler)
        self.explainer = shap.GradientExplainer(self.scorer, [bg_seq, bg_static])

    def explain(self, df: pd.DataFrame, top_k: int = 6) -> list[list[dict]]:
        X_seq, X_static = _prepare_anomaly_inputs(df, self.seq_scaler, self.static_scaler)
        seq_shap, static_shap = self.explainer.shap_values([X_seq, X_static])
        # shapes: seq_shap (n, months, channels, 1), static_shap (n, n_static, 1)
        seq_shap = seq_shap[..., 0]
        static_shap = static_shap[..., 0]

        raw_seq = X_seq.numpy()
        raw_static = X_static.numpy()

        out = []
        for i in range(len(df)):
            names, values, raw_values = [], [], []
            for m in range(self.seq_len):
                for c, channel in enumerate(self.seq_channels):
                    names.append(f"{channel} (month {m + 1})")
                    values.append(seq_shap[i, m, c])
                    raw_values.append(round(float(raw_seq[i, m, c]), 4))
            for c, col in enumerate(self.static_cols):
                names.append(col)
                values.append(static_shap[i, c])
                raw_values.append(round(float(raw_static[i, c]), 4))
            out.append(_top_k(names, np.array(values), raw_values, top_k))
        return out

    def explain_loan(self, loan_id: str, features_df: pd.DataFrame, top_k: int = 6) -> list[dict]:
        row = features_df[features_df["loan_id"] == loan_id]
        if row.empty:
            raise ValueError(f"loan_id {loan_id} not found in provided feature table")
        return self.explain(row, top_k=top_k)[0]


def main() -> None:
    test_df = pd.read_csv(OUTPUT_DIR / "test.csv")

    print("building prediction-model explainer (TreeExplainer)...")
    pred_explainer = PredictionExplainer()
    sample = test_df.sort_values("default_flag", ascending=False).head(5)
    pred_explanations = dict(zip(sample["loan_id"], pred_explainer.explain(sample)))

    print("building anomaly-model explainer (GradientExplainer, background=40 train loans)...")
    anomaly_explainer = AnomalyExplainer()
    anomaly_scores = pd.read_csv(OUTPUT_DIR / "anomaly_scores_test.csv")
    top_anomalous = anomaly_scores.sort_values("anomaly_score", ascending=False).head(5)
    anomaly_sample = test_df[test_df["loan_id"].isin(top_anomalous["loan_id"])]
    anomaly_explanations = dict(zip(anomaly_sample["loan_id"], anomaly_explainer.explain(anomaly_sample)))

    print("\n=== Example prediction-model explanation ===")
    example_id = sample["loan_id"].iloc[0]
    for row in pred_explanations[example_id]:
        print(f"  {row['feature']:<45} value={row['value']!r:<12} shap={row['shap_value']:+.4f}")

    print("\n=== Example anomaly-model explanation ===")
    example_id2 = anomaly_sample["loan_id"].iloc[0]
    for row in anomaly_explanations[example_id2]:
        print(f"  {row['feature']:<30} value={row['value']:<10} shap={row['shap_value']:+.4f}")

    out_path = OUTPUT_DIR / "shap_examples.json"
    out_path.write_text(
        json.dumps({"prediction": pred_explanations, "anomaly": anomaly_explanations}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

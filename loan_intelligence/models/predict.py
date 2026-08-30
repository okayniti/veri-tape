"""Delinquency/default prediction: XGBoost baseline + Bi-LSTM on payment
sequences, benchmarked against two even-simpler baselines.

XGBoost is treated as the "production" model for the rest of the pipeline
(calibration, SHAP, scenario simulation) because TreeExplainer gives exact,
fast SHAP values and its probabilities are simple to calibrate -- a
GradientExplainer/DeepExplainer pass over the Bi-LSTM would be heavier and
slower for a live demo. The Bi-LSTM is trained and reported here purely as
the sequence-model comparison point requested by the challenge brief; it
follows the same encoder-over-sequence + MLP-head-over-statics shape as the
anomaly GRU (models/anomaly.py), just supervised on default_flag instead of
trained for reconstruction/density.

Run directly: `python -m loan_intelligence.models.predict`
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from loan_intelligence import config as cfg
from loan_intelligence.features.time_split import time_aware_split

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"

TARGET = "default_flag"
LEAKY_OR_ID_COLS = {"loan_id", "origination_date", "default_flag", "status_at_month_24"}
CATEGORICAL_COLS = ["loan_type", "region", "employment_status", "origination_channel"]
STATIC_COLS_FOR_SEQ_MODEL = [
    "credit_score_at_origination", "dti_at_origination", "interest_rate",
    "loan_amount", "loan_to_income_ratio",
]

torch.manual_seed(cfg.RANDOM_SEED)
np.random.seed(cfg.RANDOM_SEED)


# ---------------------------------------------------------------------------
# Shared data loading
# ---------------------------------------------------------------------------

def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in LEAKY_OR_ID_COLS]


def _as_xgb_frame(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in CATEGORICAL_COLS:
        X[c] = X[c].astype("category")
    return X


def load_sequences(loan_ids: np.ndarray) -> tuple[np.ndarray, list[str]]:
    data = np.load(OUTPUT_DIR / "features_sequences.npz", allow_pickle=True)
    all_ids = data["loan_id"]
    id_to_idx = {lid: i for i, lid in enumerate(all_ids)}
    idx = [id_to_idx[lid] for lid in loan_ids]
    return data["X"][idx], list(data["channels"])


# ---------------------------------------------------------------------------
# XGBoost baseline
# ---------------------------------------------------------------------------

def train_xgboost(train_df: pd.DataFrame, feature_cols: list[str]) -> xgb.XGBClassifier:
    X_train = _as_xgb_frame(train_df, feature_cols)
    y_train = train_df[TARGET]
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method="hist",
        enable_categorical=True,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=cfg.RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    return model


def predict_xgboost(model: xgb.XGBClassifier, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return model.predict_proba(_as_xgb_frame(df, feature_cols))[:, 1]


# ---------------------------------------------------------------------------
# Bi-LSTM over payment sequences (+ MLP over static origination features)
# ---------------------------------------------------------------------------

class BiLSTMDefaultPredictor(nn.Module):
    def __init__(self, n_channels: int, n_static: int, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_channels, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2 + n_static, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x_seq)
        h = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # final fwd + bwd hidden states
        combined = torch.cat([h, x_static], dim=-1)
        return self.head(combined).squeeze(-1)  # logits


def train_bilstm(
    train_df: pd.DataFrame, epochs: int = 40, batch_size: int = 128, lr: float = 1e-3,
) -> tuple[BiLSTMDefaultPredictor, StandardScaler, StandardScaler]:
    fit_df, val_df, _ = time_aware_split(train_df, test_size=0.15)

    seq_scaler = StandardScaler()
    static_scaler = StandardScaler()

    X_seq_fit, channels = load_sequences(fit_df["loan_id"].to_numpy())
    X_seq_val, _ = load_sequences(val_df["loan_id"].to_numpy())
    n_ch = X_seq_fit.shape[-1]
    seq_scaler.fit(X_seq_fit.reshape(-1, n_ch))
    X_seq_fit = seq_scaler.transform(X_seq_fit.reshape(-1, n_ch)).reshape(X_seq_fit.shape)
    X_seq_val = seq_scaler.transform(X_seq_val.reshape(-1, n_ch)).reshape(X_seq_val.shape)

    static_scaler.fit(fit_df[STATIC_COLS_FOR_SEQ_MODEL])
    X_static_fit = static_scaler.transform(fit_df[STATIC_COLS_FOR_SEQ_MODEL])
    X_static_val = static_scaler.transform(val_df[STATIC_COLS_FOR_SEQ_MODEL])

    y_fit = fit_df[TARGET].to_numpy(dtype=np.float32)
    y_val = val_df[TARGET].to_numpy(dtype=np.float32)

    model = BiLSTMDefaultPredictor(n_channels=n_ch, n_static=len(STATIC_COLS_FOR_SEQ_MODEL))
    pos_weight = torch.tensor([(y_fit == 0).sum() / max((y_fit == 1).sum(), 1)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_seq_fit_t = torch.tensor(X_seq_fit, dtype=torch.float32)
    X_static_fit_t = torch.tensor(X_static_fit, dtype=torch.float32)
    y_fit_t = torch.tensor(y_fit, dtype=torch.float32)
    X_seq_val_t = torch.tensor(X_seq_val, dtype=torch.float32)
    X_static_val_t = torch.tensor(X_static_val, dtype=torch.float32)

    n = len(y_fit_t)
    best_val_auc, best_state, patience, bad_epochs = -1.0, None, 6, 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            batch_idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            logits = model(X_seq_fit_t[batch_idx], X_static_fit_t[batch_idx])
            loss = criterion(logits, y_fit_t[batch_idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_seq_val_t, X_static_val_t)
            val_proba = torch.sigmoid(val_logits).numpy()
        val_auc = roc_auc_score(y_val, val_proba) if len(np.unique(y_val)) > 1 else 0.5

        if val_auc > best_val_auc:
            best_val_auc, best_state, bad_epochs = val_auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    print(f"  bi-lstm: best internal validation AUC {best_val_auc:.4f} (epoch stopping)")
    return model, seq_scaler, static_scaler


def predict_bilstm(
    model: BiLSTMDefaultPredictor, seq_scaler: StandardScaler, static_scaler: StandardScaler, df: pd.DataFrame,
) -> np.ndarray:
    X_seq, _ = load_sequences(df["loan_id"].to_numpy())
    n_ch = X_seq.shape[-1]
    X_seq = seq_scaler.transform(X_seq.reshape(-1, n_ch)).reshape(X_seq.shape)
    X_static = static_scaler.transform(df[STATIC_COLS_FOR_SEQ_MODEL])

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_seq, dtype=torch.float32), torch.tensor(X_static, dtype=torch.float32))
        return torch.sigmoid(logits).numpy()


# ---------------------------------------------------------------------------
# Simple baselines + metrics
# ---------------------------------------------------------------------------

def _metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "AUC": round(auc, 4),
        "PR_AUC": round(average_precision_score(y_true, y_proba), 4),
        "Precision@0.5": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall@0.5": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1@0.5": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def train_logistic_baseline(train_df: pd.DataFrame) -> tuple[LogisticRegression, StandardScaler]:
    cols = ["credit_score_at_origination", "dti_at_origination", "interest_rate", "loan_amount", "loan_to_income_ratio", "term_months"]
    scaler = StandardScaler().fit(train_df[cols])
    X = scaler.transform(train_df[cols])
    model = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X, train_df[TARGET])
    return model, scaler


def main() -> None:
    train_df = pd.read_csv(OUTPUT_DIR / "train.csv")
    test_df = pd.read_csv(OUTPUT_DIR / "test.csv")
    feature_cols = _feature_cols(train_df)

    y_train, y_test = train_df[TARGET].to_numpy(), test_df[TARGET].to_numpy()

    results = {}

    majority_proba = np.full(len(y_test), y_train.mean())
    results["Majority-class baseline"] = _metrics(y_test, majority_proba)

    print("training logistic regression baseline (origination fields only)...")
    logit_model, logit_scaler = train_logistic_baseline(train_df)
    logit_cols = ["credit_score_at_origination", "dti_at_origination", "interest_rate", "loan_amount", "loan_to_income_ratio", "term_months"]
    logit_proba = logit_model.predict_proba(logit_scaler.transform(test_df[logit_cols]))[:, 1]
    results["Logistic regression (origination-only)"] = _metrics(y_test, logit_proba)

    print("training XGBoost...")
    xgb_model = train_xgboost(train_df, feature_cols)
    xgb_proba_test = predict_xgboost(xgb_model, test_df, feature_cols)
    xgb_proba_train = predict_xgboost(xgb_model, train_df, feature_cols)
    results["XGBoost"] = _metrics(y_test, xgb_proba_test)

    print("training Bi-LSTM on payment sequences...")
    bilstm_model, seq_scaler, static_scaler = train_bilstm(train_df)
    bilstm_proba_test = predict_bilstm(bilstm_model, seq_scaler, static_scaler, test_df)
    bilstm_proba_train = predict_bilstm(bilstm_model, seq_scaler, static_scaler, train_df)
    results["Bi-LSTM (payment sequences)"] = _metrics(y_test, bilstm_proba_test)

    print("\n=== Model comparison (held-out, time-aware test set) ===")
    comparison = pd.DataFrame(results).T
    print(comparison.to_string())

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    xgb_model.save_model(MODEL_DIR / "xgboost_model.json")
    torch.save(bilstm_model.state_dict(), MODEL_DIR / "bilstm_model.pt")
    json.dump(
        {"n_channels": len(STATIC_COLS_FOR_SEQ_MODEL)},
        open(MODEL_DIR / "bilstm_meta.json", "w"),
    )
    import joblib
    joblib.dump(seq_scaler, MODEL_DIR / "bilstm_seq_scaler.joblib")
    joblib.dump(static_scaler, MODEL_DIR / "bilstm_static_scaler.joblib")
    joblib.dump(feature_cols, MODEL_DIR / "xgboost_feature_cols.joblib")

    predictions_train = train_df[["loan_id", TARGET]].copy()
    predictions_train["xgb_proba"] = xgb_proba_train
    predictions_train["bilstm_proba"] = bilstm_proba_train
    predictions_test = test_df[["loan_id", TARGET]].copy()
    predictions_test["xgb_proba"] = xgb_proba_test
    predictions_test["bilstm_proba"] = bilstm_proba_test

    predictions_train.to_csv(OUTPUT_DIR / "predictions_train.csv", index=False)
    predictions_test.to_csv(OUTPUT_DIR / "predictions_test.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv")

    print(f"\nwrote models to {MODEL_DIR}")
    print(f"wrote predictions_train.csv, predictions_test.csv, model_comparison.csv to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

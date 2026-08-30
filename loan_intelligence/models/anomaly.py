"""Structural anomaly detection: GRU sequence encoder + MLP static encoder,
trained as an autoencoder, scored by reconstruction error.

This is deliberately *not* "loans with a high default probability" -- that's
what models/predict.py already does. An anomaly here is a record that is
structurally unusual relative to its own history and peers (a sudden DTI
spike, a payment-pattern break) whether or not it ever defaults; the
generator injects these independently of default_flag (see
data/generate_synthetic.py) precisely so this distinction is measurable.

Architecture follows the same encoder-over-sequence + MLP-head-over-statics
shape used for "The Watchtower" behavioral-log anomaly detector, adapted
from a classifier head to an autoencoder: a GRU encodes the 12-month
payment sequence, an MLP encodes static/engineered features, both are
compressed into a shared latent vector, and two decoders reconstruct each
input. Loans the model reconstructs poorly are the structurally unusual
ones. Trained unsupervised (no anomaly labels used); outputs/ground_truth.csv
is used only afterwards, to score precision @ alert-budget and lift over a
random baseline -- the same evaluation methodology used for Watchtower,
though the numbers themselves are specific to this synthetic loan tape, not
a repeat of that project's results.

Run directly: `python -m loan_intelligence.models.anomaly`
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from loan_intelligence import config as cfg
from loan_intelligence.models.predict import load_sequences

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"

STATIC_COLS = [
    "credit_score_at_origination", "dti_at_origination", "interest_rate",
    "loan_amount", "loan_to_income_ratio",
    "dti_trend_12m", "dti_volatility_12m", "dti_change_pct_12m",
    "payment_volatility_12m", "avg_payment_ratio_12m",
    "n_missed_payments_12m", "longest_missed_streak_12m",
    "balance_paydown_ratio_12m", "income_trend_12m",
    "dti_at_origination_peer_zscore", "credit_score_at_origination_peer_zscore",
]

torch.manual_seed(cfg.RANDOM_SEED)
np.random.seed(cfg.RANDOM_SEED)


class GRUMLPAutoencoder(nn.Module):
    def __init__(self, n_seq_channels: int, n_static: int, seq_len: int, hidden_size: int = 24, latent_size: int = 16):
        super().__init__()
        self.seq_len = seq_len
        self.seq_encoder = nn.GRU(n_seq_channels, hidden_size, batch_first=True)
        self.static_encoder = nn.Sequential(nn.Linear(n_static, 32), nn.ReLU(), nn.Linear(32, 16))
        self.to_latent = nn.Linear(hidden_size + 16, latent_size)

        self.static_decoder = nn.Sequential(nn.Linear(latent_size, 32), nn.ReLU(), nn.Linear(32, n_static))
        self.seq_decoder = nn.GRU(latent_size, hidden_size, batch_first=True)
        self.seq_output = nn.Linear(hidden_size, n_seq_channels)

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor):
        _, h_seq = self.seq_encoder(x_seq)
        h_seq = h_seq.squeeze(0)
        h_static = self.static_encoder(x_static)
        z = self.to_latent(torch.cat([h_seq, h_static], dim=-1))

        static_recon = self.static_decoder(z)
        z_rep = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        seq_hidden, _ = self.seq_decoder(z_rep)
        seq_recon = self.seq_output(seq_hidden)

        return seq_recon, static_recon, z


def _prepare(df: pd.DataFrame, seq_scaler: StandardScaler, static_scaler: StandardScaler) -> tuple[torch.Tensor, torch.Tensor]:
    X_seq, _ = load_sequences(df["loan_id"].to_numpy())
    n_ch = X_seq.shape[-1]
    X_seq = seq_scaler.transform(X_seq.reshape(-1, n_ch)).reshape(X_seq.shape)
    X_static = static_scaler.transform(df[STATIC_COLS])
    return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(X_static, dtype=torch.float32)


def train_autoencoder(
    train_df: pd.DataFrame, epochs: int = 60, batch_size: int = 128, lr: float = 1e-3,
) -> tuple[GRUMLPAutoencoder, StandardScaler, StandardScaler]:
    X_seq_raw, channels = load_sequences(train_df["loan_id"].to_numpy())
    n_ch = X_seq_raw.shape[-1]
    seq_len = X_seq_raw.shape[1]

    seq_scaler = StandardScaler().fit(X_seq_raw.reshape(-1, n_ch))
    static_scaler = StandardScaler().fit(train_df[STATIC_COLS])

    X_seq, X_static = _prepare(train_df, seq_scaler, static_scaler)

    model = GRUMLPAutoencoder(n_seq_channels=n_ch, n_static=len(STATIC_COLS), seq_len=seq_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    n = len(train_df)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            seq_recon, static_recon, _ = model(X_seq[idx], X_static[idx])
            loss = criterion(seq_recon, X_seq[idx]) + criterion(static_recon, X_static[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1}/{epochs}  reconstruction loss {total_loss / n:.4f}")

    return model, seq_scaler, static_scaler


def score_anomalies(
    model: GRUMLPAutoencoder, seq_scaler: StandardScaler, static_scaler: StandardScaler, df: pd.DataFrame,
) -> np.ndarray:
    X_seq, X_static = _prepare(df, seq_scaler, static_scaler)
    model.eval()
    with torch.no_grad():
        seq_recon, static_recon, _ = model(X_seq, X_static)
        seq_err = ((seq_recon - X_seq) ** 2).mean(dim=(1, 2))
        static_err = ((static_recon - X_static) ** 2).mean(dim=1)
    return (seq_err + static_err).numpy()


def evaluate_at_budget(scores: np.ndarray, is_anomalous: np.ndarray, budget: float) -> dict:
    n = len(scores)
    k = max(1, round(n * budget))
    flagged = np.argsort(scores)[::-1][:k]
    precision = is_anomalous[flagged].mean()
    base_rate = is_anomalous.mean()
    lift = precision / base_rate if base_rate > 0 else float("nan")
    return {"alert_budget": budget, "n_flagged": k, "precision": round(float(precision), 4),
            "base_rate": round(float(base_rate), 4), "lift": round(float(lift), 2)}


def main() -> None:
    train_df = pd.read_csv(OUTPUT_DIR / "train.csv")
    test_df = pd.read_csv(OUTPUT_DIR / "test.csv")
    ground_truth = pd.read_csv(OUTPUT_DIR / "ground_truth.csv")

    print("training GRU+MLP autoencoder (unsupervised, no anomaly labels used)...")
    model, seq_scaler, static_scaler = train_autoencoder(train_df)

    test_scores = score_anomalies(model, seq_scaler, static_scaler, test_df)
    test_eval = test_df[["loan_id"]].merge(ground_truth, on="loan_id", how="left")
    is_anomalous = test_eval["is_anomalous"].to_numpy()

    print("\n=== Anomaly detector evaluation (held-out test set) ===")
    for budget in (0.01, 0.025, 0.05):
        result = evaluate_at_budget(test_scores, is_anomalous, budget)
        print(
            f"  alert budget {budget:>5.1%}: flagged {result['n_flagged']:>3} loans, "
            f"precision {result['precision']:.1%}, base rate {result['base_rate']:.1%}, "
            f"lift {result['lift']}x"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_DIR / "anomaly_gru_mlp.pt")
    joblib.dump(seq_scaler, MODEL_DIR / "anomaly_seq_scaler.joblib")
    joblib.dump(static_scaler, MODEL_DIR / "anomaly_static_scaler.joblib")
    json.dump(
        {"static_cols": STATIC_COLS, "n_seq_channels": 6, "seq_len": cfg.FEATURE_WINDOW_MONTHS},
        open(MODEL_DIR / "anomaly_meta.json", "w"),
        indent=2,
    )

    alert_threshold = np.quantile(test_scores, 0.99)
    out = test_eval.copy()
    out["anomaly_score"] = test_scores
    out["flagged_top_1pct"] = test_scores >= alert_threshold
    out.to_csv(OUTPUT_DIR / "anomaly_scores_test.csv", index=False)

    print(f"\nwrote model to {MODEL_DIR}")
    print(f"wrote {OUTPUT_DIR / 'anomaly_scores_test.csv'}")


if __name__ == "__main__":
    main()

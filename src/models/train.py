from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ingestion.sensor_simulator import SensorSimulator
from models.features import engineer_features
from models.lstm_model import FailureLSTM

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_PATH = MODEL_DIR / "lstm_baseline.pt"

SEQ_LEN = 24
EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-3
TRAIN_SPLIT = 0.8


def _feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"timestamp", "equipment_id", "failure"}
    return [c for c in df.columns if c not in exclude]


def prepare_data(equipment_id: str = "PUMP-001", n_hours: int = 720) -> pd.DataFrame:
    raw = SensorSimulator(equipment_id).generate(n_hours=n_hours)
    df = engineer_features(raw)
    threshold = df["vibration"].quantile(0.85)
    df["failure"] = (df["vibration"] >= threshold).astype(np.float32)
    return df


def build_sequences(
    df: pd.DataFrame, feature_cols: list[str], seq_len: int = SEQ_LEN
) -> tuple[np.ndarray, np.ndarray]:
    features = df[feature_cols].values.astype(np.float32)
    labels = df["failure"].values.astype(np.float32)

    sequences, targets = [], []
    for i in range(seq_len, len(df)):
        sequences.append(features[i - seq_len : i])
        targets.append(labels[i])

    return np.array(sequences), np.array(targets).reshape(-1, 1)


def train(
    equipment_id: str = "PUMP-001",
    n_hours: int = 720,
    epochs: int = EPOCHS,
    model_path: Path = MODEL_PATH,
) -> float:
    df = prepare_data(equipment_id, n_hours)
    feature_cols = _feature_columns(df)
    x, y = build_sequences(df, feature_cols)

    split_idx = int(len(x) * TRAIN_SPLIT)
    x_train, x_val = x[:split_idx], x[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_x = torch.from_numpy(x_val)
    val_y = torch.from_numpy(y_val)

    model = FailureLSTM(input_size=len(feature_cols))
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_preds = model(val_x)
            val_loss = criterion(val_preds, val_y).item()

        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch}/{epochs} "
                f"train_loss={epoch_loss / len(train_loader):.4f} val_loss={val_loss:.4f}"
            )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": len(feature_cols),
            "feature_cols": feature_cols,
            "seq_len": SEQ_LEN,
        },
        model_path,
    )
    print(f"Saved baseline model to {model_path}")
    return val_loss


if __name__ == "__main__":
    train()

"""Train a small PyTorch neural network on the shared blizzard sensor dataset."""

from pathlib import Path
import argparse
import copy
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "blizzard_mlp.pt"
TARGET = "blizzard_today"
SEED = 42
THRESHOLD = 0.5


class BlizzardMLP(nn.Module):
    """A multilayer perceptron that returns one blizzard logit per day."""

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(1)


def predict_from_csv(input_path: Path, output_path: Path | None = None) -> None:
    """Load the exported network and scaler, then predict engineered feature rows."""
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    feature_names = checkpoint["feature_names"]
    inputs = pd.read_csv(input_path)
    missing = [name for name in feature_names if name not in inputs.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required feature columns: {', '.join(missing)}")

    values = inputs[feature_names].to_numpy(dtype=np.float32)
    mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
    scaled = (values - mean) / scale
    model = BlizzardMLP(input_size=len(feature_names))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(torch.from_numpy(scaled))).numpy()
    results = pd.DataFrame({
        "blizzard_probability": probabilities,
        "blizzard_prediction": (probabilities >= checkpoint["threshold"]).astype(int),
    })
    destination = output_path or input_path.with_name(f"{input_path.stem}_deep_learning_predictions.csv")
    results.to_csv(destination, index=False)
    for row_number, probability in enumerate(probabilities, start=1):
        print(f"Row {row_number}: blizzard probability={probability:.3f}, prediction={int(probability >= checkpoint['threshold'])}")
    print(f"Saved predictions to: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or use the mountain blizzard PyTorch model.")
    parser.add_argument("--predict-csv", type=Path, help="Predict from a CSV of engineered feature rows.")
    parser.add_argument("--output", type=Path, help="Optional predictions CSV path (inference only).")
    args = parser.parse_args()

    if args.predict_csv:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Run this script without --predict-csv first.")
        predict_from_csv(args.predict_csv, args.output)
        return

    # Small tabular batches are faster and more predictable without large CPU thread pools.
    torch.set_num_threads(1)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_path = DATA_DIR / "train_set.csv"
    test_path = DATA_DIR / "test_set.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Shared train/test data not found. Run xgboost/sensor_xgboost.py once to generate data/ first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    feature_names = [column for column in train_df.columns if column not in {"date", TARGET}]
    if [column for column in test_df.columns if column not in {"date", TARGET}] != feature_names:
        raise ValueError("Train and test feature columns do not match.")

    X_train_all = train_df[feature_names].to_numpy(dtype=np.float32)
    y_train_all = train_df[TARGET].to_numpy(dtype=np.float32)
    X_test = test_df[feature_names].to_numpy(dtype=np.float32)
    y_test = test_df[TARGET].to_numpy(dtype=np.int64)

    # Use the latest 15% of training rows for early stopping, preserving time order.
    validation_start = int(len(X_train_all) * 0.85)
    X_fit, X_val = X_train_all[:validation_start], X_train_all[validation_start:]
    y_fit, y_val = y_train_all[:validation_start], y_train_all[validation_start:]

    # Neural networks benefit from comparable input scales. Fit only on training rows.
    scaler = StandardScaler()
    X_fit = scaler.fit_transform(X_fit).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    dataset = TensorDataset(torch.from_numpy(X_fit), torch.from_numpy(y_fit))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = BlizzardMLP(input_size=len(feature_names))
    positive_count = max(1, int(y_fit.sum()))
    negative_count = max(1, len(y_fit) - positive_count)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative_count / positive_count))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    val_features = torch.from_numpy(X_val)
    val_targets = torch.from_numpy(y_val)

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    patience = 20
    epochs_without_improvement = 0
    max_epochs = 200
    for epoch in range(max_epochs):
        model.train()
        for batch_features, batch_targets in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_features), batch_targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_features), val_targets).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(torch.from_numpy(X_test))).numpy()
    predictions = (probabilities >= 0.5).astype(np.int64)

    print("Model: PyTorch MLP | input features:", len(feature_names))
    print(f"Rows: train={len(X_train_all)} (fit={len(X_fit)}, validation={len(X_val)}), test={len(X_test)}")
    print(f"Stopped after {epoch + 1} epoch(s); best validation loss: {best_val_loss:.4f}")
    print("\nTest classification report (positive class = blizzard):")
    print(classification_report(y_test, predictions, target_names=["no blizzard", "blizzard"], zero_division=0))
    print("Confusion matrix [[true no, false alarm], [miss, detected]]:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    if np.unique(y_test).size == 2:
        print(f"ROC AUC: {roc_auc_score(y_test, probabilities):.3f}")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "feature_names": feature_names,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "threshold": THRESHOLD,
    }, MODEL_PATH)
    print(f"Threshold: {THRESHOLD} (adjust for the cost of misses versus false alarms in real use)")
    print(f"Exported model and preprocessing settings: {MODEL_PATH}")


if __name__ == "__main__":
    main()

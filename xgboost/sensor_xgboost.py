"""Teach XGBoost classification with synthetic high-mountain weather sensors."""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "blizzard_xgb.json"
RANDOM_SEED = 42
THRESHOLD = 0.5


def make_sensor_data(days: int = 365) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create hourly observations and one daily target: blizzard conditions later today."""
    rng = np.random.default_rng(RANDOM_SEED)
    timestamps = pd.date_range("2025-01-01", periods=days * 24, freq="h")
    hour = timestamps.hour.to_numpy()
    day = timestamps.dayofyear.to_numpy()
    daily_wave = np.sin(2 * np.pi * (hour - 14) / 24)
    seasonal_wave = np.cos(2 * np.pi * (day - 15) / 365)

    # Latent storm setup varies by day. Some days develop into a sustained blizzard.
    storm_setup = rng.normal(0, 1, days)
    storm_days = (storm_setup + 0.7 * seasonal_wave.reshape(days, 24)[:, 0] > 0.85)
    storm_strength = np.where(storm_days, rng.uniform(0.8, 1.8, days), 0.0)
    storm = np.repeat(storm_strength, 24)

    temperature = -7 + 8 * seasonal_wave + 2 * daily_wave + rng.normal(0, 1.8, len(timestamps)) - 2.0 * storm
    pressure = 720 + 3 * seasonal_wave + rng.normal(0, 1.8, len(timestamps)) - 9.0 * storm
    humidity = np.clip(48 + 12 * daily_wave + rng.normal(0, 6, len(timestamps)) + 32 * storm, 10, 100)
    wind_speed = np.clip(5 + 2 * daily_wave + rng.normal(0, 2.2, len(timestamps)) + 12.0 * storm, 0, None)
    wind_gust = wind_speed + rng.gamma(2.0, 1.5, len(timestamps)) + 5.0 * storm
    snowfall = np.clip(rng.gamma(0.7, 0.35, len(timestamps)) + 2.2 * storm, 0, None)
    visibility = np.clip(12000 - 180 * humidity - 260 * snowfall - 250 * wind_speed + rng.normal(0, 1000, len(timestamps)), 50, 20000)

    hourly = pd.DataFrame({
        "timestamp": timestamps,
        "temperature_c": temperature,
        "pressure_hpa": pressure,
        "humidity_pct": humidity,
        "wind_speed_m_s": wind_speed,
        "wind_gust_m_s": wind_gust,
        "snowfall_mm_h": snowfall,
        "visibility_m": visibility,
    })
    # Operationally meaningful teaching label: a 3-hour run with strong wind and
    # visibility below 400 m between 06:00 and 23:00 local time.
    event = ((hourly["wind_speed_m_s"] >= 17.2) & (hourly["visibility_m"] < 400)).to_numpy()
    daily_labels = []
    for d in range(days):
        start = d * 24 + 6
        end = d * 24 + 24
        day_event = event[start:end]
        sustained = any(day_event[i : i + 3].all() for i in range(max(0, len(day_event) - 2)))
        daily_labels.append(int(sustained))
    labels = pd.DataFrame({"date": pd.date_range(timestamps[0].normalize(), periods=days, freq="D"), "blizzard_today": daily_labels})
    return hourly, labels


def build_morning_examples(hourly: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Use information available at 06:00 to predict conditions later the same day."""
    sensors = ["temperature_c", "pressure_hpa", "humidity_pct", "wind_speed_m_s", "wind_gust_m_s", "snowfall_mm_h", "visibility_m"]
    rows = []
    for d, date in enumerate(labels["date"]):
        start = d * 24
        past = hourly.iloc[start : start + 7]  # 00:00 through 06:00 only
        current = past.iloc[-1]
        row = {"date": date}
        for name in sensors:
            row[f"current_{name}"] = current[name]
            row[f"change_6h_{name}"] = current[name] - past.iloc[0][name]
        date_value = pd.Timestamp(date)
        day_of_year = date_value.dayofyear
        row["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365)
        row["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365)
        row["month"] = date_value.month
        row["blizzard_today"] = labels.iloc[d]["blizzard_today"]
        rows.append(row)
    return pd.DataFrame(rows)


def predict_from_csv(input_path: Path, output_path: Path | None = None) -> None:
    """Load the exported classifier and predict from one or more engineered rows."""
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    feature_names = list(model.feature_names_in_)
    inputs = pd.read_csv(input_path)
    missing = [name for name in feature_names if name not in inputs.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required feature columns: {', '.join(missing)}")
    probabilities = model.predict_proba(inputs[feature_names])[:, 1]
    results = pd.DataFrame({
        "blizzard_probability": probabilities,
        "blizzard_prediction": (probabilities >= THRESHOLD).astype(int),
    })
    destination = output_path or input_path.with_name(f"{input_path.stem}_xgboost_predictions.csv")
    results.to_csv(destination, index=False)
    for row_number, probability in enumerate(probabilities, start=1):
        print(f"Row {row_number}: blizzard probability={probability:.3f}, prediction={int(probability >= THRESHOLD)}")
    print(f"Saved predictions to: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or use the mountain blizzard XGBoost model.")
    parser.add_argument("--predict-csv", type=Path, help="Predict from a CSV of engineered feature rows.")
    parser.add_argument("--output", type=Path, help="Optional predictions CSV path (inference only).")
    args = parser.parse_args()

    if args.predict_csv:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Run this script without --predict-csv first.")
        predict_from_csv(args.predict_csv, args.output)
        return

    DATA_DIR.mkdir(exist_ok=True)
    hourly, labels = make_sensor_data()
    hourly.to_csv(DATA_DIR / "mountain_sensor_readings.csv", index=False)
    examples = build_morning_examples(hourly, labels)
    examples.to_csv(DATA_DIR / "daily_blizzard_examples.csv", index=False)
    examples.iloc[[-1]].drop(columns=["date", "blizzard_today"]).to_csv(
        DATA_DIR / "inference_example.csv", index=False
    )

    # Save a chronological split so both model workflows consume identical rows.
    target = "blizzard_today"
    split_at = int(len(examples) * 0.8)
    train_set = examples.iloc[:split_at].reset_index(drop=True)
    test_set = examples.iloc[split_at:].reset_index(drop=True)
    train_set.to_csv(DATA_DIR / "train_set.csv", index=False)
    test_set.to_csv(DATA_DIR / "test_set.csv", index=False)

    # Train and evaluate from the persisted files shared with deep_learning/.
    train_set = pd.read_csv(DATA_DIR / "train_set.csv")
    test_set = pd.read_csv(DATA_DIR / "test_set.csv")
    X_train = train_set.drop(columns=["date", target])
    y_train = train_set[target].astype(int)
    X_test = test_set.drop(columns=["date", target])
    y_test = test_set[target].astype(int)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        scale_pos_weight=max(1.0, (y_train == 0).sum() / max(1, (y_train == 1).sum())),
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print(f"Days: {len(examples)} | train: {len(X_train)} | test: {len(X_test)}")
    print(f"Training blizzard days: {int(y_train.sum())} / {len(y_train)}")
    print(f"Test blizzard days: {int(y_test.sum())} / {len(y_test)}")
    print("\nTest classification report (positive class = blizzard):")
    print(classification_report(y_test, predictions, target_names=["no blizzard", "blizzard"], zero_division=0))
    print("Confusion matrix [[true no, false alarm], [miss, detected]]:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    if y_test.nunique() == 2:
        print(f"ROC AUC: {roc_auc_score(y_test, probabilities):.3f}")
    print("\nTop engineered features:")
    importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print(importance.head(10).to_string())
    print(f"\nCreated: {DATA_DIR}")
    print(f"Exported model: {MODEL_PATH}")
    print("  mountain_sensor_readings.csv — hourly synthetic readings")
    print("  daily_blizzard_examples.csv  — 06:00 engineered features and same-day label")
    print("  train_set.csv                — first 80% of days used by model.fit()")
    print("  test_set.csv                 — last 20% of days used for evaluation")
    print("  inference_example.csv        — one synthetic feature row for prediction")


if __name__ == "__main__":
    main()

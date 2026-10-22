# XGBoost workflow

This folder contains the XGBoost binary classifier for the shared mountain blizzard problem. It predicts from 06:00 readings whether a sustained three-hour period with wind speed at least 17.2 m/s and visibility below 400 m will occur later that day.

## Run it

From the repository root in PowerShell:

```powershell
python -m venv .venv-xgboost
.\.venv-xgboost\Scripts\Activate.ps1
python -m pip install -r xgboost\requirements.txt
python xgboost\sensor_xgboost.py
```

The script generates the deterministic synthetic data under the shared root `data/` folder, saves a chronological 80/20 train/test split, and trains/evaluates using the saved CSVs. The same `train_set.csv` and `test_set.csv` are used by the deep-learning workflow.

The trained model is exported to `xgboost/models/blizzard_xgb.json`.

## Predict from a new forecast row

Prepare a CSV with one row containing the same engineered feature columns as `data/inference_example.csv` (no `date` or `blizzard_today` column). Then run:

```powershell
python xgboost\sensor_xgboost.py --predict-csv path\to\your_features.csv
```

The script loads the saved JSON model and writes a sibling `*_xgboost_predictions.csv` containing the blizzard probability and the 0/1 prediction. `data/inference_example.csv` is a synthetic example row. The input must already contain the 06:00 readings, changes since midnight, and calendar features; raw hourly readings are not automatically aggregated by this command.

## Features and model

Each sensor contributes its current 06:00 reading and change since 00:00. Day-of-year sine and cosine values encode seasonal position. XGBoost builds an ensemble of decision trees for binary classification and reports precision, recall, F1, confusion matrix, and ROC AUC. The demo uses a 0.5 probability threshold.

All readings and labels are synthetic. The reported scores demonstrate model training and evaluation only; they are not real forecast performance or a safety alert.

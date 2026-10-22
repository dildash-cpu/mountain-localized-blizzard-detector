# Mountain Localized Blizzard Detector

This proof of concept compares two machine-learning approaches for predicting same-day blizzard conditions from mountain sensor data. The input rows and labels are synthetic, intended for learning and workflow comparison only.

## Project folders

- [`xgboost/`](xgboost/README.md) — current XGBoost tree-classification workflow and its dependencies.
- [`deep_learning/`](deep_learning/README.md) — a PyTorch multilayer-perceptron solution for the same target and shared data.
- [`data/`](data/) — shared hourly source data, engineered daily examples, and chronological train/test CSVs.

After training, the workflows export reusable model files to `xgboost/models/blizzard_xgb.json` and `deep_learning/models/blizzard_mlp.pt`. Both scripts accept `--predict-csv` to predict from a CSV with one or more already engineered forecast rows. See each workflow's README for details.

## Shared problem and data

At 06:00 local time, the model uses sensor readings and their changes since midnight to classify whether a sustained three-hour period with wind speed at least 17.2 m/s and visibility below 400 m will occur later that day. Both workflows use `data/train_set.csv` and `data/test_set.csv`, with `blizzard_today` as the target. The split is chronological: the earliest 80% of days for training and latest 20% for final testing.

To regenerate the deterministic synthetic data and run XGBoost from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r xgboost\requirements.txt
python xgboost\sensor_xgboost.py
```

To run the deep-learning comparison in its own environment, see [`deep_learning/README.md`](deep_learning/README.md). The workflow uses the same saved train and test CSVs and applies feature standardization learned from training rows only.

## Run predictions with the exported models

After training has created the model files, run these from the repository root to predict from the included synthetic example row:

```powershell
.\.venv\Scripts\python.exe xgboost\sensor_xgboost.py --predict-csv data\inference_example.csv
.\.venv-deep-learning\Scripts\python.exe deep_learning\sensor_mlp.py --predict-csv data\inference_example.csv
```

These commands run inference; they do not train the models. XGBoost writes `data/inference_example_xgboost_predictions.csv`, and deep learning writes `data/inference_example_deep_learning_predictions.csv`. Each file contains a probability and its thresholded 0/1 prediction. Replace the example with a CSV containing your engineered forecast row(s) to predict from your sensor data.

## Limitations

The data and target labels are synthetic and simplified. Metrics are demonstrations of the modeling workflows, not real forecasts or safety alerts. A real system needs calibrated instruments, locally applicable warning definitions, verified observations, validation over multiple seasons and years, and operational review of alert thresholds.

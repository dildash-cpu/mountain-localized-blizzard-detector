# Deep learning workflow

This folder contains a PyTorch multilayer perceptron (MLP) for the same mountain blizzard classification problem and data as the XGBoost workflow.

## Run it

From the repository root in PowerShell, create a separate environment and install this workflow's dependencies:

```powershell
python -m venv .venv-deep-learning
.\.venv-deep-learning\Scripts\Activate.ps1
python -m pip install -r deep_learning\requirements.txt
python deep_learning\sensor_mlp.py
```

If `data/train_set.csv` and `data/test_set.csv` do not exist yet, first run `python xgboost\sensor_xgboost.py` from the repository root. Both models then use those same saved rows and `blizzard_today` labels.

The trained network and the matching feature names, scaler values, and threshold are exported together to `deep_learning/models/blizzard_mlp.pt`.

## Predict from a new forecast row

Prepare a CSV with one or more rows containing the same engineered feature columns as `data/inference_example.csv` (no `date` or `blizzard_today` column). Then run:

```powershell
python deep_learning\sensor_mlp.py --predict-csv path\to\your_features.csv
```

The script restores the saved network and applies the saved training scaler. It writes a sibling `*_deep_learning_predictions.csv` containing the probability and 0/1 prediction. `data/inference_example.csv` is synthetic sample input. The command expects already engineered 06:00 values, six-hour changes, and calendar features; it does not convert raw hourly sensor rows into a forecast row.

## What the MLP does

The network takes the 06:00 engineered sensor features and passes them through two hidden layers (64 and 32 units) with ReLU activations and dropout, then outputs one score (logit) per day. A sigmoid converts scores to probabilities; the example uses 0.5 as its classification threshold.

Unlike tree models, neural networks commonly benefit from features on comparable scales. `StandardScaler` learns each feature's mean and standard deviation from the fitting portion of the training set only. The same transform is applied to the validation and test rows, preventing test information from leaking into preprocessing. The most recent 15% of the training period is used for early stopping; the final test period remains untouched until evaluation.

The positive-class weighting helps with the less frequent blizzard class. The report includes precision, recall, F1, confusion matrix, and ROC AUC. These are synthetic data and simplified labels; scores are a learning demonstration, not operational forecast performance or a safety alert.

# Mountain Localized Blizzard Detector

Proof-of-concept (POC) for predicting blizzards in **high mountainous regions** using **localized** data from a network of on-site sensors (temperature, pressure, humidity, wind, gusts, snowfall, visibility). The goal is same-day risk assessment from conditions at the station—not a replacement for national weather services, but a complement where mountain microclimates differ sharply from valley forecasts.

**Data status:** All inputs and labels in this repo are **synthetic** today. The pipeline (ingestion → feature engineering → XGBoost classification → evaluation) is structured so real sensor streams and region-specific blizzard definitions can be swapped in later.

## Problem setup

At **06:00** local time, the model uses the current readings and recent history (change since midnight) to classify whether blizzard conditions will occur **later the same day**: a sustained three-hour period with wind speed at least 17.2 m/s and visibility below 400 m between 06:00 and midnight. Real warning criteria vary by region; production use requires definitions and verified observations from the relevant local meteorological authority.

## Run it (PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
python sensor_xgboost.py
```

Dependencies are listed in `requirements.txt`. The script generates:

- `data/mountain_sensor_readings.csv` — hourly simulated sensor measurements.
- `data/daily_blizzard_examples.csv` — engineered 06:00 feature rows and their same-day labels.

## Feature engineering

Each daily example includes each sensor’s **06:00 reading** and its **change since 00:00**, plus sine/cosine encoding of day of year for seasonality. All input features are known by 06:00; later readings are used only to build the label, avoiding leakage. Additional aggregates (means, min/max over a window) can be added if validation shows they help.

## What XGBoost is doing

Binary classification: XGBoost fits an ensemble of decision trees; the model outputs a probability, and the demo treats ≥ 0.5 as blizzard. Training uses the earliest 80% of days chronologically and tests on the latest 20%. Metrics include precision, recall, F1, and a confusion matrix. For severe weather, misses and false alarms have different costs—operational thresholds should be set with domain experts, not fixed at 0.5.

## Limitations (POC)

Synthetic data and simplified labels mean printed scores are a **demonstration only**, not an operational forecast or safety alert. A deployable **Mountain Localized Blizzard Detector** would need calibrated instruments, multi-season validation, official or expert-reviewed labels for the target terrain, and human review of alerts.

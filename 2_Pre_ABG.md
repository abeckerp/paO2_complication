# Continuous $paO_2$ Prediction and Postoperative Complications in Neurosurgical Patients
## $PaO_2$ Prediction before first ABG

Andrea S. Gutmann
2026-02-02

## Loading required libraries and data

``` python
# ======================
# Standard library
# ======================
import pickle
import pprint
from pathlib import Path

# ======================
# Third-party libraries
# ======================
import numpy as np
import pandas as pd
import yaml
from sklearn import metrics

# ======================
# Local / application
# ======================

pp = pprint.PrettyPrinter(indent=4)
np.random.seed(42)

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# for prediction

with open(Path(config.get('pickle').get('scaler_y')), 'rb') as ys:
    y_scaler = pickle.load(ys)

with open(Path(config.get('pickle').get('scaler_x')), 'rb') as xs:
    x_scaler = pickle.load(xs)

with open(Path(config.get("pickle").get("estimator22")), "rb") as e:
    fitted_estimator = pickle.load(e)

with open(config.get("pickle").get("selected_features"), "rb") as sf:
    selected_features = [str(s) for s in pickle.load(sf)]

with open("data/out/data_ready.pickle", 'rb') as d:
    data = pickle.load(d)

pre_df = data.loc[
    data.last_horowitz.isnull(),
    selected_features + ["last_horowitz", "identifier", "paO2_measured"]
]

prediction_pre_df = pre_df.loc[pre_df.paO2_measured.isnull(),:]
x_data_prediction_pre = x_scaler.transform(prediction_pre_df.drop(["last_horowitz", "identifier", "paO2_measured"], axis = 1)) # no last horowitz

performance_pre_df = pre_df.loc[pre_df.paO2_measured.notnull(),:]
```

    /Users/abeckerp/Documents/pao2-complications-project/.venv/lib/python3.11/site-packages/sklearn/utils/validation.py:2742: UserWarning: X has feature names, but MinMaxScaler was fitted without feature names
      warnings.warn(

## Performance Evaluation

``` python
print(f"Pre ABG observations for performance evaluation: {len(performance_pre_df):,}.")
x_perf_scaled = x_scaler.transform(np.array(performance_pre_df.drop(["last_horowitz", "identifier", "paO2_measured"], axis=1).dropna()))
y_perf = performance_pre_df.drop(["last_horowitz", "identifier"], axis=1).dropna()["paO2_measured"].values.reshape(-1,1)
y_perf_scaled = y_scaler.transform(y_perf)
y_pred_perf_scaled = fitted_estimator.predict(x_perf_scaled)
y_pred_perf = y_scaler.inverse_transform(y_pred_perf_scaled.reshape(-1, 1))
```

    Pre ABG observations for performance evaluation: 5,177.

``` python
# errors
errors = y_pred_perf - y_perf

mae = np.mean(abs(errors))

mape = 100 * metrics.mean_absolute_percentage_error(y_perf, y_pred_perf)

mse = metrics.mean_squared_error(y_perf, y_pred_perf)
rmse = np.sqrt(mse)

# adjusted R2
r2 = metrics.r2_score(y_perf, y_pred_perf)
p = x_perf_scaled.shape[1]-2
n = len(y_perf) 
adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))

print({
    "r2": round(r2,4),
    "mape": round(mape,4),
    "mae": round(mae,4),
    "rmse": round(rmse,4),
    "adjusted_r2": round(adjusted_r2,4)
})
```

    {'r2': 0.6618, 'mape': 18.0729, 'mae': np.float64(44.8966), 'rmse': np.float64(61.5949), 'adjusted_r2': 0.6604}

## Prediction

``` python
print(f"Predicting {len(x_data_prediction_pre):,} paO2 values without ABG.")
scaled_pao2_predictions_pre = fitted_estimator.predict(x_data_prediction_pre)

pre_df["paO2_predicted"] = None
pre_df.loc[pre_df.paO2_measured.isnull(), "paO2_predicted"] = y_scaler.inverse_transform(scaled_pao2_predictions_pre.reshape(-1,1))


with open(Path(config.get('pickle').get("pre_abg")), 'wb') as f:
    pickle.dump(pre_df, f)
```

    Predicting 43,649 paO2 values without ABG.

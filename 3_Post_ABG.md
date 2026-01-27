# Continuous $paO_2$ Prediction and Postoperative Complications in
Neurosurgical Patients
Andrea S. Gutmann
2026-01-27

### Loading required libraries

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
import matplotlib
import numpy as np
import seaborn as sns
import yaml
from sklearn import metrics

# ======================
# Local / application
# ======================

pp = pprint.PrettyPrinter(indent=4)
np.random.seed(42)

matplotlib.rcParams["figure.dpi"] = 300

sns.set_style('whitegrid')

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


with open("data/out/data_ready.pickle", 'rb') as d:
    data = pickle.load(d)

with open(Path(config.get('pickle').get('scaler_y')), 'rb') as ys:
    y_scaler = pickle.load(ys)

with open(Path(config.get('pickle').get('scaler_x_lh')), 'rb') as xs:
    x_scaler = pickle.load(xs)

with open(config.get("pickle").get("selected_features"), "rb") as sf:
    selected_features = [str(s) for s in pickle.load(sf)]

with open(Path(config.get("pickle").get("estimator23")), "rb") as e:
    estimator = pickle.load(e)


post_df = data.loc[
    data.last_horowitz.notnull(),
    selected_features + ["last_horowitz", "identifier", "paO2_measured"]
]

fitting_post_df = post_df.loc[post_df.paO2_measured.notnull(),:].dropna()


prediction_post_df = post_df.loc[post_df.paO2_measured.isnull(),:]
x_data_prediction_post = x_scaler.transform(prediction_post_df.drop(["identifier", "paO2_measured"], axis = 1))
```

    /Users/abeckerp/Documents/pao2-complications-project/.venv/lib/python3.11/site-packages/sklearn/utils/validation.py:2742: UserWarning: X has feature names, but MinMaxScaler was fitted without feature names
      warnings.warn(

## Train and Test

``` python
np.random.seed(42)
shuffled_identifiers = list(set(fitting_post_df.identifier.values))
np.random.shuffle(shuffled_identifiers)
training, test = (
    shuffled_identifiers[: int(len(shuffled_identifiers) * 0.75)],
    shuffled_identifiers[int(len(shuffled_identifiers) * 0.75) :],
)

print(f"Number of surgeries in training set: {len(training):,}, number of surgeries in test set: {len(test):,}.")

# training
training_df = fitting_post_df.loc[fitting_post_df.identifier.isin(training),:]
print(f"Number of training observations: {training_df.shape[0]:,}.")
train_x_scaled = x_scaler.transform(np.array(training_df.drop(['identifier', 'paO2_measured'], axis=1)))
train_y_scaled = y_scaler.transform(training_df["paO2_measured"].values.reshape(-1,1))

# test
test_df = fitting_post_df.loc[fitting_post_df.identifier.isin(test),:]
print(f"Number of test observations: {test_df.shape[0]:,}.")
test_x_scaled = x_scaler.transform(np.array(test_df.drop(['identifier', 'paO2_measured'], axis=1)))
test_y_scaled = y_scaler.transform(test_df["paO2_measured"].values.reshape(-1,1))

fitted_estimator = estimator.fit(train_x_scaled, train_y_scaled.ravel())
y_pred_scaled = fitted_estimator.predict(test_x_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1,1))
```

    Number of surgeries in training set: 3,476, number of surgeries in test set: 1,159.
    Number of training observations: 11,105.
    Number of test observations: 3,670.

## Performance

``` python
y_perf = test_df["paO2_measured"].values.reshape(-1,1)
y_perf_scaled = y_scaler.transform(y_perf)


# errors
errors = y_pred - y_perf

mae = np.mean(abs(errors))

mape = 100 * metrics.mean_absolute_percentage_error(y_perf, y_pred)

mse = metrics.mean_squared_error(y_perf, y_pred)
rmse = np.sqrt(mse)

# adjusted R2
r2 = metrics.r2_score(y_perf, y_pred)
p = test_x_scaled.shape[1]-2
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

    {'r2': 0.7182, 'mape': 15.2797, 'mae': np.float64(26.7381), 'rmse': np.float64(44.7966), 'adjusted_r2': 0.7166}

## Prediction

``` python
print(f"Predicting {len(x_data_prediction_post):,} paO2 values without ABG.")
scaled_pao2_predictions_post = fitted_estimator.predict(x_data_prediction_post)


post_df["paO2_predicted"] = None
post_df.loc[post_df.paO2_measured.isnull(), "paO2_predicted"] = y_scaler.inverse_transform(scaled_pao2_predictions_post.reshape(-1,1))


with open(Path(config.get('pickle').get("post_abg")), 'wb') as f:
    pickle.dump(post_df, f)
```

    Predicting 263,270 paO2 values without ABG.

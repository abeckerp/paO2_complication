# Continuous $paO_2$ Prediction and Postoperative Complications in
Neurosurgical Patients
Andrea S. Gutmann
2026-02-09

# Preprocessing

## Loading required libraries

``` python
# ======================
# Standard library
# ======================
import pickle
import pprint
from collections import Counter
from pathlib import Path

# ======================
# Third-party libraries
# ======================
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy import stats

# ======================
# Local / application
# ======================
from statistical_functions import *

pp = pprint.PrettyPrinter(indent=4)
np.random.seed(42)

matplotlib.rcParams["figure.dpi"] = 300

sns.set_style('whitegrid')

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

with open(Path(config.get('pickle').get('data_for_analysis')), 'rb') as f:
    data = pickle.load(f)

data = data.astype({'max_paO2': float, 'norm_auc_paO2': float, 'norm_auc_pf': float, 'norm_auc_fiO2': float, 'auc_fiO2': float})

complications = config.get("complications_dict").keys()
diagnoses = config.get("diagnoses") + ["other"]
```

## Prepare data

``` python
print(f"Measured: {len(data.loc[data['pao2_type_measured']==True,:]):,}, predicted: {len(data.loc[data['pao2_type_measured']==False,:]):,}, number of surgeries: {len(data.identifier.unique()):,}, number of patients: {len(data.case_number.unique()):,}.")

analysis_df = data.loc[:, ['identifier', 'case_number', 'opdatum',
       'bmi', 'age', 'los', 'already_intubated', 'not_extubated',
       'ops', 'asa', 'time_to_incision',
       'time_to_end', 'mv_time', 'theater', 'sex_male',
       'incision_closure_time', 'creatinine', 'deceased',
       'intervention_count', 'sap_procedures', 'preop_ward', 'postop_ward',
       'norm_auc_paO2', 'auc_paO2', 'norm_auc_pf', 'max_paO2', 'norm_auc_fiO2', 'auc_fiO2', 'norm_auc_omv', 'auc_omv'] + list(config.get('procedure_codes')) + list(config.get("complications_dict_short").keys()) + diagnoses].drop_duplicates()


analysis_df = analysis_df.astype({'asa': 'int', 'sex_male': 'int', 'already_intubated': 'int', 'not_extubated': 'int'}).astype({'asa': 'object', 'sex_male': 'object', 'already_intubated': 'object', 'not_extubated': 'object'})

analysis_df['asa'] = analysis_df['asa'].apply(to_roman)

analysis_df = analysis_df.astype({'asa': 'category',})

print(f"Total number of patients for analysis: {len(analysis_df):,}.")
```

    Measured: 19,665, predicted: 304,928, number of surgeries: 5,020, number of patients: 5,020.
    Total number of patients for analysis: 5,020.

Test for normal distribution of $_{norm}paO_2$
(https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html)

``` python
statistic, pvalue = stats.normaltest(analysis_df['norm_auc_paO2'])

print(f"p-value for normal distribution of norm_paO2 is {pvalue:.4f} => not normally distributed.")
```

    p-value for normal distribution of norm_paO2 is 0.0000 => not normally distributed.

Figure SDC1

``` python
selected_identifiers = list(map(lambda x: x[0], list(sorted(Counter(data["identifier"]).items(),key = lambda x:x[1]))[-5:]))

for identifier in selected_identifiers:
    print(identifier, Counter(data.loc[data["identifier"]==identifier,"pao2_type_measured"]))

zipped = zip(selected_identifiers, [Counter(data.loc[data["identifier"]==identifier,"pao2_type_measured"]) for identifier in selected_identifiers])

max_entry = sorted(zipped, key=lambda x: -x[1][1])[0]

identifier_used = 1019888 # max_entry[0]
print(identifier_used)

matplotlib.rcParams["figure.dpi"] = 300
fig, ax = plt.subplots(figsize=(10,4))
y_measured = data.loc[(data["identifier"]==identifier_used)&(data["pao2_type_measured"]==1),"paO2_combined"]
x_measured = data.loc[(data["identifier"]==identifier_used)&(data["pao2_type_measured"]==1),"idx"]
y_predicted = data.loc[(data["identifier"]==identifier_used)&(data["pao2_type_measured"]==0),"paO2_combined"]
x_predicted = data.loc[(data["identifier"]==identifier_used)&(data["pao2_type_measured"]==0),"idx"]

ax.plot(data.loc[(data["identifier"]==identifier_used),"idx"], data.loc[(data["identifier"]==identifier_used),"paO2_combined"], "-")
ax.plot(x_measured, y_measured,"o", color = "red", label="measured $paO_2$ values")
ax.plot(x_predicted, y_predicted,"o", color = "blue", label="predicted $paO_2$ values")

ax.fill_between(data.loc[(data["identifier"]==identifier_used),"idx"],data.loc[(data["identifier"]==identifier_used),"paO2_combined"], 0, alpha=0.6, label="$paO_2$ integral (AUC)")

ax.set_xlabel("5 min interval")
ax.set_ylabel("$paO_2$ in mmHg")

ax.set_xlim(0, max(data.loc[(data["identifier"]==identifier_used),"idx"]))
ax.set_ylim(0)
ax.legend()

plt.savefig(f"./plots/pao2_auc_example.png", dpi=300, bbox_inches="tight")
ax.set_title("Measured and predicted $paO_2$ values of a patient")

plt.show()
```

    1378367 Counter({0: 161, 1: 6})
    1930358 Counter({0: 157, 1: 10})
    485488 Counter({0: 173, 1: 17})
    776344 Counter({0: 185, 1: 20})
    760577 Counter({0: 246, 1: 21})
    1019888

![Example of AUC calculation of measured and predicted paO2
values.](5_Patient_characteristics_files/figure-commonmark/figure_sdc1-output-2.png)

Figure SDC2

``` python
avg_data = (
    data.groupby(['identifier', 'pao2_type_measured'])['paO2_combined']
    .mean()
    .reset_index()
    .pivot(index='identifier', columns='pao2_type_measured', values='paO2_combined')
    .dropna()  # drop patients without both values
)

# Rename columns for clarity
avg_data = avg_data.rename(columns={1: 'Measured', 0: 'Predicted'})
avg_data['Mean'] = avg_data[['Measured', 'Predicted']].mean(axis=1)
avg_data['Difference'] = avg_data['Predicted'] - avg_data['Measured']

mean_diff = avg_data['Difference'].mean()
sd_diff = avg_data['Difference'].std()

# Create joint grid: scatter in middle, KDE on the side
g = sns.JointGrid(
    data=avg_data,
    x="Mean",
    y="Difference",
    height=6
)
g.fig.set_size_inches(7, 4)

# Scatter for Bland–Altman
g.plot_joint(sns.scatterplot, alpha=0.6)

# Add horizontal lines for bias and LoA
g.ax_joint.axhline(mean_diff, color='red', linestyle='--', label=f'Mean bias = {mean_diff:.1f}')
g.ax_joint.axhline(mean_diff + 1.96*sd_diff, color='gray', linestyle='--', label='95% limits of agreement')
g.ax_joint.axhline(mean_diff - 1.96*sd_diff, color='gray', linestyle='--')
g.ax_joint.legend()

# Hide the empty top marginal axis
g.ax_marg_x.remove()

# KDE for marginal distribution of differences
sns.kdeplot(
    y=avg_data["Difference"],
    fill=True, color="skyblue", ax=g.ax_marg_y, alpha=0.6
)

# Labels and title
g.set_axis_labels("Mean of measured and predicted $paO_2$ (mmHg)",
                  "Predicted − Measured $paO_2$ (mmHg)")

plt.savefig(f"./plots/bland-altman-density.png", dpi=300, bbox_inches="tight")
g.fig.suptitle("Bland–Altman plot with distribution of differences", y=0.89)

plt.show()
```

![Bland–Altman plot with distribution of
differences.](5_Patient_characteristics_files/figure-commonmark/figure_sdc2-output-1.png)

# Patient Characteristics

Combined $paO_2$ values

``` python
# all 
desc_df = data.loc[:,config.get('one_measurements')+config.get('multi_measurements')]

single_meas = (
    desc_df[config.get("one_measurements")]
    .groupby(["identifier"])
    .aggregate("mean")
    .describe(include="all")
    .reindex(sorted(config.get("one_measurements")), axis=1)
)

multiple_meas = (
    desc_df[config.get("multi_measurements")]
    .describe(include="all")
    .reindex(sorted(config.get("multi_measurements")), axis=1)
)

descriptive_df = multiple_meas.join(single_meas).drop("identifier", axis=1)

display(
    descriptive_df.rename(columns=config.get('feature_names')).T.rename_axis("my_idx")
    .sort_values(["count", "my_idx"])
    .round(2)
    .astype({"count": "int"})
)
descriptive_df.rename(columns=config.get('feature_names')).T.rename_axis("my_idx").sort_values(["count", "my_idx"]).round(2).astype({"count": "int"}).to_csv("./data/out/descriptives.csv", float_format="%.2f", decimal=".", index_label="")
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | count | mean | std | min | 25% | 50% | 75% | max |
|----|----|----|----|----|----|----|----|----|
| my_idx |  |  |  |  |  |  |  |  |
| \$\_{norm}p/F ratio\$ | 5020 | 455.55 | 85.81 | 76.54 | 399.46 | 455.39 | 507.55 | 1465.54 |
| \$\_{norm}paO_2\$ | 5020 | 220.24 | 46.32 | 39.78 | 188.21 | 214.06 | 245.95 | 455.95 |
| \$paO_2\$ AUC in \$mmHg^2\$ | 5020 | 69543.14 | 26860.62 | 7119.95 | 51015.52 | 67640.50 | 85394.13 | 360536.78 |
| Age in years | 5020 | 53.90 | 15.94 | 18.00 | 43.00 | 55.00 | 66.00 | 94.00 |
| BMI in \$kg/m^2\$ | 5020 | 25.17 | 4.39 | 14.81 | 22.31 | 24.62 | 27.47 | 59.20 |
| Mechanically ventilated before surgery | 5020 | 0.07 | 0.25 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| Preoperative creatinine value | 5020 | 0.94 | 0.29 | 0.20 | 0.80 | 0.90 | 1.00 | 9.30 |
| incision to closure time in min | 5020 | 239.34 | 103.57 | 13.00 | 172.00 | 231.00 | 296.56 | 1194.25 |
| initially measured p/F ratio in mmHg | 5020 | 477.14 | 125.91 | 300.00 | 398.79 | 467.42 | 530.04 | 2598.15 |
| max. \$paO_2\$ value in mmHg | 5020 | 408.97 | 58.64 | 139.21 | 386.19 | 411.77 | 436.38 | 678.32 |
| mechanical ventilation time in min | 5020 | 348.73 | 119.86 | 58.00 | 269.00 | 342.00 | 416.81 | 1382.00 |
| no extubation after surgery | 5020 | 0.19 | 0.40 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| number of surgeries during hospital stay | 5020 | 1.05 | 0.29 | 1.00 | 1.00 | 1.00 | 1.00 | 7.00 |
| postoperative (in-hospital) length of stay in days | 5020 | 10.95 | 9.93 | 1.00 | 6.00 | 7.00 | 12.00 | 175.00 |
| sex (0=female) | 5020 | 0.43 | 0.50 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| last measured p/F ratio in mmHg | 276252 | 449.37 | 117.38 | 26.00 | 385.42 | 453.12 | 509.79 | 2598.15 |
| static pulmonary compliance in \$ml/cm H_2O\$ | 324459 | 48.31 | 13.09 | 0.14 | 40.44 | 47.27 | 55.04 | 3122.86 |
| \$CO_2\$ in mmHg | 324593 | 34.45 | 3.82 | 0.00 | 32.94 | 34.96 | 36.72 | 72.11 |
| \$FiO_2\$ | 324593 | 0.49 | 0.18 | 0.14 | 0.38 | 0.44 | 0.52 | 1.08 |
| \$SpO_2\$ in % | 324593 | 98.72 | 1.23 | 43.42 | 98.00 | 99.00 | 100.00 | 100.00 |
| \$pAO_2\$ in mmHg | 324593 | 286.75 | 121.83 | -44.88 | 208.83 | 249.09 | 303.96 | 666.60 |
| Gadrey’s \$paO_2\$ in mmHg | 324593 | 106.64 | 18.46 | 26.12 | 91.64 | 105.20 | 132.76 | 132.76 |
| combination of measured and predicted \$paO_2\$ | 324593 | 217.94 | 80.60 | 5.02 | 164.42 | 196.07 | 242.75 | 678.32 |
| diastolic blood pressure in mmHg | 324593 | 59.54 | 9.27 | 0.34 | 53.44 | 58.86 | 64.78 | 254.12 |
| heart rate in 1/min | 324593 | 58.23 | 12.29 | 23.00 | 49.84 | 56.08 | 64.58 | 293.83 |
| hemoglobin in g/dL | 324593 | 12.29 | 1.44 | 0.00 | 11.51 | 12.35 | 13.21 | 17.90 |
| index of measurement | 324593 | 38.71 | 25.59 | 1.00 | 18.00 | 35.00 | 55.00 | 268.00 |
| mean arterial pressure in mmHg | 324593 | 79.38 | 10.00 | 0.43 | 72.98 | 78.44 | 84.98 | 254.75 |
| measurement taken intraoperatively | 324593 | 0.73 | 0.44 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| p/F ratio in mmHg | 324593 | 453.57 | 112.56 | 26.00 | 395.59 | 456.53 | 510.42 | 2598.15 |
| pH | 324593 | 7.42 | 0.04 | 7.05 | 7.40 | 7.42 | 7.44 | 7.98 |
| respiratory minute volume in L | 324593 | 5.55 | 1.32 | 0.01 | 4.60 | 5.40 | 6.33 | 28.27 |
| respiratory rate in 1/min | 324593 | 10.83 | 2.18 | 0.04 | 10.00 | 10.44 | 12.00 | 46.95 |
| systolic blood pressure in mmHg | 324593 | 114.25 | 13.96 | 0.66 | 105.02 | 113.08 | 122.04 | 298.68 |
| temperature in °C | 324593 | 36.22 | 0.78 | 9.41 | 35.70 | 36.23 | 36.80 | 39.78 |

</div>

Measured $paO_2$

``` python
## measured paO2 
desc_df = data.loc[data['pao2_type_measured']==True,config.get('one_measurements')+config.get('multi_measurements')]

single_meas = (
    desc_df[config.get("one_measurements")]
    .groupby(["identifier"])
    .aggregate("mean")
    .describe(include="all")
    .reindex(sorted(config.get("one_measurements")), axis=1)
)

multiple_meas = (
    desc_df[config.get("multi_measurements")]
    .describe(include="all")
    .reindex(sorted(config.get("multi_measurements")), axis=1)
)

descriptive_df = multiple_meas.join(single_meas).drop("identifier", axis=1)

display(
    descriptive_df.T.rename_axis("my_idx")
    .sort_values(["count", "my_idx"])
    .round(2)
    .astype({"count": "int"})
)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | count | mean | std | min | 25% | 50% | 75% | max |
|----|----|----|----|----|----|----|----|----|
| my_idx |  |  |  |  |  |  |  |  |
| first_horowitz | 5019 | 477.15 | 125.92 | 300.00 | 398.76 | 467.42 | 530.04 | 2598.15 |
| age | 5020 | 53.90 | 15.94 | 18.00 | 43.00 | 55.00 | 66.00 | 94.00 |
| already_intubated | 5020 | 0.07 | 0.25 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| auc_paO2 | 5020 | 69543.14 | 26860.62 | 7119.95 | 51015.52 | 67640.50 | 85394.13 | 360536.78 |
| bmi | 5020 | 25.17 | 4.39 | 14.81 | 22.31 | 24.62 | 27.47 | 59.20 |
| creatinine | 5020 | 0.94 | 0.29 | 0.20 | 0.80 | 0.90 | 1.00 | 9.30 |
| incision_closure_time | 5020 | 239.34 | 103.57 | 13.00 | 172.00 | 231.00 | 296.56 | 1194.25 |
| intervention_count | 5020 | 1.05 | 0.29 | 1.00 | 1.00 | 1.00 | 1.00 | 7.00 |
| los | 5020 | 10.95 | 9.93 | 1.00 | 6.00 | 7.00 | 12.00 | 175.00 |
| max_paO2 | 5020 | 408.97 | 58.64 | 139.21 | 386.19 | 411.77 | 436.38 | 678.32 |
| mv_time | 5020 | 348.73 | 119.86 | 58.00 | 269.00 | 342.00 | 416.81 | 1382.00 |
| norm_auc_paO2 | 5020 | 220.24 | 46.32 | 39.78 | 188.21 | 214.06 | 245.95 | 455.95 |
| norm_auc_pf | 5020 | 455.55 | 85.81 | 76.54 | 399.46 | 455.39 | 507.55 | 1465.54 |
| not_extubated | 5020 | 0.19 | 0.40 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| sex_male | 5020 | 0.43 | 0.50 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| last_horowitz | 14593 | 445.97 | 113.92 | 26.00 | 382.29 | 450.87 | 507.69 | 2598.15 |
| compliance | 19531 | 48.52 | 24.89 | 0.14 | 40.52 | 47.37 | 55.22 | 3122.86 |
| co2 | 19665 | 34.43 | 3.56 | 0.00 | 32.86 | 34.93 | 36.72 | 58.38 |
| diastolic | 19665 | 59.33 | 9.26 | 16.01 | 53.21 | 58.56 | 64.27 | 246.97 |
| fio2 | 19665 | 0.49 | 0.17 | 0.20 | 0.38 | 0.44 | 0.51 | 1.00 |
| gadrey | 19665 | 106.67 | 18.49 | 32.41 | 91.64 | 105.20 | 132.76 | 132.76 |
| heart_rate | 19665 | 58.63 | 12.76 | 27.17 | 49.97 | 56.83 | 65.00 | 171.36 |
| hemoglobin | 19665 | 12.19 | 1.59 | 0.00 | 11.39 | 12.30 | 13.20 | 17.80 |
| horowitz | 19665 | 454.23 | 118.07 | 26.00 | 386.24 | 455.32 | 513.33 | 2598.15 |
| idx | 19665 | 37.19 | 25.59 | 1.00 | 15.00 | 34.00 | 53.00 | 260.00 |
| mean_art_press | 19665 | 79.30 | 10.03 | 25.01 | 72.98 | 78.43 | 84.85 | 246.98 |
| pAO2 | 19665 | 279.96 | 114.54 | -44.88 | 209.19 | 248.79 | 296.56 | 656.01 |
| paO2_combined | 19665 | 216.22 | 93.55 | 30.90 | 156.50 | 191.00 | 239.00 | 607.40 |
| ph | 19665 | 7.41 | 0.04 | 7.05 | 7.39 | 7.42 | 7.44 | 7.98 |
| respiratory_rate | 19665 | 10.89 | 2.19 | 0.04 | 10.00 | 10.99 | 12.00 | 41.41 |
| rmv | 19665 | 5.59 | 1.35 | 0.01 | 4.62 | 5.40 | 6.40 | 16.31 |
| spo2 | 19665 | 98.73 | 1.23 | 59.62 | 98.00 | 99.00 | 100.00 | 100.00 |
| systolic | 19665 | 114.89 | 14.30 | 39.02 | 105.68 | 114.00 | 122.92 | 271.32 |
| temperature | 19665 | 36.22 | 0.92 | 9.41 | 35.75 | 36.29 | 36.80 | 39.00 |
| timepoint_intraop | 19665 | 0.71 | 0.46 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |

</div>

Imputed $paO_2$

``` python
## imputed paO2
desc_df = data.loc[data['pao2_type_measured']==False,config.get('one_measurements')+config.get('multi_measurements')]

single_meas = (
    desc_df[config.get("one_measurements")]
    .groupby(["identifier"])
    .aggregate("mean")
    .describe(include="all")
    .reindex(sorted(config.get("one_measurements")), axis=1)
)

multiple_meas = (
    desc_df[config.get("multi_measurements")]
    .describe(include="all")
    .reindex(sorted(config.get("multi_measurements")), axis=1)
)

descriptive_df = multiple_meas.join(single_meas).drop("identifier", axis=1)

display(
    descriptive_df.T.rename_axis("my_idx")
    .sort_values(["count", "my_idx"])
    .round(2)
    .astype({"count": "int"})
)
# descriptive_df.T.rename_axis("my_idx").sort_values(["count", "my_idx"]).round(2).astype({"count": "int"}).to_csv(config.get('csv').get('descriptives'))
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | count | mean | std | min | 25% | 50% | 75% | max |
|----|----|----|----|----|----|----|----|----|
| my_idx |  |  |  |  |  |  |  |  |
| first_horowitz | 5016 | 477.21 | 125.92 | 300.00 | 398.94 | 467.46 | 530.06 | 2598.15 |
| age | 5020 | 53.90 | 15.94 | 18.00 | 43.00 | 55.00 | 66.00 | 94.00 |
| already_intubated | 5020 | 0.07 | 0.25 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| auc_paO2 | 5020 | 69543.14 | 26860.62 | 7119.95 | 51015.52 | 67640.50 | 85394.13 | 360536.78 |
| bmi | 5020 | 25.17 | 4.39 | 14.81 | 22.31 | 24.62 | 27.47 | 59.20 |
| creatinine | 5020 | 0.94 | 0.29 | 0.20 | 0.80 | 0.90 | 1.00 | 9.30 |
| incision_closure_time | 5020 | 239.34 | 103.57 | 13.00 | 172.00 | 231.00 | 296.56 | 1194.25 |
| intervention_count | 5020 | 1.05 | 0.29 | 1.00 | 1.00 | 1.00 | 1.00 | 7.00 |
| los | 5020 | 10.95 | 9.93 | 1.00 | 6.00 | 7.00 | 12.00 | 175.00 |
| max_paO2 | 5020 | 408.97 | 58.64 | 139.21 | 386.19 | 411.77 | 436.38 | 678.32 |
| mv_time | 5020 | 348.73 | 119.86 | 58.00 | 269.00 | 342.00 | 416.81 | 1382.00 |
| norm_auc_paO2 | 5020 | 220.24 | 46.32 | 39.78 | 188.21 | 214.06 | 245.95 | 455.95 |
| norm_auc_pf | 5020 | 455.55 | 85.81 | 76.54 | 399.46 | 455.39 | 507.55 | 1465.54 |
| not_extubated | 5020 | 0.19 | 0.40 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| sex_male | 5020 | 0.43 | 0.50 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| last_horowitz | 261659 | 449.56 | 117.57 | 26.00 | 385.65 | 453.33 | 510.00 | 2598.15 |
| co2 | 304928 | 34.45 | 3.84 | 0.00 | 32.94 | 34.96 | 36.73 | 72.11 |
| compliance | 304928 | 48.30 | 11.95 | 10.00 | 40.44 | 47.27 | 55.02 | 249.46 |
| diastolic | 304928 | 59.55 | 9.27 | 0.34 | 53.46 | 58.88 | 64.80 | 254.12 |
| fio2 | 304928 | 0.49 | 0.18 | 0.14 | 0.38 | 0.44 | 0.52 | 1.08 |
| gadrey | 304928 | 106.63 | 18.46 | 26.12 | 91.64 | 105.20 | 132.76 | 132.76 |
| heart_rate | 304928 | 58.21 | 12.25 | 23.00 | 49.83 | 56.06 | 64.52 | 293.83 |
| hemoglobin | 304928 | 12.30 | 1.43 | 5.02 | 11.52 | 12.35 | 13.21 | 17.90 |
| horowitz | 304928 | 453.52 | 112.20 | 26.00 | 396.54 | 456.60 | 510.26 | 2598.15 |
| idx | 304928 | 38.80 | 25.58 | 1.00 | 18.00 | 35.00 | 55.00 | 268.00 |
| mean_art_press | 304928 | 79.39 | 9.99 | 0.43 | 72.98 | 78.44 | 84.99 | 254.75 |
| pAO2 | 304928 | 287.18 | 122.27 | -43.83 | 208.81 | 249.15 | 304.56 | 666.60 |
| paO2_combined | 304928 | 218.05 | 79.70 | 5.02 | 164.91 | 196.42 | 242.99 | 678.32 |
| ph | 304928 | 7.42 | 0.04 | 7.06 | 7.40 | 7.42 | 7.44 | 7.97 |
| respiratory_rate | 304928 | 10.83 | 2.18 | 5.00 | 10.00 | 10.37 | 12.00 | 46.95 |
| rmv | 304928 | 5.55 | 1.32 | 2.00 | 4.60 | 5.40 | 6.33 | 28.27 |
| spo2 | 304928 | 98.72 | 1.23 | 43.42 | 98.00 | 99.00 | 100.00 | 100.00 |
| systolic | 304928 | 114.21 | 13.94 | 0.66 | 105.01 | 113.04 | 122.01 | 298.68 |
| temperature | 304928 | 36.22 | 0.77 | 32.00 | 35.70 | 36.22 | 36.80 | 39.78 |
| timepoint_intraop | 304928 | 0.73 | 0.44 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |

</div>

## $paO_2$ Values (mean per patient)

``` python
measured1 = data.loc[(data["pao2_type_measured"]==1),:].groupby("identifier").agg({"paO2_combined":"mean"})
measured1["paO2_type"] = "measured"
measured1["timepoint"] = "peri-operative"
predicted1 = data.loc[(data["pao2_type_measured"]==0),:].groupby("identifier").agg({"paO2_combined":"mean"})
predicted1["paO2_type"] = "predicted"
predicted1["timepoint"] = "peri-operative"


plot_data = pd.concat([measured1, predicted1], axis=0)

# measured paO2 
print(f"Descriptives of measured perioperative paO2 values (mean per patient):\n{data.loc[(data['pao2_type_measured']==1),:].groupby('identifier').agg({'paO2_combined':'mean'}).describe().round(2)}")

# predicted paO2
print(f"Descriptives of predicted perioperative paO2 values (mean per patient):\n{data.loc[(data['pao2_type_measured']==0),:].groupby('identifier').agg({'paO2_combined':'mean'}).describe().round(2)}")

# all paO2
print(f"Descriptives of all perioperative paO2 values (mean per patient):\n{data.groupby('identifier').agg({'paO2_combined':'mean'}).describe().round(2)}")


pval, ph_dunn, effect_sizes = compare_median("paO2_combined", plot_data, "paO2_type")
```

    Descriptives of measured perioperative paO2 values (mean per patient):
           paO2_combined
    count        5020.00
    mean          223.29
    std            71.22
    min            91.50
    25%           173.75
    50%           209.33
    75%           256.89
    max           602.70
    Descriptives of predicted perioperative paO2 values (mean per patient):
           paO2_combined
    count        5020.00
    mean          221.40
    std            45.62
    min            27.78
    25%           189.86
    50%           215.70
    75%           246.87
    max           449.67
    Descriptives of all perioperative paO2 values (mean per patient):
           paO2_combined
    count        5020.00
    mean          221.43
    std            46.09
    min            37.45
    25%           189.60
    50%           215.25
    75%           247.20
    max           455.37
              paO2_combined                  
                     median        mean count
    paO2_type                                
    measured     209.329167  223.293129  5020
    predicted    215.701288  221.400492  5020
    p-value: 0.0

    ... Post hoc test...
         1    2
    1    -  ***
    2  ***    -
           1      2
    1 1.0000 0.0000
    2 0.0000 1.0000

    1, 2: Effect size: Hedges' g: 0.03 (very small)
    [(np.int64(0), np.int64(1), np.float64(0.031648429221880464))]

    1, 2: Effect size: Hedges' g: 0.03 (very small)

No aggregation per patient

``` python
pval, ph_dunn, effect_sizes = compare_median("paO2_combined", data, "pao2_type_measured")
```

                       paO2_combined                 
                              median     mean   count
    pao2_type_measured                               
    0                       196.4151 218.0542  304928
    1                       191.0000 216.2164   19665
    p-value: 0.0

    ... Post hoc test...
         1    2
    1    -  ***
    2  ***    -
           1      2
    1 1.0000 0.0000
    2 0.0000 1.0000

    1, 2: Effect size: Hedges' g: 0.02 (very small)
    [(np.int64(0), np.int64(1), np.float64(0.02279958948215179))]

    1, 2: Effect size: Hedges' g: 0.02 (very small)

## Sociodemographics / Descriptive Statistics

``` python
for c in ['bmi', 'age', 'los',  'time_to_incision', 'time_to_end', 'mv_time', 'incision_closure_time',]: 
    print(f"Descriptives for {c.upper()}:\n{analysis_df[c].describe().round(2)}\n")

print(f"ASA:\n{analysis_df.groupby('asa', observed=False).agg('count')['identifier']}")
print(f"Sex:\n{analysis_df.groupby('sex_male', observed=False).agg('count')['identifier']}")
print(f"Deceased:\n{analysis_df.groupby('deceased', observed=False).agg('count')['identifier']}")
print(return_table(analysis_df, ['asa', 'deceased']))
print(f"Already intubated before surgery:\n{analysis_df.groupby('already_intubated', observed=False).agg('count')['identifier']}")
print(f"Not extubated after surgery:\n{analysis_df.groupby('not_extubated', observed=False).agg('count')['identifier']}")
print(f"Preop Ward:\n{analysis_df.groupby('preop_ward', observed=False).agg('count').sort_values('identifier', ascending=False)['identifier']}")
print(f"Preop ICU:\n{analysis_df.preop_ward.str.count('Intensiv|IMC').sum():,} ({100/len(analysis_df)*analysis_df.preop_ward.str.count('Intensiv').sum():.1f}%)")
print(f"Postop Ward:\n{analysis_df.groupby('postop_ward', observed=False).agg('count').sort_values('identifier', ascending=False)['identifier']}")
print(f"Postop ICU:\n{analysis_df.postop_ward.str.count('Intensiv|IMC').sum():,} ({100/len(analysis_df)*analysis_df.postop_ward.str.count('Intensiv|IMC').sum():.1f}%)")
```

    Descriptives for BMI:
    count   5020.0000
    mean      25.1700
    std        4.3900
    min       14.8100
    25%       22.3100
    50%       24.6200
    75%       27.4700
    max       59.2000
    Name: bmi, dtype: float64

    Descriptives for AGE:
    count   5020.0000
    mean      53.9000
    std       15.9400
    min       18.0000
    25%       43.0000
    50%       55.0000
    75%       66.0000
    max       94.0000
    Name: age, dtype: float64

    Descriptives for LOS:
    count   5020.0000
    mean      10.9500
    std        9.9300
    min        1.0000
    25%        6.0000
    50%        7.0000
    75%       12.0000
    max      175.0000
    Name: los, dtype: float64

    Descriptives for TIME_TO_INCISION:
    count   5020.0000
    mean      83.3700
    std       27.1600
    min        4.2500
    25%       67.0000
    50%       82.0000
    75%       99.0000
    max      222.0000
    Name: time_to_incision, dtype: float64

    Descriptives for TIME_TO_END:
    count   5020.0000
    mean      26.0200
    std       19.7000
    min        0.0300
    25%       13.0000
    50%       20.0000
    75%       32.0000
    max      194.0000
    Name: time_to_end, dtype: float64

    Descriptives for MV_TIME:
    count   5020.0000
    mean     348.7300
    std      119.8600
    min       58.0000
    25%      269.0000
    50%      342.0000
    75%      416.8100
    max     1382.0000
    Name: mv_time, dtype: float64

    Descriptives for INCISION_CLOSURE_TIME:
    count   5020.0000
    mean     239.3400
    std      103.5700
    min       13.0000
    25%      172.0000
    50%      231.0000
    75%      296.5600
    max     1194.2500
    Name: incision_closure_time, dtype: float64

    ASA:
    asa
    I       255
    II     2143
    III    2175
    IV      414
    V        33
    Name: identifier, dtype: int64
    Sex:
    sex_male
    0    2854
    1    2166
    Name: identifier, dtype: int64
    Deceased:
    deceased
    False    4931
    True       89
    Name: identifier, dtype: int64
    asa  deceased
    I    False        255
         True           0
    II   False       2137
         True           6
    III  False       2147
         True          28
    IV   False        363
         True          51
    V    False         29
         True           4
    dtype: int64
    [array([255,   0]), array([2137,    6]), array([2147,   28]), array([363,  51]), array([29,  4])]
    Already intubated before surgery:
    already_intubated
    0    4673
    1     347
    Name: identifier, dtype: int64
    Not extubated after surgery:
    not_extubated
    0    4049
    1     971
    Name: identifier, dtype: int64
    Preop Ward:
    preop_ward
    H9                        2222
    I9A                        672
    I9                         331
    G21                        325
    G21B                       322
    G22B                       238
    H3B Intensiv               172
    Nothilfe/ZNA               145
    ITS5 Intensiv              145
    I9B                        143
    I2  Intensiv                75
    Nothilfe                    48
    I9 IMC                      42
    G8 Stroke Unit IMC          33
    I8 (HNGI8)                   8
    ANIS5 Intensiv               7
    unbekannt                    7
    H8                           6
    H8 Epilepsieeinheit          6
    G8                           5
    ITS3 Intensiv                5
    G21A                         5
    I7                           4
    I3 Intensiv                  4
    F2A Intensiv                 4
    ITS2 Intensiv                4
    H2 Intensiv                  4
    F10A                         3
    K21                          3
    F8                           3
    F2B  Intensiv                2
    H7                           2
    G5 Intensiv                  2
    G10A                         1
    I22B                         1
    ZNA Notaufnahmestation       1
    S5 Intensiv                  1
    OP GRH                       1
    G3                           1
    L21 Intensiv                 1
    K22A                         1
    K22                          1
    G6                           1
    G7                           1
    G2                           1
    H10                          1
    ITS1 Intensiv                1
    F2B Intensiv                 1
    F3                           1
    F6                           1
    F7                           1
    G11                          1
    Außer Haus                   1
    G10B                         1
    I3                           1
    G22                          1
    Name: identifier, dtype: int64
    Preop ICU:
    503 (8.5%)
    Postop Ward:
    postop_ward
    H3B Intensiv           2509
    ITS5 Intensiv          1939
    I2  Intensiv            176
    H9                       65
    ANIS5 Intensiv           51
    I9B                      37
    I9                       32
    ITS2 Intensiv            32
    I9 IMC                   31
    H2 Intensiv              25
    I3 Intensiv              20
    I9A                      19
    ITS3 Intensiv            14
    G21B                     13
    G21                      13
    G9B  Intensiv             7
    F2A Intensiv              6
    G8 Stroke Unit IMC        5
    G5 Intensiv               4
    F2B Intensiv              3
    ITS1 Intensiv             2
    H8                        2
    H8 Epilepsieeinheit       2
    H7                        2
    G22B                      2
    F2B  Intensiv             2
    S5 Intensiv               2
    I8 (HNGI8)                1
    H3A Intensiv              1
    G8                        1
    F2C Intensiv              1
    OP GRH                    1
    Name: identifier, dtype: int64
    Postop ICU:
    4,830 (96.2%)

## Diagnosis, Post-OP Complications, Surgery

``` python
print(analysis_df.groupby("intervention_count").count()['identifier'])

print("Diagnoses: ")
diag_dict = {
    d: (
        int(analysis_df[d].sum()),
        f"{round(100/len(analysis_df)*analysis_df[d].sum(),1)} %",
    )
    for d in diagnoses
}
for d, (n, p) in sorted(diag_dict.items(), key=lambda x: -x[1][0]):
    print(f"{d}: {n:,} ({p})")


print(
    f"\nPatients with one underlying disease: {(analysis_df.apply(lambda row: 1 if row[diagnoses].sum() == 1 else 0, axis=1).sum()):,} ({round(100/len(analysis_df)*(analysis_df.apply(lambda row: 1 if row[diagnoses].sum() == 1 else 0, axis=1).sum()),1)} %)."
)
print(
    f"Patients with two underlying diseases: {(analysis_df.apply(lambda row: 1 if row[diagnoses].sum() ==2 else 0, axis=1).sum())} ({round(100/len(analysis_df)*(analysis_df.apply(lambda row: 1 if row[diagnoses].sum() ==2 else 0, axis=1).sum()),1)} %)."
)
print(
    f"Patients with three underlying diseases: {(analysis_df.apply(lambda row: 1 if row[diagnoses].sum() ==3 else 0, axis=1).sum())} ({round(100/len(analysis_df)*(analysis_df.apply(lambda row: 1 if row[diagnoses].sum() ==3 else 0, axis=1).sum()),1)} %)."
)
print(
    f"Patients with four underlying diseases: {(analysis_df.apply(lambda row: 1 if row[diagnoses].sum() ==4 else 0, axis=1).sum())} ({round(100/len(analysis_df)*(analysis_df.apply(lambda row: 1 if row[diagnoses].sum() ==4 else 0, axis=1).sum()),1)} %)."
)

# print(f"Complications:\n{analysis_df.loc[:, [el for el in complications]].sum()}")

print("\nComplications: ")
compl_dict = {
    c: (
        int(analysis_df[c].sum()),
        f"{round(100/len(analysis_df)*analysis_df[c].sum(),1)} %",
    )
    for c in complications
}
for c, (n, p) in sorted(compl_dict.items(), key=lambda x: -x[1][0]):
    print(f"{c}: {n:,} ({p})")


print(
    f"\nPatients with one postoperative complication: {(analysis_df.apply(lambda row: 1 if row[complications].sum() == 1 else 0, axis=1).sum()):,} ({round(100/len(analysis_df)*(analysis_df.apply(lambda row: 1 if row[complications].sum() == 1 else 0, axis=1).sum()),1)} %)."
)
print(
    f"Patients with two postoperative complications: {(analysis_df.apply(lambda row: 1 if row[complications].sum() ==2 else 0, axis=1).sum())} ({round(100/len(analysis_df)*(analysis_df.apply(lambda row: 1 if row[complications].sum() ==2 else 0, axis=1).sum()),1)} %)."
)
print(
    f"Patients with three or more postoperative complications: {(analysis_df.apply(lambda row: 1 if row[complications].sum() >2 else 0, axis=1).sum())} ({round(100/len(analysis_df)*(analysis_df.apply(lambda row: 1 if row[complications].sum() > 2 else 0, axis=1).sum()),1)} %)."
)
```

    intervention_count
    1    4794
    2     192
    3      24
    4       6
    5       3
    7       1
    Name: identifier, dtype: int64
    Diagnoses: 
    benign_neoplasm: 2,009 (40.0 %)
    malignant_neoplasm: 963 (19.2 %)
    intracranial_hemorrhage: 471 (9.4 %)
    cerebral_aneurysm: 422 (8.4 %)
    other: 390 (7.8 %)
    TBI: 217 (4.3 %)
    epilepsy: 204 (4.1 %)
    trigeminus: 193 (3.8 %)
    SAH: 189 (3.8 %)
    neoplasm: 101 (2.0 %)
    other_aneurysm_dissection: 27 (0.5 %)

    Patients with one underlying disease: 4,863 (96.9 %).
    Patients with two underlying diseases: 148 (2.9 %).
    Patients with three underlying diseases: 9 (0.2 %).
    Patients with four underlying diseases: 0 (0.0 %).

    Complications: 
    pneumonia: 187 (3.7 %)
    stroke: 171 (3.4 %)
    pulmonary_embolism: 168 (3.3 %)
    cerebral_vasospasm: 126 (2.5 %)
    sepsis: 67 (1.3 %)
    renal_failure: 55 (1.1 %)
    myocardial_infarction: 17 (0.3 %)
    liver_failure: 12 (0.2 %)

    Patients with one postoperative complication: 481 (9.6 %).
    Patients with two postoperative complications: 109 (2.2 %).
    Patients with three or more postoperative complications: 31 (0.6 %).

SDC 5

``` python
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

sdc5_data = []
sdc5_figure_data = []
annot_data = []

for c, (n,p) in sorted(compl_dict.items(), key=lambda x: -x[1][0]):    
    sdc5_row = []
    annot_row = []
    sdc5_figure_row = []
    for d, (n,p) in sorted(diag_dict.items(), key=lambda x: -x[1][0]):
        cell_count = analysis_df.loc[(analysis_df[d] == True) & (analysis_df[c] == True), :].shape[0]
        sdc5_row.append(cell_count)
        annot_row.append(f"{100/n*cell_count:.1f}%\n(N={cell_count})")
        sdc5_figure_row.append(100/n*cell_count)
    sdc5_data.append(sdc5_row)
    annot_data.append(annot_row)
    sdc5_figure_data.append(sdc5_figure_row)

sdc5 = pd.DataFrame(
    np.array(sdc5_data).T,
    index=[
        f"{config.get('long_names').get(d)} (N={n:,})"
        for d, (n,p) in sorted(diag_dict.items(), key=lambda x: -x[1][0])
    ],
    columns=[
        f"{config.get('long_names').get(c)} (N={n:,})"
        for c, (n,p) in sorted(compl_dict.items(), key=lambda x: -x[1][0])
    ],
)
sdc5_figure = pd.DataFrame(
    np.array(sdc5_figure_data).T,
    index=[
        f"{config.get('long_names').get(d)} (N={n:,})"
        for d, (n,p) in sorted(diag_dict.items(), key=lambda x: -x[1][0])
    ],
    columns=[
        f"{config.get('long_names').get(c)} (N={n:,})"
        for c, (n,p) in sorted(compl_dict.items(), key=lambda x: -x[1][0])
    ],
)

display(sdc5)
display(sdc5_figure)

sdc5.to_csv("./data/out/diag_comp.csv")
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | Pneumonia (N=187) | Ischaemic stroke (N=171) | Pulmonary embolism (N=168) | Cerebral vasospasm (N=126) | Sepsis (N=67) | Renal failure (N=55) | Myocardial infarction (N=17) | Liver failure (N=12) |
|----|----|----|----|----|----|----|----|----|
| Benign neoplasm (N=2,009) | 36 | 36 | 94 | 11 | 10 | 8 | 3 | 2 |
| Malignant neoplasm (N=963) | 37 | 15 | 25 | 10 | 9 | 7 | 1 | 0 |
| Intracranial hemorrhage (N=471) | 44 | 30 | 13 | 18 | 18 | 22 | 6 | 1 |
| Cerebral aneurysm (N=422) | 8 | 25 | 15 | 21 | 6 | 1 | 3 | 1 |
| Other diseases of the brain (N=390) | 17 | 19 | 8 | 5 | 10 | 4 | 0 | 1 |
| Traumatic brain injury (N=217) | 28 | 10 | 6 | 5 | 8 | 8 | 2 | 3 |
| Epilepsy (N=204) | 1 | 4 | 1 | 0 | 0 | 0 | 0 | 0 |
| Facial nerve disorders and disorders of trigeminal nerve (N=193) | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| Subarachnoid hemorrhage (N=189) | 27 | 48 | 11 | 80 | 16 | 9 | 2 | 5 |
| Neoplasm of uncertain or unknown behavior (N=101) | 6 | 3 | 3 | 3 | 2 | 2 | 1 | 0 |
| Other aneurysms and dissections (N=27) | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |

</div>

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | Pneumonia (N=187) | Ischaemic stroke (N=171) | Pulmonary embolism (N=168) | Cerebral vasospasm (N=126) | Sepsis (N=67) | Renal failure (N=55) | Myocardial infarction (N=17) | Liver failure (N=12) |
|----|----|----|----|----|----|----|----|----|
| Benign neoplasm (N=2,009) | 1.7919 | 1.7919 | 4.6789 | 0.5475 | 0.4978 | 0.3982 | 0.1493 | 0.0996 |
| Malignant neoplasm (N=963) | 3.8422 | 1.5576 | 2.5961 | 1.0384 | 0.9346 | 0.7269 | 0.1038 | 0.0000 |
| Intracranial hemorrhage (N=471) | 9.3418 | 6.3694 | 2.7601 | 3.8217 | 3.8217 | 4.6709 | 1.2739 | 0.2123 |
| Cerebral aneurysm (N=422) | 1.8957 | 5.9242 | 3.5545 | 4.9763 | 1.4218 | 0.2370 | 0.7109 | 0.2370 |
| Other diseases of the brain (N=390) | 4.3590 | 4.8718 | 2.0513 | 1.2821 | 2.5641 | 1.0256 | 0.0000 | 0.2564 |
| Traumatic brain injury (N=217) | 12.9032 | 4.6083 | 2.7650 | 2.3041 | 3.6866 | 3.6866 | 0.9217 | 1.3825 |
| Epilepsy (N=204) | 0.4902 | 1.9608 | 0.4902 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Facial nerve disorders and disorders of trigeminal nerve (N=193) | 0.0000 | 1.0363 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Subarachnoid hemorrhage (N=189) | 14.2857 | 25.3968 | 5.8201 | 42.3280 | 8.4656 | 4.7619 | 1.0582 | 2.6455 |
| Neoplasm of uncertain or unknown behavior (N=101) | 5.9406 | 2.9703 | 2.9703 | 2.9703 | 1.9802 | 1.9802 | 0.9901 | 0.0000 |
| Other aneurysms and dissections (N=27) | 3.7037 | 3.7037 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

</div>

Figure 2

``` python
ax = sns.heatmap(
    sdc5_figure,
    linewidth=0.5,
    annot=np.array(annot_data).T,
    cmap="gray_r",
    fmt="s",
    annot_kws={"size": 7.5},
)
# add label to colorbar
cbar = ax.collections[0].colorbar
cbar.set_label("Percentage (%)")
plt.savefig("./plots/heatmap.png", dpi=300, bbox_inches="tight")
ax.set_title("Heatmap of postoperative complications per diagnosis")

plt.show()
```

![Heatmap of postoperative complications per
diagnosis](5_Patient_characteristics_files/figure-commonmark/heatmap-output-1.png)

## Differences in $paO_2$ and p/F ratio between sexes

``` python
pval, ph_dunn, effect_sizes = compare_median("norm_auc_paO2", analysis_df, "sex_male")
print(f"\nDifferences between male and female in normalized perioperative auc values: {round(pval,4)}.\n\n")

median_fio2_df = pd.DataFrame(
    {
        "median_fio2": list(
            data[data["sex_male"] == True]
            .groupby("identifier")["fio2"]
            .median()
            .values
        )
        + list(
            data[data["sex_male"] == False]
            .groupby("identifier")["fio2"]
            .median()
            .values
        ),
        "sex_male": sum(analysis_df["sex_male"] == True) * [True]
        + sum(analysis_df["sex_male"] == False) * [False],
    }
)
pval, ph_dunn, effect_sizes = compare_median("median_fio2", median_fio2_df, "sex_male")

median_horowitz_df = pd.DataFrame(
    {
        "median_horowitz": list(
            data[data["sex_male"] == True]
            .groupby("identifier")["horowitz"]
            .median()
            .values
        )
        + list(
            data[data["sex_male"] == False]
            .groupby("identifier")["horowitz"]
            .median()
            .values
        ),
        "sex_male": sum(analysis_df["sex_male"] == True) * [True]
        + sum(analysis_df["sex_male"] == False) * [False],
    }
)
pval, ph_dunn, effect_sizes = compare_median("median_horowitz", median_horowitz_df, "sex_male")

pvalue, ph_dunn, effect_sizes = compare_median("norm_auc_pf", analysis_df, "sex_male")
print(f"\nDifferences between male and female in normalized perioperative p/F auc values: {round(pvalue,4)}.\n\n")
```

             norm_auc_paO2               
                    median     mean count
    sex_male                             
    0             221.4527 227.0479  2854
    1             206.2754 211.2733  2166
    p-value: 0.0

    ... Post hoc test...
         1    2
    1    -  ***
    2  ***    -
           1      2
    1 1.0000 0.0000
    2 0.0000 1.0000

    1, 2: Effect size: Hedges' g: 0.35 (small)
    [(np.int64(0), np.int64(1), np.float64(0.34552949035658936))]

    1, 2: Effect size: Hedges' g: 0.35 (small)

    Differences between male and female in normalized perioperative auc values: 0.0.


             median_fio2             
                  median   mean count
    sex_male                         
    False         0.4300 0.4540  2854
    True          0.4200 0.4391  2166
    p-value: 0.0

    ... Post hoc test...
         1    2
    1    -  ***
    2  ***    -
           1      2
    1 1.0000 0.0000
    2 0.0000 1.0000

    1, 2: Effect size: Hedges' g: 0.14 (very small)
    [(np.int64(0), np.int64(1), np.float64(0.13614155912595877))]

    1, 2: Effect size: Hedges' g: 0.14 (very small)
             median_horowitz               
                      median     mean count
    sex_male                               
    False           464.6450 461.0330  2854
    True            443.7500 444.5865  2166
    p-value: 0.0

    ... Post hoc test...
         1    2
    1    -  ***
    2  ***    -
           1      2
    1 1.0000 0.0000
    2 0.0000 1.0000

    1, 2: Effect size: Hedges' g: 0.17 (very small)
    [(np.int64(0), np.int64(1), np.float64(0.1747727872605381))]

    1, 2: Effect size: Hedges' g: 0.17 (very small)
             norm_auc_pf               
                  median     mean count
    sex_male                           
    0           464.9892 463.0380  2854
    1           444.9809 445.6756  2166
    p-value: 0.0

    ... Post hoc test...
         1    2
    1    -  ***
    2  ***    -
           1      2
    1 1.0000 0.0000
    2 0.0000 1.0000

    1, 2: Effect size: Hedges' g: 0.2 (small)
    [(np.int64(0), np.int64(1), np.float64(0.2033796496246368))]

    1, 2: Effect size: Hedges' g: 0.2 (small)

    Differences between male and female in normalized perioperative p/F auc values: 0.0.

## Differences in $paO_2$ per ASA

``` python
pvalue, ph_dunn, effect_sizes = compare_median("norm_auc_paO2", analysis_df, "asa", False)
print(
    f"\nDifferences between ASA classes in normalized perioperative auc values: {round(pvalue,4)}.\n\n"
)
```

        norm_auc_paO2               
               median     mean count
    asa                             
    I        227.9127 233.5727   255
    II       218.2228 222.7500  2143
    III      207.3290 213.7089  2175
    IV       221.4046 230.7889   414
    V        221.1013 252.5793    33

    ... Post hoc test...
         1    2    3    4   5
    1    -  ***  ***    *  NS
    2  ***    -  ***   NS  NS
    3  ***  ***    -  ***   *
    4    *   NS  ***    -  NS
    5   NS   NS    *   NS   -
           1      2      3      4      5
    1 1.0000 0.0006 0.0000 0.0189 0.9828
    2 0.0006 1.0000 0.0000 0.4863 0.2266
    3 0.0000 0.0000 1.0000 0.0000 0.0107
    4 0.0189 0.4863 0.0000 1.0000 0.3196
    5 0.9828 0.2266 0.0107 0.3196 1.0000

    1, 2: Effect size: Hedges' g: 0.26 (small)

    1, 3: Effect size: Hedges' g: 0.44 (small)

    1, 4: Effect size: Hedges' g: 0.05 (very small)

    1, 5: Effect size: Hedges' g: 0.4 (small)

    2, 3: Effect size: Hedges' g: 0.21 (small)

    2, 4: Effect size: Hedges' g: 0.17 (very small)

    2, 5: Effect size: Hedges' g: 0.7 (moderate)

    3, 4: Effect size: Hedges' g: 0.35 (small)

    3, 5: Effect size: Hedges' g: 0.85 (large)

    4, 5: Effect size: Hedges' g: 0.34 (small)
    [(np.int64(0), np.int64(1), np.float64(0.2567611267405748)), (np.int64(0), np.int64(2), np.float64(0.44229401648963435)), (np.int64(0), np.int64(3), np.float64(0.049922579440962574)), (np.int64(0), np.int64(4), np.float64(0.39730899052896174)), (np.int64(1), np.int64(2), np.float64(0.20683593801629482)), (np.int64(1), np.int64(3), np.float64(0.17448712522570137)), (np.int64(1), np.int64(4), np.float64(0.6950791720936668)), (np.int64(2), np.int64(3), np.float64(0.3527960229310341)), (np.int64(2), np.int64(4), np.float64(0.8471685293091694)), (np.int64(3), np.int64(4), np.float64(0.34065156669732916))]

    1, 2: Effect size: Hedges' g: 0.26 (small)

    1, 3: Effect size: Hedges' g: 0.44 (small)

    1, 4: Effect size: Hedges' g: 0.05 (very small)

    1, 5: Effect size: Hedges' g: 0.4 (small)

    2, 3: Effect size: Hedges' g: 0.21 (small)

    2, 4: Effect size: Hedges' g: 0.17 (very small)

    2, 5: Effect size: Hedges' g: 0.7 (moderate)

    3, 4: Effect size: Hedges' g: 0.35 (small)

    3, 5: Effect size: Hedges' g: 0.85 (large)

    4, 5: Effect size: Hedges' g: 0.34 (small)

    Differences between ASA classes in normalized perioperative auc values: 0.0.

Table 1

``` python
medians = (
    analysis_df
    .groupby("asa", observed=False)["norm_auc_paO2"]
    .median()
)
# medians.index = medians.index.map(lambda x: x - 1)  # remove if already 0–4
ph_dunn.index = medians.index
ph_dunn.columns = medians.index

effect_dict = {}

for i, j, eff in effect_sizes:
    effect_dict[(to_roman(int(i)+1), to_roman(int(j)+1))] = round(eff,2)
    effect_dict[(to_roman(int(j)+1), to_roman(int(i)+1))] = round(eff,2)  # symmetric


table1 = pd.DataFrame("", index=medians.index, columns=medians.index)

for i in medians.index:
    for j in medians.index:

        if i == j:
            table1.loc[i, j] = "-"
            continue

        p = ph_dunn.loc[i, j]
        eff = effect_dict[(i, j)]

        # format p-value
        if p < 0.0001:
            p_str = "<0.0001"
        else:
            p_str = f"{p:.4f}"

        table1.loc[i, j] = f"{p_str} ({eff:.2f})"

row_labels = [
    f"ASA {i} ({int(medians.loc[i])} mmHg)"
    for i in medians.index
]

table1.index = row_labels
table1.columns = row_labels

display(table1)
table1.to_csv("./data/out/table1.csv")
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | ASA I (227 mmHg) | ASA II (218 mmHg) | ASA III (207 mmHg) | ASA IV (221 mmHg) | ASA V (221 mmHg) |
|----|----|----|----|----|----|
| ASA I (227 mmHg) | \- | 0.0006 (0.26) | \<0.0001 (0.44) | 0.0189 (0.05) | 0.9828 (0.40) |
| ASA II (218 mmHg) | 0.0006 (0.26) | \- | \<0.0001 (0.21) | 0.4863 (0.17) | 0.2266 (0.70) |
| ASA III (207 mmHg) | \<0.0001 (0.44) | \<0.0001 (0.21) | \- | \<0.0001 (0.35) | 0.0107 (0.85) |
| ASA IV (221 mmHg) | 0.0189 (0.05) | 0.4863 (0.17) | \<0.0001 (0.35) | \- | 0.3196 (0.34) |
| ASA V (221 mmHg) | 0.9828 (0.40) | 0.2266 (0.70) | 0.0107 (0.85) | 0.3196 (0.34) | \- |

</div>

``` python
with open(Path(config.get('pickle').get('analysis')), 'wb') as f:
    pickle.dump(analysis_df, f)
```

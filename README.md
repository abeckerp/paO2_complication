# Perioperative oxygen exposure and its association to postoperative complications in neurosurgical patients
We imputed paO<sub>2</sub> values in five-minute intervals using previously published algorithms ([Gutmann et al., 2025](https://www.springermedizin.de/comparing-supervised-machine-learning-algorithms-for-the-predict/51420898)) in patients undergoing craniotomies and calculated the area under the curve for all measured and predicted paO<sub>2</sub> values. The AUC values were normalized by their intervention time.

## Authors
- [@abeckerp](https://github.com/abeckerp)

## Files
Due to data privacy issues, only rendered notebooks are provided.
- 1_Preprocessing: contains data cleaning, missing values, and exclusion criteria
- 2_Pre_ABG: prediction of paO<sub>2</sub> values before the first arterial blood gas analysis was available
- 3_Post_ABG: refitting and prediction of paO<sub>2</sub> values after the first arterial blood gas analysis was available
- 4_Calculate_AUC: calculation of paO<sub>2</sub> integral and normalization by intervention time
- 5_Patient_characteristics: Descriptive data
- 6_Postop_complications: Analysis of postoperative complications
- 7_Prevalence_ORs: prevalences and odds ratios of complications given a specific diagnosis
- 8_Confounding_bias: Analysis of potential confounders or bias

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

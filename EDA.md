# Exploratory Data Analysis — Smoking and Drinking Dataset

Generated with `python src/eda.py`. Figures are saved to `figures/eda/`.

## 1. Overview

| Property | Value |
|---|---|
| Rows | 991,346 |
| Columns | 24 (22 features + 2 targets) |
| Memory usage | 277.5 MB |
| Missing values | None |
| Duplicate rows | 26 |

**Features**

- **Demographics / body:** `sex` (categorical), `age`, `height`, `weight`, `waistline`
- **Senses:** `sight_left`, `sight_right`, `hear_left`, `hear_right` (hearing is coded 1 = normal, 2 = abnormal)
- **Blood pressure:** `SBP`, `DBP`
- **Blood tests:** `BLDS` (fasting blood sugar), `tot_chole`, `HDL_chole`, `LDL_chole`, `triglyceride`, `hemoglobin`, `serum_creatinine`, `SGOT_AST`, `SGOT_ALT`, `gamma_GTP`
- **Urine:** `urine_protein` (ordinal, 1–6)

**Targets**

- `SMK_stat_type_cd` — smoking status (1 = never, 2 = former, 3 = current)
- `DRK_YN` — drinker (Y/N)

`age`, `height` and `weight` are recorded in bins of 5 years, 5 cm and 5 kg.

## 2. Summary statistics

| Feature | Mean | Std | Min | 25% | 50% | 75% | Max |
|---|---|---|---|---|---|---|---|
| age | 47.61 | 14.18 | 20 | 35 | 45 | 60 | 85 |
| height | 162.24 | 9.28 | 130 | 155 | 160 | 170 | 190 |
| weight | 63.28 | 12.51 | 25 | 55 | 60 | 70 | 140 |
| waistline | 81.23 | 11.85 | 8 | 74.1 | 81 | 87.8 | **999** |
| sight_left | 0.98 | 0.61 | 0.1 | 0.7 | 1.0 | 1.2 | **9.9** |
| sight_right | 0.98 | 0.60 | 0.1 | 0.7 | 1.0 | 1.2 | **9.9** |
| hear_left | 1.03 | 0.17 | 1 | 1 | 1 | 1 | 2 |
| hear_right | 1.03 | 0.17 | 1 | 1 | 1 | 1 | 2 |
| SBP | 122.43 | 14.54 | 67 | 112 | 120 | 131 | 273 |
| DBP | 76.05 | 9.89 | 32 | 70 | 76 | 82 | 185 |
| BLDS | 100.42 | 24.18 | 25 | 88 | 96 | 105 | 852 |
| tot_chole | 195.56 | 38.66 | 30 | 169 | 193 | 219 | **2344** |
| HDL_chole | 56.94 | 17.24 | 1 | 46 | 55 | 66 | **8110** |
| LDL_chole | 113.04 | 35.84 | 1 | 89 | 111 | 135 | **5119** |
| triglyceride | 132.14 | 102.20 | 1 | 73 | 106 | 159 | **9490** |
| hemoglobin | 14.23 | 1.58 | 1 | 13.2 | 14.3 | 15.4 | 25 |
| urine_protein | 1.09 | 0.44 | 1 | 1 | 1 | 1 | 6 |
| serum_creatinine | 0.86 | 0.48 | 0.1 | 0.7 | 0.8 | 1.0 | **98** |
| SGOT_AST | 25.99 | 23.49 | 1 | 19 | 23 | 28 | **9999** |
| SGOT_ALT | 25.76 | 26.31 | 1 | 15 | 20 | 29 | **7210** |
| gamma_GTP | 37.14 | 50.42 | 1 | 16 | 23 | 39 | 999 |

Bold maximums are physiologically implausible and are most likely data-entry errors or placeholder codes.

![Numeric distributions](figures/eda/numeric_distributions.png)

*Histograms are clipped at the 0.5th and 99.5th percentiles so extreme values do not flatten them.*

## 3. Target distribution

| Smoking status | Count | Percent |
|---|---|---|
| Never (1) | 602,441 | 60.77% |
| Former (2) | 174,951 | 17.65% |
| Current (3) | 213,954 | 21.58% |

| Drinker | Count | Percent |
|---|---|---|
| N | 495,858 | 50.02% |
| Y | 495,488 | 49.98% |

- `DRK_YN` is perfectly balanced.
- `SMK_stat_type_cd` is imbalanced: "never" is about 3× as common as each of the other classes. Consider stratified splits and class weights or macro-averaged metrics.

**Smoking status vs. drinking (row %)**

| Smoking status | N | Y |
|---|---|---|
| Never | 64.6 | 35.4 |
| Former | 31.1 | 68.9 |
| Current | 24.5 | 75.5 |

The two targets are strongly related: current smokers are more than twice as likely to drink as people who have never smoked.

![Target distribution](figures/eda/target_distribution.png)

## 4. Data quality and outliers

**Missing values:** none. **Duplicates:** 26 rows (negligible; can be dropped).

**Outliers by the 1.5 × IQR rule**

| Feature | Outliers | Percent | 99.9th pct | Max |
|---|---|---|---|---|
| gamma_GTP | 94,142 | 9.50% | 658.0 | 999 |
| SGOT_ALT | 72,928 | 7.36% | 228.0 | 7210 |
| SGOT_AST | 67,614 | 6.82% | 201.7 | 9999 |
| BLDS | 64,202 | 6.48% | 321.0 | 852 |
| urine_protein | 56,171 | 5.67% | 5.0 | 6 |
| triglyceride | 54,104 | 5.46% | 1056.0 | 9490 |
| hear_left | 31,222 | 3.15% | 2.0 | 2 |
| hear_right | 30,212 | 3.05% | 2.0 | 2 |
| weight | 19,221 | 1.94% | 110.0 | 140 |
| SBP | 15,836 | 1.60% | 184.0 | 273 |
| HDL_chole | 13,858 | 1.40% | 121.0 | 8110 |
| DBP | 13,122 | 1.32% | 119.0 | 185 |
| sight_left | 11,582 | 1.17% | 9.9 | 9.9 |
| tot_chole | 11,298 | 1.14% | 350.0 | 2344 |
| sight_right | 10,495 | 1.06% | 9.9 | 9.9 |
| LDL_chole | 10,098 | 1.02% | 250.0 | 5119 |
| hemoglobin | 9,760 | 0.98% | 18.5 | 25 |
| serum_creatinine | 8,005 | 0.81% | 2.9 | 98 |
| waistline | 4,417 | 0.45% | 115.0 | 999 |
| height | 86 | 0.01% | 185.0 | 190 |
| age | 0 | 0.00% | 85.0 | 85 |

Observations:

- **Placeholder codes:** `waistline == 999` (57 rows) and `sight == 9.9` (3,118 left, 3,111 right — likely a code for blindness). These should be treated as missing or as a separate category, not as numeric values.
- **Extreme lab values:** the 99.9th percentile is far below the maximum for `SGOT_AST`, `SGOT_ALT`, `HDL_chole`, `LDL_chole`, `tot_chole`, `triglyceride` and `serum_creatinine`. Clipping at the 0.1st/99.9th percentile (winsorizing) is a reasonable fix.
- **Right-skewed liver/lipid markers:** `gamma_GTP`, `SGOT_ALT`, `SGOT_AST` and `triglyceride` have long right tails; a log transform should make them closer to normal.
- `hear_*` and `urine_protein` "outliers" are just the minority levels of ordinal variables and should not be removed.

## 5. Correlations

Spearman correlation was used (robust to skew and outliers). `sex` is encoded as Male = 1 and `DRK_YN` as Y = 1.

![Correlation matrix](figures/eda/correlation_matrix.png)

**Top correlations with `SMK_stat_type_cd`**

| Feature | r |
|---|---|
| sex | 0.621 |
| height | 0.500 |
| hemoglobin | 0.495 |
| gamma_GTP | 0.404 |
| weight | 0.400 |
| DRK_YN | 0.361 |
| serum_creatinine | 0.360 |
| waistline | 0.283 |
| SGOT_ALT | 0.246 |
| triglyceride | 0.244 |

**Top correlations with `DRK_YN`**

| Feature | r |
|---|---|
| height | 0.379 |
| sex | 0.369 |
| SMK_stat_type_cd | 0.361 |
| hemoglobin | 0.318 |
| gamma_GTP | 0.299 |
| age | −0.283 |
| weight | 0.273 |
| serum_creatinine | 0.188 |
| sight_left | 0.165 |
| sight_right | 0.163 |

Many of the top features (height, hemoglobin, weight, serum creatinine) are also strongly tied to sex, so part of their predictive power is a proxy for sex. `gamma_GTP` is the most interesting lab marker — it is a known indicator of alcohol use and liver stress.

**Highly correlated feature pairs (|r| > 0.7)**

| Feature A | Feature B | r |
|---|---|---|
| tot_chole | LDL_chole | 0.887 |
| weight | waistline | 0.781 |
| sex | height | 0.746 |
| SGOT_AST | SGOT_ALT | 0.731 |
| SBP | DBP | 0.725 |
| sex | hemoglobin | 0.717 |
| sight_left | sight_right | 0.717 |

These pairs are redundant and are candidates for removal or combination (e.g. mean of left/right sight) during feature selection, particularly for linear models.

## 6. Features by target

![Features by drinking status](figures/eda/features_by_DRK_YN.png)

![Features by smoking status](figures/eda/features_by_smoking.png)

*Boxplots use a random sample of 100,000 rows, with values above the 99th percentile excluded.*

- Drinkers are younger (median 45 vs. 50), taller, heavier and have higher hemoglobin.
- `gamma_GTP` shows the clearest separation among lab values: drinkers have a noticeably higher median and a much wider spread.
- `SGOT_ALT`, `waistline` and `HDL_chole` differ only slightly between groups.

## 7. Demographics

| Sex | Drinker (%) | Current smoker (%) |
|---|---|---|
| Female | 30.4 | 3.5 |
| Male | 67.3 | 37.6 |

| Age | Drinker (%) | Current smoker (%) | n |
|---|---|---|---|
| 20 | 63.9 | 24.6 | 21,971 |
| 25 | 67.2 | 25.6 | 64,370 |
| 30 | 66.2 | 28.2 | 77,600 |
| 35 | 66.4 | 33.3 | 84,726 |
| 40 | 60.0 | 28.1 | 130,385 |
| 45 | 56.0 | 25.1 | 118,355 |
| 50 | 49.5 | 21.2 | 129,434 |
| 55 | 43.1 | 17.9 | 111,223 |
| 60 | 36.1 | 14.7 | 106,063 |
| 65 | 30.7 | 11.5 | 52,961 |
| 70 | 24.3 | 8.2 | 50,666 |
| 75 | 18.5 | 6.4 | 25,333 |
| 80 | 14.2 | 4.8 | 14,968 |
| 85 | 8.7 | 2.6 | 3,291 |

![Demographics](figures/eda/demographics.png)

- Smoking is overwhelmingly male: about 1 in 3 men currently smoke vs. 1 in 30 women.
- Both drinking and smoking peak around age 25–35 and decline steadily after that.

## 8. Implications for preprocessing and modelling

1. **Drop the 26 duplicate rows.**
2. **Handle placeholder codes:** treat `waistline == 999` as missing (impute) and `sight == 9.9` as missing or flag it with a binary indicator.
3. **Clip extreme lab values** at the 0.1st/99.9th percentile, then **log-transform** the skewed markers (`gamma_GTP`, `SGOT_AST`, `SGOT_ALT`, `triglyceride`, `BLDS`).
4. **Encode** `sex` and `DRK_YN` as binary; keep `hear_*` and `urine_protein` as ordinal.
5. **Scale** numeric features for distance- and gradient-based models (SVM, kNN, neural networks).
6. **Address class imbalance** for smoking (stratified splits, class weights, macro-F1).
7. **Feature selection:** consider removing one feature from each highly correlated pair, and be aware that sex dominates both targets.
8. **Feature engineering ideas:** BMI from height/weight, AST/ALT ratio, mean sight/hearing across both sides, non-HDL cholesterol.

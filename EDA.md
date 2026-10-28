# Exploratory Data Analysis — Predicting Drinkers

Generated with `python src/eda.py`. Figures are saved to `figures/eda/`.

**Problem statement:** binary classification of whether a person is a drinker (`DRK_YN`) from health-screening measurements. Smoking status (`SMK_stat_type_cd`) is used as an input feature, not as a second target.

## 1. Overview

| Property | Value |
|---|---|
| Rows | 991,346 |
| Columns | 24 (23 features + 1 target) |
| Memory usage | 277.5 MB |
| Missing values | None |
| Duplicate rows | 26 |

**Target**

- `DRK_YN` — drinker (Y/N)

**Features**

- **Demographics / body:** `sex` (categorical), `age`, `height`, `weight`, `waistline`
- **Lifestyle:** `SMK_stat_type_cd` — smoking status (1 = never, 2 = former, 3 = current), categorical
- **Senses:** `sight_left`, `sight_right`, `hear_left`, `hear_right` (hearing is coded 1 = normal, 2 = abnormal)
- **Blood pressure:** `SBP`, `DBP`
- **Blood tests:** `BLDS` (fasting blood sugar), `tot_chole`, `HDL_chole`, `LDL_chole`, `triglyceride`, `hemoglobin`, `serum_creatinine`, `SGOT_AST`, `SGOT_ALT`, `gamma_GTP`
- **Urine:** `urine_protein` (ordinal, 1–6)

`age`, `height` and `weight` are recorded in bins of 5 years, 5 cm and 5 kg.

## 2. Target distribution

| Drinker | Count | Percent |
|---|---|---|
| N | 495,858 | 50.02% |
| Y | 495,488 | 49.98% |

`DRK_YN` is almost perfectly balanced, so no resampling or class weighting is needed. Accuracy is a meaningful metric, and ROC-AUC/F1 can be reported alongside it. Splits should still be stratified to keep the balance identical across train, validation and test.

![Target distribution](figures/eda/target_distribution.png)

## 3. Smoking status as a feature

| Smoking status | n | Share | Drinker (%) |
|---|---|---|---|
| Never | 602,441 | 60.8% | 35.4 |
| Former | 174,951 | 17.6% | 68.9 |
| Current | 213,954 | 21.6% | 75.5 |

Smoking is one of the strongest single predictors of drinking: former and current smokers are about twice as likely to drink as people who have never smoked.

Smoking is, however, heavily tied to sex:

| Sex | Never (%) | Former (%) | Current (%) |
|---|---|---|---|
| Female | 94.2 | 2.3 | 3.5 |
| Male | 31.3 | 31.2 | 37.6 |

So part of the smoking effect could just be a sex effect. Looking within each sex shows that smoking carries real information on its own:

| Sex | Never | Former | Current |
|---|---|---|---|
| Female | 28.5 | 59.4 | 62.1 |
| Male | 54.0 | 69.5 | 76.6 |

*Drinker rate (%) by smoking status within each sex.*

- The effect remains in both sexes, and is strongest among women (28% → 62%), though only ~6% of women have ever smoked.
- Former and current smokers have similar drinker rates. The main split is **never vs. ever smoked**, not current vs. former.
- The gap holds at every age (see section 8), so it is not explained by age either: current smokers are younger (median 40 vs. 50), but smokers drink more than never-smokers within every age group.

![Smoking vs drinking](figures/eda/smoking_vs_drinking.png)

**Encoding:** the three levels are not equally spaced in their effect on drinking (never → former is a large jump, former → current a small one), so one-hot encoding with "never" as reference is more appropriate than treating the code as a number. This is what `data_pipeline.py` does.

## 4. Summary statistics (numeric features)

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

## 5. Data quality and outliers

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

- **Placeholder codes:** `waistline == 999` (57 rows) and `sight == 9.9` (3,118 left, 3,111 right — likely a code for blindness). These should be treated as missing, not as numeric values. The drinker rate among `sight == 9.9` rows is 34.8%, below the 50% average, so a missing-indicator could carry a little signal; the 57 `waistline == 999` rows are too few to matter.
- **Extreme lab values:** the 99.9th percentile is far below the maximum for `SGOT_AST`, `SGOT_ALT`, `HDL_chole`, `LDL_chole`, `tot_chole`, `triglyceride` and `serum_creatinine`. Clipping at the 0.1st/99.9th percentile (winsorizing) is a reasonable fix.
- **Right-skewed liver/lipid markers:** `gamma_GTP`, `SGOT_ALT`, `SGOT_AST` and `triglyceride` have long right tails; a log transform should make them closer to normal. This matters for drinking in particular, since `gamma_GTP` is one of the most informative lab features (section 7) and its high tail is mostly drinkers.
- `hear_*` and `urine_protein` "outliers" are just the minority levels of ordinal variables and should not be removed.

## 6. Correlations

Spearman correlation was used (robust to skew and outliers). `sex` is encoded as Male = 1, `DRK_YN` as Y = 1, and smoking status as its ordinal code (never < former < current).

![Correlation matrix](figures/eda/correlation_matrix.png)

**Correlation with `DRK_YN`**

| Feature | r | | Feature | r |
|---|---|---|---|---|
| height | 0.379 | | DBP | 0.099 |
| sex | 0.369 | | triglyceride | 0.086 |
| SMK_stat_type_cd | 0.361 | | SGOT_AST | 0.063 |
| hemoglobin | 0.318 | | hear_left | −0.058 |
| gamma_GTP | 0.299 | | hear_right | −0.058 |
| age | −0.283 | | HDL_chole | 0.043 |
| weight | 0.273 | | LDL_chole | −0.042 |
| serum_creatinine | 0.188 | | SBP | 0.038 |
| sight_left | 0.165 | | BLDS | 0.037 |
| sight_right | 0.163 | | tot_chole | 0.021 |
| waistline | 0.126 | | urine_protein | 0.005 |
| SGOT_ALT | 0.100 | | | |

- The three strongest predictors are **height, sex and smoking status**, all with r ≈ 0.36–0.38. No single feature is close to determining the target, so models will need to combine many weak signals.
- Height, hemoglobin, weight and serum creatinine are all strongly tied to sex (r = 0.75 for sex–height, 0.72 for sex–hemoglobin), so much of their correlation with drinking is a proxy for sex (see section 7).
- **Age** is the strongest negative predictor: drinking declines steadily with age.
- `gamma_GTP` is the most interesting lab marker — it is a known clinical indicator of alcohol use and liver stress.
- Blood pressure, blood sugar, cholesterol and urine protein are nearly uncorrelated with drinking on their own.

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

Smoking status is not in any highly correlated pair (its strongest link is sex, r = 0.62), so it adds information that is not already present in the other features. The pairs above are redundant and are candidates for removal or combination (e.g. mean of left/right sight) during feature selection, particularly for linear models.

## 7. Features by drinking status

**Class separation per feature** (values clipped at the 0.1st/99.9th percentile; Cohen's d = standardized difference in means, Y − N)

| Feature | Median N | Median Y | d (all) | d (female) | d (male) |
|---|---|---|---|---|---|
| height | 160 | 165 | 0.81 | 0.48 | 0.28 |
| hemoglobin | 13.7 | 14.9 | 0.63 | 0.06 | 0.25 |
| age | 50 | 45 | −0.59 | −0.88 | −0.40 |
| weight | 60 | 65 | 0.55 | 0.03 | 0.22 |
| gamma_GTP | 19 | 29 | 0.44 | 0.08 | 0.41 |
| serum_creatinine | 0.8 | 0.9 | 0.30 | −0.14 | −0.15 |
| waistline | 80 | 82.5 | 0.24 | −0.28 | 0.07 |
| triglyceride | 102 | 112 | 0.22 | −0.20 | 0.21 |
| DBP | 75 | 78 | 0.20 | −0.09 | 0.20 |
| HDL_chole | 54 | 56 | 0.10 | 0.45 | 0.34 |
| SBP | 120 | 121 | 0.07 | −0.27 | 0.10 |

![Features by drinking status](figures/eda/features_by_DRK_YN.png)

*Boxplots use a random sample of 100,000 rows split by sex, with values above the 99th percentile excluded.*

Splitting by sex changes the picture considerably:

- **Sex proxies shrink:** `hemoglobin`, `weight` and `serum_creatinine` separate drinkers well overall, but within each sex the effect is small or even reversed. Most of their signal comes from men being both heavier/higher-hemoglobin and more likely to drink.
- **Age is the strongest within-sex signal**, especially for women (d = −0.88): female drinkers are much younger than female non-drinkers.
- **`gamma_GTP` is a male-specific signal:** a clear shift among men (d = 0.41), almost none among women. This is consistent with heavier drinking in men raising liver enzymes.
- **`HDL_chole` is hidden overall but strong within sex** (d ≈ 0.34–0.45). Drinkers have higher HDL in both sexes, but because women have higher HDL and drink less, the two effects cancel in the pooled data. Alcohol is known to raise HDL, so this is a plausible real effect.
- **Sign flips for women** (`waistline`, `SBP`, `triglyceride`): female drinkers have lower values, most likely because they are younger, not because of drinking itself.

The takeaway is that sex and age interact strongly with the other features. Tree-based models and neural networks can learn these interactions directly; linear models will need interaction terms (e.g. polynomial features or `sex × feature`) to use them.

## 8. Demographics

| Sex | Drinker (%) | n |
|---|---|---|
| Female | 30.4 | 464,931 |
| Male | 67.3 | 526,415 |

| Age | Drinker (%) | n |
|---|---|---|
| 20 | 63.9 | 21,971 |
| 25 | 67.2 | 64,370 |
| 30 | 66.2 | 77,600 |
| 35 | 66.4 | 84,726 |
| 40 | 60.0 | 130,385 |
| 45 | 56.0 | 118,355 |
| 50 | 49.5 | 129,434 |
| 55 | 43.1 | 111,223 |
| 60 | 36.1 | 106,063 |
| 65 | 30.7 | 52,961 |
| 70 | 24.3 | 50,666 |
| 75 | 18.5 | 25,333 |
| 80 | 14.2 | 14,968 |
| 85 | 8.7 | 3,291 |

![Demographics](figures/eda/demographics.png)

- Men are more than twice as likely to drink as women (67% vs. 30%).
- Drinking peaks around age 25–35 and declines steadily after that. The decline starts much earlier for women: the female drinker rate falls from about 60% at age 20–25, while the male rate stays above 70% until about age 50.
- Former and current smokers have a drinker rate 20–25 percentage points above never-smokers at every age (right panel), confirming that smoking is informative beyond age.

## 9. Implications for preprocessing and modelling

1. **Drop the 26 duplicate rows.**
2. **Handle placeholder codes:** treat `waistline == 999` and `sight == 9.9` as missing and impute (median from the training set), optionally with a missing-indicator for sight.
3. **Clip extreme lab values** at the 0.1st/99.9th percentile, then **log-transform** the skewed markers (`gamma_GTP`, `SGOT_AST`, `SGOT_ALT`, `triglyceride`, `BLDS`, `serum_creatinine`).
4. **Encode:** `DRK_YN` as Y = 1; `sex` as binary; **smoking status one-hot with "never" as reference** (its levels are not evenly spaced in effect); keep `hear_*` and `urine_protein` as ordinal.
5. **Scale** numeric features for distance- and gradient-based models (logistic regression, SVM, kNN, neural networks).
6. **No class rebalancing needed** — the target is 50/50. Use stratified splits and report accuracy, F1 and ROC-AUC.
7. **Interactions matter:** sex and age change how other features relate to drinking. Expect non-linear models (trees, ensembles, neural networks) to outperform a plain linear model; for logistic regression, polynomial/interaction features should help.
8. **Feature selection:** consider removing one feature from each highly correlated pair. Keep smoking status — it is among the top three predictors and is not redundant with any other feature.
9. **Feature engineering ideas:** BMI from height/weight, AST/ALT ratio, non-HDL cholesterol, pulse pressure, mean sight/hearing across both sides, and a binary "ever smoked" flag (never vs. former/current captures most of the smoking signal).

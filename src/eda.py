import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import FIGURES_DIR, RANDOM_STATE
from data import load_dataset
from plotting import save_figure

FIGURE_DIR = FIGURES_DIR / "eda"
TARGET = "DRK_YN"
# Smoking status is a categorical predictor, not a numeric measurement
SMOKING = "SMK_stat_type_cd"
SMOKING_LABELS = {1.0: "Never", 2.0: "Former", 3.0: "Current"}
PLOT_SAMPLE_SIZE = 100_000


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def numeric_features(df):
    return df.select_dtypes(include="number").drop(columns=[SMOKING])


def drinker_rate(s):
    return (s == "Y").mean() * 100


# Overview: shape, types, memory usage
def overview(df):
    section("OVERVIEW")
    print(f"Rows: {df.shape[0]:,}  Columns: {df.shape[1]}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print("\nData types:")
    print(df.dtypes.to_string())


# Missing values and duplicate rows
def data_quality(df):
    section("DATA QUALITY")
    missing = df.isna().sum()
    print("Missing values per column:")
    print(missing[missing > 0].to_string() if missing.any() else "  None")
    print(f"\nDuplicate rows: {df.duplicated().sum():,}")


def summary_statistics(df):
    section("SUMMARY STATISTICS (numeric)")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(numeric_features(df).describe().T.round(2).to_string())

    section("CATEGORICAL COLUMNS")
    for col in ["sex", SMOKING]:
        print(f"\n{col}:")
        print(df[col].value_counts().sort_index().to_string())


def target_distribution(df):
    section(f"TARGET DISTRIBUTION ({TARGET})")
    counts = df[TARGET].value_counts().sort_index()
    share = (counts / len(df) * 100).round(2)
    print(pd.DataFrame({"count": counts, "percent": share}).to_string())

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x=df[TARGET], order=["N", "Y"], ax=ax)
    ax.set_title("Drinker")
    ax.set_xlabel("")
    save_figure(fig, FIGURE_DIR / "target_distribution.png")


# Smoking status as a predictor of drinking, and how much of it is explained by sex
def smoking_feature(df):
    section("SMOKING STATUS AS A FEATURE")
    smoking = df[SMOKING].map(SMOKING_LABELS)
    order = list(SMOKING_LABELS.values())

    print("Drinker rate by smoking status (%):")
    print(df.groupby(smoking)[TARGET].agg(drinker=drinker_rate, n="size").reindex(order).round(1).to_string())

    print("\nSmoking status by sex (row %):")
    print((pd.crosstab(df["sex"], smoking, normalize="index") * 100)[order].round(1).to_string())

    print("\nDrinker rate by smoking status within each sex (%):")
    by_sex = df.groupby(["sex", smoking])[TARGET].agg(drinker_rate).unstack()[order]
    print(by_sex.round(1).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(x=smoking, order=order, ax=axes[0])
    axes[0].set_title("Smoking status")
    axes[0].set_xlabel("")
    by_sex.T.plot(kind="bar", ax=axes[1], rot=0)
    axes[1].set_title("Drinker rate (%) by smoking status and sex")
    axes[1].set_xlabel("")
    save_figure(fig, FIGURE_DIR / "smoking_vs_drinking.png")


# Outliers via the 1.5 * IQR rule, plus values far above the 99.9th percentile
def outliers(df):
    section("OUTLIERS (1.5 * IQR rule)")
    numeric = numeric_features(df)
    q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
    iqr = q3 - q1
    mask = (numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)
    result = pd.DataFrame({
        "n_outliers": mask.sum(),
        "percent": (mask.mean() * 100).round(2),
        "p99.9": numeric.quantile(0.999).round(1),
        "max": numeric.max(),
    }).sort_values("percent", ascending=False)
    print(result.to_string())

    # Values that look like placeholder codes rather than real measurements
    print("\nSuspicious values:")
    print(f"  waistline == 999: {(df['waistline'] == 999).sum():,}")
    print(f"  sight == 9.9 (likely blind code): left {(df['sight_left'] == 9.9).sum():,}, right {(df['sight_right'] == 9.9).sum():,}")
    print(f"  drinker rate where waistline == 999: {drinker_rate(df.loc[df['waistline'] == 999, TARGET]):.1f}%")
    print(f"  drinker rate where sight_left == 9.9: {drinker_rate(df.loc[df['sight_left'] == 9.9, TARGET]):.1f}%")


# Bin edges aligned to the column's measurement step (e.g. 5 for age, 1 for waistline, 0.1 for hemoglobin).
# With a fixed bin count, the bin width drifts against that grid and some bins catch no values, which shows up as fake dips.
def aligned_bins(values, max_bins=50):
    step = next((s for s in [10, 5, 1, 0.5, 0.1, 0.01] if np.isclose(values / s, np.round(values / s)).mean() >= 0.8), None)
    if step is None:
        return max_bins
    lo, hi = values.min(), values.max()
    width = step * max(1, np.ceil((hi - lo) / step / max_bins))
    return np.arange(lo - step / 2, hi + width, width)


def plot_distributions(df):
    numeric = numeric_features(df)
    # Clip at the 0.5/99.5 percentiles so extreme outliers do not flatten the histograms
    clipped = numeric.clip(numeric.quantile(0.005), numeric.quantile(0.995), axis=1)
    n_cols = 4
    n_rows = int(np.ceil(len(numeric.columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    for ax, col in zip(axes.flat, numeric.columns):
        ax.hist(clipped[col], bins=aligned_bins(clipped[col]), color="steelblue")
        ax.set_title(col)
    for ax in axes.flat[len(numeric.columns):]:
        ax.axis("off")
    save_figure(fig, FIGURE_DIR / "numeric_distributions.png")


def correlations(df):
    section("CORRELATIONS")
    encoded = df.copy()
    encoded["sex"] = (encoded["sex"] == "Male").astype(int)
    encoded[TARGET] = (encoded[TARGET] == "Y").astype(int)
    # Never < former < current, so the code is usable as an ordinal in a rank correlation
    corr = encoded.corr(method="spearman")

    print(f"\nSpearman correlation with {TARGET} (all features, by |r|):")
    target_corr = corr[TARGET].drop(TARGET)
    print(target_corr.reindex(target_corr.abs().sort_values(ascending=False).index).round(3).to_string())

    features = corr.drop(index=TARGET, columns=TARGET)
    upper = features.where(np.triu(np.ones(features.shape, dtype=bool), k=1)).stack()
    print("\nStrongly correlated feature pairs (|r| > 0.7):")
    print(upper[upper.abs() > 0.7].sort_values(key=abs, ascending=False).round(3).to_string())

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax)
    ax.set_title("Spearman correlation matrix")
    save_figure(fig, FIGURE_DIR / "correlation_matrix.png")


# Per-feature class separation: medians per class and standardized mean difference (Cohen's d)
def feature_separation(df):
    section(f"FEATURE SEPARATION BY {TARGET}")
    numeric = numeric_features(df)
    clipped = numeric.clip(numeric.quantile(0.001), numeric.quantile(0.999), axis=1)
    drinker = df[TARGET] == "Y"
    yes, no = clipped[drinker], clipped[~drinker]
    pooled_std = np.sqrt((yes.var() + no.var()) / 2)
    result = pd.DataFrame({
        "median_N": no.median(),
        "median_Y": yes.median(),
        "cohens_d": (yes.mean() - no.mean()) / pooled_std,
    })
    print(result.reindex(result["cohens_d"].abs().sort_values(ascending=False).index).round(3).to_string())

    # Same comparison within each sex, to see what is left after removing the sex effect
    print("\nCohen's d within each sex:")
    within = {}
    for sex, group in clipped.groupby(df["sex"]):
        g_drinker = drinker[group.index]
        g_yes, g_no = group[g_drinker], group[~g_drinker]
        within[sex] = (g_yes.mean() - g_no.mean()) / np.sqrt((g_yes.var() + g_no.var()) / 2)
    within = pd.DataFrame(within)
    print(within.reindex(within.abs().max(axis=1).sort_values(ascending=False).index).round(3).to_string())


def plot_by_target(df):
    sample = df.sample(min(PLOT_SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE)
    features = ["age", "height", "weight", "hemoglobin", "gamma_GTP", "serum_creatinine", "HDL_chole", "SGOT_ALT"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, col in zip(axes.flat, features):
        upper = sample[col].quantile(0.99)
        sns.boxplot(data=sample[sample[col] <= upper], x="sex", y=col, hue=TARGET, hue_order=["N", "Y"],
                    order=["Female", "Male"], showfliers=False, ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")
    save_figure(fig, FIGURE_DIR / f"features_by_{TARGET}.png")


# Demographics: drinker rate by sex, age and smoking status
def demographics(df):
    section("DEMOGRAPHICS")
    print("Drinker rate by sex (%):")
    print(df.groupby("sex")[TARGET].agg(drinker=drinker_rate, n="size").round(1).to_string())
    print("\nDrinker rate by age (%):")
    print(df.groupby("age")[TARGET].agg(drinker=drinker_rate, n="size").round(1).to_string())

    smoking = df[SMOKING].map(SMOKING_LABELS)
    print("\nMedian age by smoking status:")
    print(df.groupby(smoking)["age"].median().reindex(SMOKING_LABELS.values()).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df.groupby(["age", "sex"])[TARGET].agg(drinker_rate).unstack().plot(
        marker="o", ax=axes[0], title="Drinker rate (%) by age and sex")
    df.groupby(["age", smoking])[TARGET].agg(drinker_rate).unstack()[list(SMOKING_LABELS.values())].plot(
        marker="o", ax=axes[1], title="Drinker rate (%) by age and smoking status")
    save_figure(fig, FIGURE_DIR / "demographics.png")


def main():
    df = load_dataset()
    overview(df)
    data_quality(df)
    summary_statistics(df)
    target_distribution(df)
    smoking_feature(df)
    outliers(df)
    correlations(df)
    feature_separation(df)
    plot_distributions(df)
    plot_by_target(df)
    demographics(df)
    print(f"\nFigures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    main()

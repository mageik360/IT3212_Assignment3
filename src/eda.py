from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data import load_dataset

FIGURE_DIR = Path(__file__).resolve().parent.parent / "figures" / "eda"
TARGETS = ["SMK_stat_type_cd", "DRK_YN"]
SMOKING_LABELS = {1.0: "Never", 2.0: "Former", 3.0: "Current"}
PLOT_SAMPLE_SIZE = 100_000
RANDOM_STATE = 42


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def save_figure(fig, name):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / name, dpi=120)
    plt.close(fig)


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
        print(df.describe().T.round(2).to_string())

    section("CATEGORICAL COLUMNS")
    for col in df.select_dtypes(include="object").columns:
        print(f"\n{col}:")
        print(df[col].value_counts().to_string())


def target_distribution(df):
    section("TARGET DISTRIBUTION")
    for target in TARGETS:
        counts = df[target].value_counts().sort_index()
        share = (counts / len(df) * 100).round(2)
        print(f"\n{target}:")
        print(pd.DataFrame({"count": counts, "percent": share}).to_string())

    print("\nSmoking status vs drinking (row %):")
    print((pd.crosstab(df["SMK_stat_type_cd"].map(SMOKING_LABELS), df["DRK_YN"], normalize="index") * 100).round(1).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.countplot(x=df["SMK_stat_type_cd"].map(SMOKING_LABELS), order=SMOKING_LABELS.values(), ax=axes[0])
    axes[0].set_title("Smoking status")
    axes[0].set_xlabel("")
    sns.countplot(x=df["DRK_YN"], order=["N", "Y"], ax=axes[1])
    axes[1].set_title("Drinker")
    axes[1].set_xlabel("")
    save_figure(fig, "target_distribution.png")


# Outliers via the 1.5 * IQR rule, plus values far above the 99.9th percentile
def outliers(df):
    section("OUTLIERS (1.5 * IQR rule)")
    numeric = df.select_dtypes(include="number").drop(columns=["SMK_stat_type_cd"])
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


def plot_distributions(df):
    numeric = df.select_dtypes(include="number").drop(columns=["SMK_stat_type_cd"])
    # Clip at the 0.5/99.5 percentiles so extreme outliers do not flatten the histograms
    clipped = numeric.clip(numeric.quantile(0.005), numeric.quantile(0.995), axis=1)
    n_cols = 4
    n_rows = int(np.ceil(len(numeric.columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    for ax, col in zip(axes.flat, numeric.columns):
        ax.hist(clipped[col], bins=50, color="steelblue")
        ax.set_title(col)
    for ax in axes.flat[len(numeric.columns):]:
        ax.axis("off")
    save_figure(fig, "numeric_distributions.png")


def correlations(df):
    section("CORRELATIONS")
    encoded = df.copy()
    encoded["sex"] = (encoded["sex"] == "Male").astype(int)
    encoded["DRK_YN"] = (encoded["DRK_YN"] == "Y").astype(int)
    corr = encoded.corr(method="spearman")

    for target in TARGETS:
        print(f"\nSpearman correlation with {target} (top 10 by |r|):")
        target_corr = corr[target].drop(target)
        print(target_corr.reindex(target_corr.abs().sort_values(ascending=False).index).head(10).round(3).to_string())

    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    print("\nStrongly correlated feature pairs (|r| > 0.7):")
    print(upper[upper.abs() > 0.7].sort_values(key=abs, ascending=False).round(3).to_string())

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax)
    ax.set_title("Spearman correlation matrix")
    save_figure(fig, "correlation_matrix.png")


def plot_by_target(df):
    sample = df.sample(min(PLOT_SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE)
    sample = sample.assign(smoking=sample["SMK_stat_type_cd"].map(SMOKING_LABELS))
    features = ["age", "height", "weight", "waistline", "hemoglobin", "gamma_GTP", "SGOT_ALT", "HDL_chole"]

    for target, hue_order in [("DRK_YN", ["N", "Y"]), ("smoking", list(SMOKING_LABELS.values()))]:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for ax, col in zip(axes.flat, features):
            upper = sample[col].quantile(0.99)
            sns.boxplot(data=sample[sample[col] <= upper], x=target, y=col, order=hue_order, showfliers=False, ax=ax)
            ax.set_title(col)
            ax.set_xlabel("")
        save_figure(fig, f"features_by_{target}.png")

    # Demographics: target rate by sex and age group
    section("DEMOGRAPHICS")
    print("Target rates by sex (%):")
    print(df.groupby("sex").agg(
        drinker=("DRK_YN", lambda s: (s == "Y").mean() * 100),
        current_smoker=("SMK_stat_type_cd", lambda s: (s == 3).mean() * 100),
    ).round(1).to_string())
    print("\nTarget rates by age (%):")
    print(df.groupby("age").agg(
        drinker=("DRK_YN", lambda s: (s == "Y").mean() * 100),
        current_smoker=("SMK_stat_type_cd", lambda s: (s == 3).mean() * 100),
        n=("age", "size"),
    ).round(1).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    drink_rate = df.assign(drinker=df["DRK_YN"] == "Y").groupby(["age", "sex"])["drinker"].mean().unstack()
    drink_rate.plot(marker="o", ax=axes[0], title="Drinker rate by age and sex")
    smoke_rate = df.assign(smoker=df["SMK_stat_type_cd"] == 3).groupby(["age", "sex"])["smoker"].mean().unstack()
    smoke_rate.plot(marker="o", ax=axes[1], title="Current smoker rate by age and sex")
    save_figure(fig, "demographics.png")


def main():
    df = load_dataset()
    overview(df)
    data_quality(df)
    summary_statistics(df)
    target_distribution(df)
    outliers(df)
    correlations(df)
    plot_distributions(df)
    plot_by_target(df)
    print(f"\nFigures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    main()

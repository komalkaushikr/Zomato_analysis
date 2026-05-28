"""
visualizer.py
-------------
Reusable plot functions extracted from notebooks 01_eda.ipynb and 03_insights.ipynb.

Every function signature:
    plot_*(df: pd.DataFrame, save: bool = True, fig_dir: Path | None = None)
        -> matplotlib.figure.Figure

Usage
-----
    from src.visualizer import *
    import pandas as pd

    df = pd.read_csv("data/processed/zomato_clean.csv")
    plot_rating_distribution(df)
    plot_top_locations(df, n=15)
    # ... etc.
"""

import pathlib
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ── global defaults ────────────────────────────────────────────────────────────
PALETTE  = "Set2"
ROOT     = pathlib.Path(__file__).resolve().parent.parent
_FIG_DIR = ROOT / "outputs" / "figures"

sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.15)
matplotlib.rcParams["figure.dpi"] = 120


def _resolve_fig_dir(fig_dir) -> pathlib.Path:
    d = pathlib.Path(fig_dir) if fig_dir else _FIG_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig, name: str, fig_dir: pathlib.Path) -> None:
    path = fig_dir / name
    fig.savefig(path, bbox_inches="tight")
    print(f"[visualizer] Saved -> {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — DISTRIBUTIONS  (Phase 5)
# ══════════════════════════════════════════════════════════════════════════════

def plot_rating_distribution(
    df: pd.DataFrame,
    save: bool = True,
    fig_dir=None,
) -> plt.Figure:
    """Histogram of restaurant ratings with median line."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        df["rate"].dropna(), bins=30,
        color=sns.color_palette(PALETTE)[0],
        edgecolor="white", linewidth=0.6,
    )
    median_val = df["rate"].median()
    ax.axvline(
        median_val, color="crimson", linestyle="--",
        linewidth=1.8, label=f"Median = {median_val:.2f}",
    )
    ax.set_title("Distribution of Restaurant Ratings", fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("Rating (out of 5)", fontsize=13)
    ax.set_ylabel("Number of Restaurants", fontsize=13)
    ax.legend(fontsize=11)
    sns.despine()
    plt.tight_layout()
    if save:
        _save(fig, "01_rating_distribution.png", _resolve_fig_dir(fig_dir))
    return fig


def plot_top_locations(
    df: pd.DataFrame,
    n: int = 15,
    save: bool = True,
    fig_dir=None,
) -> plt.Figure:
    """Horizontal bar chart — top N locations by restaurant count."""
    top = df["location"].value_counts().head(n).reset_index()
    top.columns = ["location", "count"]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=top, x="count", y="location", palette=PALETTE, ax=ax)
    for bar, val in zip(ax.patches, top["count"]):
        ax.text(
            bar.get_width() + 2,
            bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontsize=10,
        )
    ax.set_title(f"Top {n} Locations by Restaurant Count", fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("Number of Restaurants", fontsize=13)
    ax.set_ylabel("Location", fontsize=13)
    sns.despine()
    plt.tight_layout()
    if save:
        _save(fig, "02_top15_locations.png", _resolve_fig_dir(fig_dir))
    return fig


def plot_restaurant_type_pie(
    df: pd.DataFrame,
    save: bool = True,
    fig_dir=None,
) -> plt.Figure:
    """Pie chart of restaurant type breakdown (slices < 2 % grouped into Other)."""
    type_counts = df["rest_type"].value_counts()
    threshold   = 0.02 * len(df)
    main_types  = type_counts[type_counts >= threshold]
    other_sum   = type_counts[type_counts < threshold].sum()
    if other_sum > 0:
        main_types = pd.concat([main_types, pd.Series({"Other": other_sum})])

    colors  = sns.color_palette(PALETTE, len(main_types))
    explode = [0.05] * len(main_types)

    fig, ax = plt.subplots(figsize=(9, 9))
    _, texts, autotexts = ax.pie(
        main_types, labels=main_types.index,
        autopct="%1.1f%%", colors=colors, explode=explode,
        startangle=140, pctdistance=0.82,
        textprops={"fontsize": 11},
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_color("white")
        at.set_fontweight("bold")
    ax.set_title("Restaurant Type Breakdown", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    if save:
        _save(fig, "03_restaurant_type_pie.png", _resolve_fig_dir(fig_dir))
    return fig


def plot_votes_distribution(
    df: pd.DataFrame,
    save: bool = True,
    fig_dir=None,
) -> plt.Figure:
    """Side-by-side histogram: full range and zoomed (< 5 000 votes)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(
        df["votes"], bins=50,
        color=sns.color_palette(PALETTE)[2],
        edgecolor="white", linewidth=0.5,
    )
    axes[0].set_title("Votes Distribution (full range)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Votes")
    axes[0].set_ylabel("Frequency")

    zoomed = df[df["votes"] < 5000]["votes"]
    axes[1].hist(
        zoomed, bins=50,
        color=sns.color_palette(PALETTE)[3],
        edgecolor="white", linewidth=0.5,
    )
    axes[1].set_title("Votes Distribution (< 5 000 votes)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Votes")
    axes[1].set_ylabel("Frequency")

    for ax in axes:
        sns.despine(ax=ax)
    plt.suptitle("Customer Engagement - Vote Counts", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save:
        _save(fig, "04_votes_distribution.png", _resolve_fig_dir(fig_dir))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — RELATIONSHIPS  (Phase 6)
# ══════════════════════════════════════════════════════════════════════════════

def plot_cost_vs_rating(
    df: pd.DataFrame,
    save: bool = True,
    fig_dir=None,
) -> plt.Figure:
    """Scatter of cost vs rating with mean-rating trend line per cost bin."""
    sc = df.dropna(subset=["rate", "approx_cost(for two people)"]).copy()
    sc["cost_f"] = sc["approx_cost(for two people)"].astype(float)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(sc["cost_f"], sc["rate"], alpha=0.25, s=18,
               color=sns.color_palette(PALETTE)[0])
    sc["cost_bin"] = pd.cut(sc["cost_f"], bins=10)
    mb   = sc.groupby("cost_bin", observed=True)["rate"].mean()
    mids = [iv.mid for iv in mb.index]
    ax.plot(mids, mb.values, color="crimson", linewidth=2.2,
            marker="o", markersize=6, label="Mean rating per cost bin")
    ax.set_title("Cost vs Rating", fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("Approx Cost for Two (Rs)", fontsize=13)
    ax.set_ylabel("Rating", fontsize=13)
    ax.legend(fontsize=11)
    sns.despine()
    plt.tight_layout()
    if save:
        _save(fig, "05_cost_vs_rating.png", _resolve_fig_dir(fig_dir))
    return fig


def plot_online_order_vs_rating(
    df: pd.DataFrame,
    save: bool = True,
    fig_dir=None,
) -> plt.Figure:
    """Boxplot + jitter strip — online order availability vs rating."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df, x="online_order", y="rate",
                palette=PALETTE, width=0.45,
                order=["Yes", "No"], ax=ax)
    sample_size = min(600, len(df))
    sns.stripplot(data=df.sample(sample_size, random_state=42),
                  x="online_order", y="rate",
                  color="black", alpha=0.18, size=3,
                  order=["Yes", "No"], ax=ax)
    medians = df.groupby("online_order")["rate"].median()
    for i, cat in enumerate(["Yes", "No"]):
        ax.text(i, medians[cat] + 0.08,
                f"Median: {medians[cat]:.2f}",
                ha="center", fontsize=10,
                color="darkred", fontweight="bold")
    ax.set_title("Online Order Availability vs Rating", fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("Accepts Online Orders", fontsize=13)
    ax.set_ylabel("Rating", fontsize=13)
    sns.despine()
    plt.tight_layout()
    if save:
        _save(fig, "06_online_order_vs_rating.png", _resolve_fig_dir(fig_dir))
    return fig


def plot_cuisine_popularity(
    df: pd.DataFrame,
    n: int = 15,
    save: bool = True,
    fig_dir=None,
) -> plt.Figure:
    """Horizontal bar chart of top N cuisines by occurrence."""
    cuisine_series = (
        df["cuisines"].dropna()
        .str.split(",").explode()
        .str.strip().str.title()
    )
    top = cuisine_series.value_counts().head(n).reset_index()
    top.columns = ["cuisine", "count"]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=top, x="count", y="cuisine", palette=PALETTE, ax=ax)
    for bar, val in zip(ax.patches, top["count"]):
        ax.text(bar.get_width() + 3,
                bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=10)
    ax.set_title(f"Top {n} Cuisines by Popularity", fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("Number of Restaurants Serving This Cuisine", fontsize=13)
    ax.set_ylabel("Cuisine", fontsize=13)
    sns.despine()
    plt.tight_layout()
    if save:
        _save(fig, "07_cuisine_popularity.png", _resolve_fig_dir(fig_dir))
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    save: bool = True,
    fig_dir=None,
) -> plt.Figure:
    """Lower-triangle correlation heatmap of numeric columns."""
    num_cols = [
        "rate", "votes", "approx_cost(for two people)",
        "cuisine_count", "is_chain", "price_per_person",
    ]
    num_cols = [c for c in num_cols if c in df.columns]
    corr = df[num_cols].astype(float).corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0,
        linewidths=0.5, linecolor="white",
        annot_kws={"size": 11}, ax=ax,
    )
    ax.set_title("Correlation Heatmap - Numeric Features",
                 fontsize=16, fontweight="bold", pad=14)
    plt.tight_layout()
    if save:
        _save(fig, "08_correlation_heatmap.png", _resolve_fig_dir(fig_dir))
    return fig


def plot_votes_vs_rating_regression(
    df: pd.DataFrame,
    vote_cap: int = 8000,
    save: bool = True,
    fig_dir=None,
) -> plt.Figure:
    """Scatter of votes vs rating with numpy regression line and Pearson r."""
    reg_df = df[df["votes"] < vote_cap].dropna(subset=["rate", "votes"])

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(reg_df["votes"], reg_df["rate"],
               alpha=0.2, s=15,
               color=sns.color_palette(PALETTE)[1])
    m, b   = np.polyfit(reg_df["votes"], reg_df["rate"], 1)
    x_line = np.linspace(reg_df["votes"].min(), reg_df["votes"].max(), 300)
    ax.plot(x_line, m * x_line + b, color="crimson", linewidth=2.2,
            label=f"Regression: y = {m:.5f}x + {b:.2f}")
    r = reg_df["votes"].corr(reg_df["rate"])
    ax.set_title(
        f"Votes vs Rating  |  Pearson r = {r:.4f}",
        fontsize=16, fontweight="bold", pad=14,
    )
    ax.set_xlabel("Number of Votes", fontsize=13)
    ax.set_ylabel("Rating", fontsize=13)
    ax.legend(fontsize=11)
    sns.despine()
    plt.tight_layout()
    if save:
        _save(fig, "09_votes_vs_rating_regression.png", _resolve_fig_dir(fig_dir))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — BUSINESS INSIGHTS  (Phase 8)
# ══════════════════════════════════════════════════════════════════════════════

def plot_insight_online_order(
    df: pd.DataFrame,
    save: bool = True,
    fig_dir=None,
) -> plt.Figure:
    """Insight 1 — boxplot + rating-bucket grouped bar for online order."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.boxplot(data=df, x="online_order", y="rate",
                order=["Yes", "No"], palette=PALETTE, width=0.45, ax=axes[0])
    medians = df.groupby("online_order")["rate"].median()
    for i, cat in enumerate(["Yes", "No"]):
        axes[0].text(i, medians[cat] + 0.1,
                     f"Median\n{medians[cat]:.2f}",
                     ha="center", fontsize=10,
                     color="darkred", fontweight="bold")
    axes[0].set_title("Rating by Online Order Availability", fontweight="bold")
    axes[0].set_xlabel("Accepts Online Orders")
    axes[0].set_ylabel("Rating")
    sns.despine(ax=axes[0])

    bucket_pct = (
        df.groupby(["online_order", "rating_bucket"], observed=True)
        .size().reset_index(name="count")
    )
    bucket_pct["pct"] = (
        bucket_pct.groupby("online_order")["count"]
        .transform(lambda x: x / x.sum() * 100)
    )
    pivot = bucket_pct.pivot(index="rating_bucket", columns="online_order", values="pct")
    pivot.plot(kind="bar", ax=axes[1],
               color=sns.color_palette(PALETTE, 2),
               edgecolor="white", rot=0, width=0.6)
    axes[1].set_title("Rating Bucket Distribution by Online Order", fontweight="bold")
    axes[1].set_xlabel("Rating Bucket")
    axes[1].set_ylabel("% of Restaurants")
    axes[1].legend(title="Online Order", fontsize=10)
    sns.despine(ax=axes[1])

    plt.suptitle("Insight 1: Online Ordering Lifts Ratings",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save:
        _save(fig, "insight_01_online_order_ratings.png", _resolve_fig_dir(fig_dir))
    return fig


def plot_insight_golden_zones(
    df: pd.DataFrame,
    min_count: int = 50,
    save: bool = True,
    fig_dir=None,
) -> plt.Figure:
    """Insight 2 — bubble scatter: neighbourhood density vs avg rating."""
    loc_stats = (
        df.groupby("location")
        .agg(count=("name", "count"),
             avg_rating=("rate", "mean"),
             avg_votes=("votes", "mean"))
        .reset_index()
    )
    loc_stats = loc_stats[loc_stats["count"] >= min_count].copy()

    fig, ax = plt.subplots(figsize=(12, 7))
    scatter = ax.scatter(
        loc_stats["count"], loc_stats["avg_rating"],
        s=loc_stats["avg_votes"] / 8,
        c=loc_stats["avg_rating"],
        cmap="RdYlGn", alpha=0.8,
        edgecolors="grey", linewidth=0.5,
        vmin=2.5, vmax=4.5,
    )
    plt.colorbar(scatter, ax=ax, label="Avg Rating")
    for _, row in loc_stats.nlargest(8, "count").iterrows():
        ax.annotate(row["location"],
                    xy=(row["count"], row["avg_rating"]),
                    xytext=(6, 3), textcoords="offset points",
                    fontsize=9, color="#333333")
    ax.axhline(loc_stats["avg_rating"].median(), color="crimson",
               linestyle="--", linewidth=1.2, alpha=0.6, label="Median rating")
    ax.axvline(loc_stats["count"].median(), color="steelblue",
               linestyle="--", linewidth=1.2, alpha=0.6, label="Median count")
    ax.set_title(
        "Neighbourhood Landscape: Density vs Average Rating\n(bubble size = avg votes)",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel("Number of Restaurants", fontsize=12)
    ax.set_ylabel("Average Rating", fontsize=12)
    ax.legend(fontsize=10)
    sns.despine()
    plt.tight_layout()
    if save:
        _save(fig, "insight_02_location_golden_zones.png", _resolve_fig_dir(fig_dir))
    return fig


def plot_insight_price_tiers(
    df: pd.DataFrame,
    save: bool = True,
    fig_dir=None,
) -> plt.Figure:
    """Insight 3 — restaurant count and avg rating by price tier."""
    df2 = df.dropna(subset=["price_per_person"]).copy()
    df2["price_tier"] = pd.cut(
        df2["price_per_person"],
        bins=[0, 200, 400, 700, 1800],
        labels=["Budget\n(<=200)", "Mid\n(201-400)",
                "Upper-Mid\n(401-700)", "Premium\n(700+)"],
    )
    tier_stats = (
        df2.groupby("price_tier", observed=True)
        .agg(count=("rate", "count"), avg_rating=("rate", "mean"))
        .reset_index()
    )
    colors = sns.color_palette("RdYlGn", len(tier_stats))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bars = axes[0].bar(tier_stats["price_tier"].astype(str),
                       tier_stats["count"],
                       color=colors, edgecolor="white", width=0.55)
    for bar, val in zip(bars, tier_stats["count"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 15,
                     str(val), ha="center", fontweight="bold", fontsize=10)
    axes[0].set_title("Restaurant Count by Price Tier", fontweight="bold")
    axes[0].set_xlabel("Price Tier (per person)")
    axes[0].set_ylabel("Number of Restaurants")
    sns.despine(ax=axes[0])

    bars2 = axes[1].bar(tier_stats["price_tier"].astype(str),
                        tier_stats["avg_rating"],
                        color=colors, edgecolor="white", width=0.55)
    for bar, val in zip(bars2, tier_stats["avg_rating"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.02,
                     f"{val:.2f}", ha="center", fontweight="bold", fontsize=10)
    axes[1].set_title("Average Rating by Price Tier", fontweight="bold")
    axes[1].set_xlabel("Price Tier (per person)")
    axes[1].set_ylabel("Average Rating")
    axes[1].set_ylim(0, 5)
    sns.despine(ax=axes[1])

    plt.suptitle("Insight 3: Budget Overcrowded - Premium Earns Better Ratings",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save:
        _save(fig, "insight_03_price_tier_ratings.png", _resolve_fig_dir(fig_dir))
    return fig


def plot_insight_rest_type(
    df: pd.DataFrame,
    top_n: int = 6,
    save: bool = True,
    fig_dir=None,
) -> plt.Figure:
    """Insight 4 — violin plot of rating + avg votes bar for top N rest types."""
    top_types = df["rest_type"].value_counts().head(top_n).index.tolist()
    df_top    = df[df["rest_type"].isin(top_types)].copy()
    type_stats = (
        df_top.groupby("rest_type")
        .agg(count=("name", "count"),
             avg_votes=("votes", "mean"))
        .sort_values("count", ascending=False)
        .reset_index()
    )
    order = type_stats["rest_type"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.violinplot(data=df_top, x="rest_type", y="rate",
                   order=order, palette=PALETTE,
                   inner="quartile", ax=axes[0])
    axes[0].set_title("Rating Distribution by Restaurant Type", fontweight="bold")
    axes[0].set_xlabel("Restaurant Type")
    axes[0].set_ylabel("Rating")
    axes[0].tick_params(axis="x", rotation=30)
    sns.despine(ax=axes[0])

    bars = axes[1].bar(type_stats["rest_type"], type_stats["avg_votes"],
                       color=sns.color_palette(PALETTE, len(type_stats)),
                       edgecolor="white")
    for bar, val in zip(bars, type_stats["avg_votes"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 10,
                     f"{val:.0f}", ha="center", fontsize=9, fontweight="bold")
    axes[1].set_title("Average Votes by Restaurant Type", fontweight="bold")
    axes[1].set_xlabel("Restaurant Type")
    axes[1].set_ylabel("Average Votes")
    axes[1].tick_params(axis="x", rotation=30)
    sns.despine(ax=axes[1])

    plt.suptitle("Insight 4: Casual Dining Outperforms Quick Bites",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save:
        _save(fig, "insight_04_rest_type_ratings.png", _resolve_fig_dir(fig_dir))
    return fig


def plot_insight_cuisine_diversity(
    df: pd.DataFrame,
    save: bool = True,
    fig_dir=None,
) -> plt.Figure:
    """Insight 5 — avg rating and avg votes by cuisine_count."""
    cc_stats = (
        df.groupby("cuisine_count")
        .agg(count=("rate", "count"),
             avg_rating=("rate", "mean"),
             avg_votes=("votes", "mean"))
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bars = axes[0].bar(cc_stats["cuisine_count"].astype(str),
                       cc_stats["avg_rating"],
                       color=sns.color_palette("Blues_d", len(cc_stats)),
                       edgecolor="white", width=0.5)
    for bar, val in zip(bars, cc_stats["avg_rating"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.02,
                     f"{val:.2f}", ha="center", fontweight="bold", fontsize=11)
    axes[0].set_title("Average Rating by Number of Cuisines", fontweight="bold")
    axes[0].set_xlabel("Number of Cuisines")
    axes[0].set_ylabel("Average Rating")
    axes[0].set_ylim(0, 5)
    sns.despine(ax=axes[0])

    bars2 = axes[1].bar(cc_stats["cuisine_count"].astype(str),
                        cc_stats["avg_votes"],
                        color=sns.color_palette("Greens_d", len(cc_stats)),
                        edgecolor="white", width=0.5)
    for bar, val in zip(bars2, cc_stats["avg_votes"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 20,
                     f"{val:.0f}", ha="center", fontweight="bold", fontsize=11)
    axes[1].set_title("Average Votes by Number of Cuisines", fontweight="bold")
    axes[1].set_xlabel("Number of Cuisines")
    axes[1].set_ylabel("Average Votes")
    sns.despine(ax=axes[1])

    plt.suptitle("Insight 5: Cuisine Diversity Correlates With Higher Ratings",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save:
        _save(fig, "insight_05_cuisine_diversity.png", _resolve_fig_dir(fig_dir))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE — run all plots at once
# ══════════════════════════════════════════════════════════════════════════════

def plot_all(df: pd.DataFrame, fig_dir=None) -> None:
    """
    Generate and save every plot in one call.
    Useful for a full refresh after the data pipeline updates.
    """
    fns = [
        plot_rating_distribution,
        plot_top_locations,
        plot_restaurant_type_pie,
        plot_votes_distribution,
        plot_cost_vs_rating,
        plot_online_order_vs_rating,
        plot_cuisine_popularity,
        plot_correlation_heatmap,
        plot_votes_vs_rating_regression,
        plot_insight_online_order,
        plot_insight_golden_zones,
        plot_insight_price_tiers,
        plot_insight_rest_type,
        plot_insight_cuisine_diversity,
    ]
    for fn in fns:
        try:
            fn(df, save=True, fig_dir=fig_dir)
            plt.close("all")
        except Exception as exc:
            print(f"[visualizer] WARNING: {fn.__name__} failed — {exc}")
    print(f"[visualizer] Done — {len(fns)} plots attempted.")


# ── CLI smoke-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _df = pd.read_csv(ROOT / "data" / "processed" / "zomato_clean.csv")
    _df["rating_bucket"] = pd.Categorical(
        _df["rating_bucket"],
        categories=["Poor", "Average", "Good", "Excellent"],
        ordered=True,
    )
    plot_all(_df)

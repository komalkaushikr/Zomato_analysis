"""
cleaner.py
----------
Phase 3 : Data cleaning
Phase 4 : Feature engineering   (functions added below the cleaning section)

Public API
----------
Phase 3
    clean_data(df)         -> cleaned DataFrame
    save_clean_data(df)    -> saves to data/processed/zomato_clean.csv

Phase 4
    engineer_features(df)  -> adds derived columns
"""

import pathlib
import numpy as np
import pandas as pd

ROOT       = pathlib.Path(__file__).resolve().parent.parent
RAW_PATH   = ROOT / "data" / "raw"   / "zomato_raw.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "zomato_clean.csv"

# ============================================================
#  PHASE 3 — CLEANING
# ============================================================

# ── 1. Duplicates ────────────────────────────────────────────
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop fully duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    print(f"[cleaner] Duplicates removed : {before - len(df)}")
    return df.reset_index(drop=True)


# ── 2. Rate column ───────────────────────────────────────────
def clean_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert rate strings like '4.1/5' -> 4.1 (float).
    'NEW' and '-' become NaN.
    """
    def _parse(val):
        val = str(val).strip()
        if val in ("NEW", "-", "nan", ""):
            return np.nan
        return float(val.split("/")[0])

    df["rate"] = df["rate"].apply(_parse)
    print(f"[cleaner] rate NaNs after clean : {df['rate'].isna().sum()}")
    return df


# ── 3. Cost column ───────────────────────────────────────────
def clean_cost(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove commas, strip whitespace, cast to Int64 (nullable int).
    Unparsable values become NaN.
    """
    col = "approx_cost(for two people)"
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    print(f"[cleaner] cost NaNs after clean : {df[col].isna().sum()}")
    return df


# ── 4. Location standardization ──────────────────────────────
def standardize_location(df: pd.DataFrame) -> pd.DataFrame:
    """Strip extra whitespace, apply Title Case to location and listed_in(city)."""
    for col in ["location", "listed_in(city)"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
    return df


# ── 5. Null handling ─────────────────────────────────────────
def handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    - rate      : fill NaN with median (restaurants without ratings kept)
    - dish_liked: fill NaN with 'Not Available'
    - cuisines  : drop rows where cuisines is null (rare edge case)
    """
    median_rate = df["rate"].median()
    df["rate"] = df["rate"].fillna(median_rate)
    print(f"[cleaner] rate NaNs filled with median ({median_rate:.2f})")

    df["dish_liked"] = df["dish_liked"].fillna("Not Available")

    before = len(df)
    df = df.dropna(subset=["cuisines"])
    print(f"[cleaner] Rows dropped (null cuisines) : {before - len(df)}")

    return df.reset_index(drop=True)


# ── 6. Drop irrelevant columns ───────────────────────────────
_COLS_TO_DROP = ["url", "address", "phone", "reviews_list", "menu_item"]

def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that add no analytical value."""
    cols_present = [c for c in _COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_present)
    print(f"[cleaner] Dropped columns : {cols_present}")
    return df


# ── 7. Master cleaning pipeline ──────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run all cleaning steps in order and return the cleaned DataFrame."""
    print("\n[cleaner] ===== CLEANING PIPELINE =====")
    df = remove_duplicates(df)
    df = clean_rate(df)
    df = clean_cost(df)
    df = standardize_location(df)
    df = handle_nulls(df)
    df = drop_irrelevant_columns(df)
    print(f"[cleaner] Final shape : {df.shape}")
    print("[cleaner] ================================\n")
    return df


def save_clean_data(df: pd.DataFrame) -> None:
    """Save cleaned DataFrame to data/processed/zomato_clean.csv."""
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"[cleaner] Saved clean data -> {CLEAN_PATH}")


# ============================================================
#  PHASE 4 — FEATURE ENGINEERING
# ============================================================

# ── F1. cuisine_count ────────────────────────────────────────
def add_cuisine_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count how many cuisines each restaurant serves.
    e.g. 'North Indian, Chinese, Biryani' -> 3
    """
    df["cuisine_count"] = (
        df["cuisines"]
        .astype(str)
        .apply(lambda x: len([c.strip() for c in x.split(",") if c.strip()]))
    )
    print(f"[engineer] cuisine_count  | mean={df['cuisine_count'].mean():.2f} "
          f"| max={df['cuisine_count'].max()}")
    return df


# ── F2. is_chain ─────────────────────────────────────────────
def add_is_chain(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag restaurants whose name appears in more than one location.
    A restaurant is a 'chain' if (name, location) combos > 1 distinct location.
    """
    location_counts = (
        df.groupby("name")["location"]
        .nunique()
        .reset_index(name="loc_count")
    )
    chain_names = set(location_counts.loc[location_counts["loc_count"] > 1, "name"])
    df["is_chain"] = df["name"].isin(chain_names).astype(int)
    print(f"[engineer] is_chain       | chains={df['is_chain'].sum():,} "
          f"({df['is_chain'].mean()*100:.1f}%)")
    return df


# ── F3. price_per_person ─────────────────────────────────────
def add_price_per_person(df: pd.DataFrame) -> pd.DataFrame:
    """
    approx_cost(for two people) / 2, rounded to nearest integer.
    Stored as float to handle any NaN propagation gracefully.
    """
    col = "approx_cost(for two people)"
    df["price_per_person"] = (df[col].astype(float) / 2).round(0)
    print(f"[engineer] price_per_person | mean=Rs.{df['price_per_person'].mean():.0f} "
          f"| max=Rs.{df['price_per_person'].max():.0f}")
    return df


# ── F4. has_online_order ─────────────────────────────────────
def add_has_online_order(df: pd.DataFrame) -> pd.DataFrame:
    """Convert online_order Yes/No string to boolean (True/False)."""
    df["has_online_order"] = df["online_order"].str.strip().str.title() == "Yes"
    print(f"[engineer] has_online_order | True={df['has_online_order'].sum():,} "
          f"({df['has_online_order'].mean()*100:.1f}%)")
    return df


# ── F5. rating_bucket ────────────────────────────────────────
_RATING_BINS   = [0.0, 2.5, 3.5, 4.0, 5.0]
_RATING_LABELS = ["Poor", "Average", "Good", "Excellent"]

def add_rating_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin the cleaned numeric rate into four ordered categories:
      Poor      : 0.0 – 2.5
      Average   : 2.5 – 3.5
      Good      : 3.5 – 4.0
      Excellent : 4.0 – 5.0
    """
    df["rating_bucket"] = pd.cut(
        df["rate"],
        bins=_RATING_BINS,
        labels=_RATING_LABELS,
        include_lowest=True,
    )
    print(f"[engineer] rating_bucket  |\n{df['rating_bucket'].value_counts().sort_index()}\n")
    return df


# ── F6. Master feature-engineering pipeline ──────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all engineered features to a cleaned DataFrame.
    Call AFTER clean_data().
    """
    print("\n[engineer] ===== FEATURE ENGINEERING =====")
    df = add_cuisine_count(df)
    df = add_is_chain(df)
    df = add_price_per_person(df)
    df = add_has_online_order(df)
    df = add_rating_bucket(df)
    print(f"[engineer] Final shape : {df.shape}")
    print("[engineer] ==========================================\n")
    return df


# ── CLI runner ───────────────────────────────────────────────
if __name__ == "__main__":
    from src.data_loader import load_raw_data
    raw   = load_raw_data()
    clean = clean_data(raw)
    final = engineer_features(clean)
    save_clean_data(final)
    print(final[["name", "rate", "rating_bucket", "cuisine_count",
                 "is_chain", "price_per_person", "has_online_order"]].head(8).to_string())
    print("\nNew column dtypes:")
    new_cols = ["cuisine_count","is_chain","price_per_person",
                "has_online_order","rating_bucket"]
    print(final[new_cols].dtypes)

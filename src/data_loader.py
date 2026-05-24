"""
data_loader.py
--------------
Loads the Zomato Bengaluru dataset.

Priority:
  1. data/raw/zomato_raw.csv already present  -> load it directly
  2. Try downloading from a public mirror      -> save then load
  3. Generate realistic synthetic data (5000+ rows) -> save then load

Usage:
    from src.data_loader import load_raw_data
    df = load_raw_data()
"""

import os
import pathlib
import requests
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "zomato_raw.csv"

# Public mirror of the Zomato Bengaluru Kaggle dataset (CSV direct-download).
# If this URL goes stale the synthetic generator acts as fallback.
_DATASET_URL = (
    "https://raw.githubusercontent.com/dsrscientist/"
    "dataset1/master/zomato.csv"
)


# ── public entry-point ─────────────────────────────────────────────────────────

def load_raw_data() -> pd.DataFrame:
    """Return the raw Zomato DataFrame, fetching / generating it if needed."""
    if RAW_PATH.exists():
        print(f"[loader] Reading existing file: {RAW_PATH}")
        return pd.read_csv(RAW_PATH, encoding="latin-1")

    print("[loader] Raw file not found. Trying download …")
    df = _try_download()

    if df is None:
        print("[loader] Download failed. Generating synthetic data …")
        df = _generate_synthetic(n=5500)

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_PATH, index=False)
    print(f"[loader] Saved {len(df):,} rows -> {RAW_PATH}")
    return df


# ── download helper ────────────────────────────────────────────────────────────

def _try_download() -> pd.DataFrame | None:
    try:
        resp = requests.get(_DATASET_URL, timeout=30)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text), encoding="latin-1")
        if len(df) > 100:          # sanity check
            print(f"[loader] Downloaded {len(df):,} rows.")
            return df
    except Exception as exc:
        print(f"[loader] Download error: {exc}")
    return None


# ── synthetic data generator ───────────────────────────────────────────────────

def _generate_synthetic(n: int = 5500, seed: int = 42) -> pd.DataFrame:
    """
    Generate a DataFrame that mirrors the real Zomato Bengaluru CSV schema:
    url, address, name, online_order, book_table, rate, votes, phone,
    location, rest_type, dish_liked, cuisines,
    approx_cost(for two people), reviews_list,
    menu_item, listed_in(type), listed_in(city)
    """
    rng = np.random.default_rng(seed)

    # ── reference pools ───────────────────────────────────────────────────
    locations = [
        "Koramangala", "Indiranagar", "Whitefield", "BTM Layout",
        "Jayanagar", "JP Nagar", "HSR Layout", "Marathahalli",
        "Bellandur", "Electronic City", "MG Road", "Brigade Road",
        "Bannerghatta Road", "Yelahanka", "Hebbal", "Malleshwaram",
        "Rajajinagar", "Banashankari", "Sarjapur Road", "Brookefield",
    ]
    rest_types = [
        "Quick Bites", "Casual Dining", "Cafe", "Delivery",
        "Dessert Parlor", "Bakery", "Fine Dining", "Food Court",
        "Bar", "Beverage Shop", "Sweet Shop",
    ]
    cuisine_pool = [
        "North Indian", "South Indian", "Chinese", "Fast Food",
        "Biryani", "Pizza", "Burger", "Continental", "Italian",
        "Cafe", "Bakery", "Desserts", "Beverages", "Seafood",
        "Mughlai", "Rolls", "Momos", "Sandwich", "Mexican",
        "Thai", "Japanese", "Mediterranean", "Lebanese",
    ]
    listed_types = [
        "Delivery", "Dine-out", "Dine-out, Delivery",
        "Cafes", "Desserts", "Pubs and bars", "Buffet",
    ]
    name_prefixes = [
        "Spice", "Royal", "The", "Café", "Hotel", "New", "Sri",
        "Green", "Golden", "Blue", "Fresh", "Urban", "Classic",
    ]
    name_suffixes = [
        "Kitchen", "Bites", "Corner", "Palace", "Hub", "Garden",
        "Express", "House", "Point", "Lounge", "Bistro", "Dhaba",
    ]
    dish_pool = [
        "Butter Chicken", "Biryani", "Pizza", "Burger", "Dosa",
        "Paneer Tikka", "Pasta", "Momos", "Noodles", "Ice Cream",
        "Rolls", "Sandwich", "Idli", "Vada", "Thali", "Kebab",
        "Fried Rice", "Chole Bhature", "Pav Bhaji", "Halwa",
    ]

    # ── simulate chains (some names repeat across rows) ───────────────────
    chain_names = [
        "McDonald's", "KFC", "Subway", "Domino's Pizza",
        "Burger King", "Pizza Hut", "Barbeque Nation",
        "Café Coffee Day", "Starbucks", "Haldiram's",
    ]
    is_chain_mask = rng.random(n) < 0.18          # ~18 % are chains

    # ── build columns ─────────────────────────────────────────────────────
    names = []
    for i in range(n):
        if is_chain_mask[i]:
            names.append(rng.choice(chain_names))
        else:
            names.append(
                f"{rng.choice(name_prefixes)} {rng.choice(name_suffixes)}"
            )

    locs   = rng.choice(locations, n)
    r_type = rng.choice(rest_types, n)

    # rate: realistic mix of float ratings, "NEW", and "-"
    raw_ratings = []
    for _ in range(n):
        rv = rng.random()
        if rv < 0.05:
            raw_ratings.append("NEW")
        elif rv < 0.08:
            raw_ratings.append("-")
        else:
            raw_ratings.append(f"{rng.uniform(2.0, 5.0):.1f}/5")

    votes = rng.integers(0, 12000, n)

    # online_order / book_table
    online = rng.choice(["Yes", "No"], n, p=[0.65, 0.35])
    book   = rng.choice(["Yes", "No"], n, p=[0.30, 0.70])

    # cost — realistic distribution (₹150 – ₹3 500 for two)
    costs = rng.choice(
        [150, 200, 250, 300, 350, 400, 450, 500, 600, 700,
         800, 900, 1000, 1200, 1500, 2000, 2500, 3000, 3500],
        n,
    )
    # add commas to some values to mimic raw data quirks
    cost_strs = [
        f"{c:,}" if rng.random() < 0.5 else str(c) for c in costs
    ]

    # cuisines (1–3 per restaurant)
    cuisines = [
        ", ".join(rng.choice(cuisine_pool, rng.integers(1, 4), replace=False).tolist())
        for _ in range(n)
    ]

    # dish_liked (0–3, some NaN)
    dish_liked = []
    for _ in range(n):
        if rng.random() < 0.20:
            dish_liked.append(np.nan)
        else:
            dish_liked.append(
                ", ".join(rng.choice(dish_pool, rng.integers(1, 4), replace=False).tolist())
            )

    # listed_in(city) mirrors location
    listed_city = locs.copy()

    # address
    streets = ["1st Cross", "2nd Main", "5th Block", "Ring Road",
                "100 Feet Road", "Old Airport Road", "Station Road"]
    addresses = [
        f"No. {rng.integers(1,200)}, {rng.choice(streets)}, {loc}, Bengaluru"
        for loc in locs
    ]

    phones = [f"+91-{rng.integers(70000_00000, 99999_99999)}" for _ in range(n)]

    urls = [
        f"https://www.zomato.com/bangalore/{name.lower().replace(' ','-')}-{loc.lower().replace(' ','-')}"
        for name, loc in zip(names, locs)
    ]

    df = pd.DataFrame({
        "url":                        urls,
        "address":                    addresses,
        "name":                       names,
        "online_order":               online,
        "book_table":                 book,
        "rate":                       raw_ratings,
        "votes":                      votes,
        "phone":                      phones,
        "location":                   locs,
        "rest_type":                  r_type,
        "dish_liked":                 dish_liked,
        "cuisines":                   cuisines,
        "approx_cost(for two people)":cost_strs,
        "reviews_list":               [f"[('Rated {rng.uniform(1,5):.1f}', 'Sample review')]" for _ in range(n)],
        "menu_item":                  [str([]) for _ in range(n)],
        "listed_in(type)":            rng.choice(listed_types, n),
        "listed_in(city)":            listed_city,
    })

    # inject ~2 % duplicates to mimic real data
    dup_idx = rng.choice(n, size=int(n * 0.02), replace=False)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)

    return df


# ── convenience ────────────────────────────────────────────────────────────────

def get_data_info(df: pd.DataFrame) -> None:
    """Print a quick summary of the loaded DataFrame."""
    print(f"\nShape : {df.shape}")
    print(f"Cols  : {list(df.columns)}")
    print(f"\nNull counts:\n{df.isnull().sum()}")
    print(f"\nDtypes:\n{df.dtypes}")


if __name__ == "__main__":
    df = load_raw_data()
    get_data_info(df)


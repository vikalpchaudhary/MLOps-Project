import pandas as pd
from pathlib import Path

TRAIN_PATH = Path("data/raw/train.csv")
HOLDOUT_PATH = Path("data/raw/holdout_features.csv")
LATER_PATH = Path("data/raw/later.csv")

def check_leaking_columns(train_df: pd.DataFrame, holdout_df: pd.DataFrame):
    """Check for columns present in train but missing in holdout (excluding the target delay_min)."""
    train_cols = set(train_df.columns)
    holdout_cols = set(holdout_df.columns)
    
    # Missing columns except the target
    potential_leaks = train_cols - holdout_cols - {"delay_min"}
    
    print("\n--- 1. LEAKING COLUMN CHECK ---")
    if potential_leaks:
        for col in potential_leaks:
            # Check correlation with the target delay_min
            corr = train_df[col].corr(train_df["delay_min"])
            print(f"Found leaking column: '{col}' with correlation to target: {corr:.4f}")
    else:
        print("No leaking column found.")

def check_unit_mixup(train_df: pd.DataFrame):
    print("\n--- 2. UNIT MIXUP CHECK ---")
    numeric_cols = ["route_km", "stops", "passengers", "hour", "temp_c", "driver_exp_yr"]
    for col in numeric_cols:
        grouped = train_df.groupby("depot_zone")[col].agg(["min", "mean", "max"])
        print(f"\nFeature '{col}' by depot_zone:")
        print(grouped)

def check_sentinel_values(train_df: pd.DataFrame):
    print("\n--- 3. SENTINEL VALUES CHECK ---")
    numeric_cols = ["route_km", "stops", "passengers", "hour", "temp_c", "driver_exp_yr", "delay_min"]
    for col in numeric_cols:
        # Check suspicious values like -999, -1, 999, 9999, etc.
        suspicious = train_df[train_df[col].isin([-999, -1, 99, 999, 9999])][col].value_counts()
        if not suspicious.empty:
            print(f"Possible sentinel in '{col}':\n{suspicious}")
        # Also check min and max
        col_min, col_max = train_df[col].min(), train_df[col].max()
        if col_min < 0 or col_max > 500:
            print(f"Extreme range in '{col}': min={col_min}, max={col_max}")

def check_duplicates(train_df: pd.DataFrame):
    """Check for exact duplicate rows."""
    extra_duplicates = train_df.duplicated().sum()
    print("\n--- 4. DUPLICATE ROWS CHECK ---")
    print(f"Number of extra duplicate rows: {extra_duplicates}")

def check_stuck_sensor(train_df: pd.DataFrame):
    print("\n--- 5. STUCK SENSOR CHECK ---")
    # Sort by date and check for consecutive identical readings
    sorted_df = train_df.sort_values("date")
    numeric_cols = ["route_km", "stops", "passengers", "temp_c"]
    
    for col in numeric_cols:
        # Check how many times the value stays exactly constant day over day
        daily_col = sorted_df.groupby("date")[col].nunique()
        # Find if on any run of dates, a column has 0 variance (variance == 0 across days)
        runs = (sorted_df[col] != sorted_df[col].shift()).cumsum()
        longest_run = sorted_df.groupby(runs).filter(lambda x: len(x) > 5)
        if not longest_run.empty:
            val = longest_run[col].iloc[0]
            start_date = longest_run["date"].min()
            end_date = longest_run["date"].max()
            count = len(longest_run)
            print(f"Sensor in '{col}' stuck at value {val} from {start_date} to {end_date} (count: {count})")

def check_target_typos(train_df: pd.DataFrame):
    print("\n--- 6.1. TARGET TYPOS CHECK ---")
    # Look at the top 15 highest delay values
    top_delays = train_df["delay_min"].sort_values(ascending=False).head(15)
    print("Top 15 largest delay_min values:")
    print(top_delays.to_string())
    
def inspect_large_delays(train_df: pd.DataFrame):
    print("\n--- 6.2. TARGET TYPOS DETAILED INSPECTION ---")
    # Rows with delay_min > 50
    large = train_df[train_df["delay_min"] > 50][["record_id", "date", "delay_logged_by_conductor", "delay_min"]]
    print(large)
    print(f"\nTotal rows with delay_min > 50: {len(large)}")

def check_spelling_variants(train_df: pd.DataFrame, later_df: pd.DataFrame):
    print("\n--- 7. SPELLING VARIANTS CHECK ---")
    text_cols = ["weather", "depot_zone"]
    for col in text_cols:
        train_vals = train_df[col].unique()
        later_vals = later_df[col].unique()
        print(f"\nColumn '{col}':")
        print(f"  train.csv unique values ({len(train_vals)}): {train_vals}")
        print(f"  later.csv clean values ({len(later_vals)}): {later_vals}")

def clean_data(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Cleans raw bus delay data based on audited findings:
    1. Removes duplicate rows.
    2. Drops the leaking column 'delay_logged_by_conductor' if present.
    3. Corrects driver_exp_yr for North depot (recorded in months instead of years).
    4. Corrects 10x typos in target delay_min (if present).
    """
    if not is_train:
        return df.drop(columns=["delay_logged_by_conductor"], errors="ignore").copy()

    df_clean = df.drop_duplicates().copy()

    # 2. Drop leaking column
    if "delay_logged_by_conductor" in df_clean.columns:
        df_clean = df_clean.drop(columns=["delay_logged_by_conductor"])

    # 3. Fix unit mixup: North depot driver_exp_yr was recorded in months (factor of 12)
    north_mask = df_clean["depot_zone"] == "North"
    df_clean.loc[north_mask, "driver_exp_yr"] = df_clean.loc[north_mask, "driver_exp_yr"] / 12.0

    # 4. Fix target typos (only if target delay_min is in the dataframe)
    if "delay_min" in df_clean.columns:
        typo_mask = df_clean["delay_min"] > 50.0
        df_clean.loc[typo_mask, "delay_min"] = df_clean.loc[typo_mask, "delay_min"] / 10.0

    return df_clean

if __name__ == "__main__":
    train_df = pd.read_csv(TRAIN_PATH)
    holdout_df = pd.read_csv(HOLDOUT_PATH)
    later_df = pd.read_csv(LATER_PATH)
    
    check_leaking_columns(train_df, holdout_df)
    check_unit_mixup(train_df)
    check_sentinel_values(train_df)
    check_duplicates(train_df)
    check_stuck_sensor(train_df)
    check_target_typos(train_df)
    inspect_large_delays(train_df)
    check_spelling_variants(train_df, later_df)

    print("\n--- 8. VERIFYING DATA CLEANING ---")
    cleaned_train = clean_data(train_df)
    print(f"Original shape: {train_df.shape}")
    print(f"Cleaned shape:  {cleaned_train.shape}")
    print(f"Remaining duplicates: {cleaned_train.duplicated().sum()}")
    print("Cleaned driver_exp_yr summary by depot_zone:")
    print(cleaned_train.groupby("depot_zone")["driver_exp_yr"].agg(["min", "mean", "max"]))
    if "delay_min" in cleaned_train.columns:
        print(f"Cleaned delay_min max: {cleaned_train['delay_min'].max():.2f}")

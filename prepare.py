
import pandas as pd


def audit_raw_data():
    raw_df = pd.read_csv("transactions_raw.csv")

    print("=== RAW DATA AUDIT ===")
    print(f"Rows as read: {raw_df.shape[0]}")
    print(f"Columns as read: {raw_df.shape[1]}")
    print(f"Exact duplicate rows: {raw_df.duplicated().sum()}")

    deduplicated_df = raw_df.drop_duplicates().copy()
    amounts = pd.to_numeric(
        deduplicated_df["amount_inr"], errors="coerce"
    )

    print(f"Rows after de-duplication: {len(deduplicated_df)}")
    print(f"Missing amount_inr: {deduplicated_df['amount_inr'].isna().sum()}")
    print(
        "Missing distance_from_home_km:",
        deduplicated_df["distance_from_home_km"].isna().sum()
    )
    print(
        "Out-of-range amounts:",
        ((amounts < 1) | (amounts > 1_000_000)).sum()
    )
    print(
        "Distinct raw merchant_category spellings:",
        raw_df["merchant_category"].nunique()
    )
    print("Distinct raw city spellings:", raw_df["city"].nunique())


def clean_transactions(raw_df):
    # Work on a copy so the original DataFrame stays unchanged.
    df = raw_df.copy()

    # (a) Remove exact duplicate rows.
    df = df.drop_duplicates()

    # (b) Normalize category and city spellings.
    for column in ["merchant_category", "city"]:
        df[column] = df[column].str.strip().str.title()

    # Ensure amounts are numeric before applying cleaning rules.
    df["amount_inr"] = pd.to_numeric(
        df["amount_inr"], errors="coerce"
    )

    # (c) Drop rows with missing amounts.
    df = df.dropna(subset=["amount_inr"])

    # (d) Drop impossible amounts.
    df = df[
        (df["amount_inr"] >= 1)
        & (df["amount_inr"] <= 1_000_000)
    ]

    # (e) Reset the index.
    df = df.reset_index(drop=True)

    return df


def add_features(df):
    # Return a new DataFrame without modifying the input.
    result = df.copy()

    result["amount_vs_avg_ratio"] = (
        result["amount_inr"] / result["avg_amount_30d_inr"]
    )

    result["is_night"] = (
        result["hour_of_day"] <= 5
    ).astype(int)

    result["high_value"] = (
        result["amount_inr"] > 50_000
    ).astype(int)

    return result


def print_part_a_results(raw_df, clean_df, featured_df):
    print("\n=== CLEANING CHECKPOINTS ===")
    print(f"Cleaned rows: {len(clean_df)}")
    print(f"Duplicate rows remaining: {clean_df.duplicated().sum()}")

    print("\nMerchant category counts:")
    print(clean_df["merchant_category"].value_counts().sort_index())

    print("\nDistinct cleaned cities:", clean_df["city"].nunique())

    print("\nPayment mode counts:")
    print(clean_df["payment_mode"].value_counts().to_string())

    print(f"\nMinimum amount_inr: {clean_df['amount_inr'].min():.2f}")
    print(f"Maximum amount_inr: {clean_df['amount_inr'].max():.2f}")
    print(
        "Remaining missing distance_from_home_km:",
        clean_df["distance_from_home_km"].isna().sum()
    )

    print("\n=== FEATURE ENGINEERING CHECKPOINTS ===")
    print(f"is_night sum: {featured_df['is_night'].sum()}")
    print(f"high_value sum: {featured_df['high_value'].sum()}")
    print(
        "Maximum amount_vs_avg_ratio:",
        f"{featured_df['amount_vs_avg_ratio'].max():.2f}"
    )

    fraud_count = int(clean_df["is_fraud"].sum())
    total_rows = len(clean_df)
    fraud_rate = fraud_count / total_rows

    print("\n=== FRAUD AND ACCURACY CHECKPOINTS ===")
    print(f"Fraud rows: {fraud_count}")
    print(f"Total cleaned rows: {total_rows}")
    print(f"Fraud rate: {fraud_rate * 100:.2f}%")
    print(f"Never-fraud accuracy: {(1 - fraud_rate) * 100:.2f}%")
    print("Never-fraud recall: 0.00")

    print("\n=== CHARGEBACK LEAKAGE AUDIT ===")
    leakage = clean_df.groupby("chargeback_filed")["is_fraud"].agg(
        rows="count", fraud_rows="sum"
    )
    print(leakage.to_string())

    print("\nExcluded model columns: txn_id, customer_id, chargeback_filed, is_fraud")
    print(
        "Model-input columns before one-hot encoding:",
        [
            "amount_inr",
            "hour_of_day",
            "txn_count_last_24h",
            "avg_amount_30d_inr",
            "distance_from_home_km",
            "is_new_device",
            "account_age_days",
            "amount_vs_avg_ratio",
            "is_night",
            "high_value",
            "merchant_category",
            "city",
            "payment_mode",
        ]
    )

    print("\n=== ONE-ROW AND INPUT-UNCHANGED CHECKS ===")
    one_row_input = raw_df.iloc[[0]].copy()
    original_copy = one_row_input.copy(deep=True)

    one_row_clean = clean_transactions(one_row_input)
    input_unchanged = one_row_input.equals(original_copy)

    print(f"Input unchanged after cleaning: {input_unchanged}")

    if not one_row_clean.empty:
        one_row_features = add_features(one_row_clean)
        print(f"One-row cleaning/features: PASS ({len(one_row_features)} row)")
    else:
        print("One-row cleaning/features: first raw row was removed by cleaning")


if __name__ == "__main__":
    raw = pd.read_csv("transactions_raw.csv")
    audit_raw_data()

    cleaned = clean_transactions(raw)
    featured = add_features(cleaned)

    print_part_a_results(raw, cleaned, featured)
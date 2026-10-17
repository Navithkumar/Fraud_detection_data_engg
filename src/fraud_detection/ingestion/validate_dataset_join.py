from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/raw")


def main() -> None:
    transaction_path = DATA_DIR / "train_transaction.csv"
    identity_path = DATA_DIR / "train_identity.csv"

    # Load only the join keys first to validate key quality.
    transactions = pd.read_csv(
        transaction_path,
        usecols=["TransactionID"],
    )
    identity = pd.read_csv(
        identity_path,
        usecols=["TransactionID"],
    )

    for name, df in [
        ("transactions", transactions),
        ("identity", identity),
    ]:
        null_keys = df["TransactionID"].isna().sum()
        duplicate_keys = df["TransactionID"].duplicated().sum()

        print(f"\n{name.upper()}")
        print(f"Rows: {len(df):,}")
        print(f"Null TransactionIDs: {null_keys:,}")
        print(f"Duplicate TransactionIDs: {duplicate_keys:,}")

        if null_keys or duplicate_keys:
            raise ValueError(f"{name} failed TransactionID quality checks")

    # A left join preserves all transaction records.
    joined = transactions.merge(
        identity,
        on="TransactionID",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    matched = int((joined["_merge"] == "both").sum())
    unmatched = int((joined["_merge"] == "left_only").sum())

    print("\nJOIN VALIDATION")
    print(f"Transaction rows: {len(transactions):,}")
    print(f"Identity rows: {len(identity):,}")
    print(f"Matched transactions: {matched:,}")
    print(f"Transactions without identity: {unmatched:,}")
    print(f"Identity coverage: {matched / len(transactions) * 100:.2f}%")

    assert len(joined) == len(transactions), (
        "Left join changed the transaction row count"
    )


if __name__ == "__main__":
    main()

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/raw")
CHUNK_SIZE = 100_000


def profile_csv(filename: str) -> None:
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    row_count = 0
    null_counts = None
    id_min = None
    id_max = None
    unique_ids = set()

    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
        row_count += len(chunk)

        current_nulls = chunk.isna().sum()
        null_counts = (
            current_nulls
            if null_counts is None
            else null_counts.add(current_nulls, fill_value=0)
        )

        if "TransactionID" in chunk.columns:
            ids = chunk["TransactionID"].dropna()
            if not ids.empty:
                chunk_min = ids.min()
                chunk_max = ids.max()
                id_min = chunk_min if id_min is None else min(id_min, chunk_min)
                id_max = chunk_max if id_max is None else max(id_max, chunk_max)
                unique_ids.update(ids.tolist())

        print(f"\rProcessed {row_count:,} rows from {filename}", end="")

    print(f"\n\nFILE: {filename}")
    print(f"Rows: {row_count:,}")
    print(f"Columns: {len(null_counts)}")
    print("\nHighest full-file null percentages:")

    null_pct = (null_counts / row_count * 100).sort_values(ascending=False)
    print(null_pct.head(20).round(2).to_string())

    if id_min is not None:
        print(f"\nTransactionID minimum: {id_min}")
        print(f"TransactionID maximum: {id_max}")
        print(f"Unique TransactionIDs: {len(unique_ids):,}")
        print(f"Duplicate TransactionIDs: {row_count - len(unique_ids):,}")


def main() -> None:
    for filename in (
        "train_transaction.csv",
        "train_identity.csv",
    ):
        profile_csv(filename)


if __name__ == "__main__":
    main()

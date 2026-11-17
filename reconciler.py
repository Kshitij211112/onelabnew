"""Payment and settlement reconciliation with categorized exceptions."""

from __future__ import annotations

import pandas as pd


PLATFORM_COLUMNS = {
    "transaction_id",
    "timestamp",
    "type",
    "gross_amount",
    "net_amount",
    "status",
}
BANK_COLUMNS = {
    "settlement_id",
    "transaction_id",
    "original_transaction_id",
    "settlement_date",
    "type",
    "settled_amount",
}
EXCEPTION_COLUMNS = [
    "category",
    "transaction_id",
    "variance_amount",
    "explanation",
]
CATEGORIES = (
    "CROSS_MONTH_TIMING",
    "CUMULATIVE_ROUNDING_DRIFT",
    "DUPLICATE_ENTRY",
    "ORPHAN_REFUND",
    "UNMATCHED_OTHER",
)


def _validate_columns(
    frame: pd.DataFrame, required: set[str], source_name: str
) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {', '.join(sorted(missing))}")


def _money(value: object) -> float:
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0).iloc[0])


def _validate_numeric_column(frame: pd.DataFrame, column: str, source: str) -> None:
    values = pd.to_numeric(frame[column], errors="coerce")
    invalid = values.isna() | values.isin([float("inf"), float("-inf")])
    if invalid.any():
        rows = ", ".join(str(index) for index in frame.index[invalid][:5])
        raise ValueError(f"{source} has invalid {column} values at row(s): {rows}")


def _identifier(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _exception(
    category: str, transaction_id: object, variance: float, explanation: str
) -> dict[str, object]:
    return {
        "category": category,
        "transaction_id": str(transaction_id),
        "variance_amount": round(float(variance), 6),
        "explanation": explanation,
    }


def reconcile(
    platform: pd.DataFrame, bank: pd.DataFrame
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Reconcile platform and bank rows; amounts use platform-minus-bank sign."""
    _validate_columns(platform, PLATFORM_COLUMNS, "Platform transactions")
    _validate_columns(bank, BANK_COLUMNS, "Bank settlements")

    platform = platform.copy()
    bank = bank.copy()
    _validate_numeric_column(platform, "net_amount", "Platform transactions")
    _validate_numeric_column(bank, "settled_amount", "Bank settlements")
    platform["net_amount"] = pd.to_numeric(platform["net_amount"])
    bank["settled_amount"] = pd.to_numeric(bank["settled_amount"])
    platform["timestamp"] = pd.to_datetime(platform["timestamp"], errors="coerce")
    bank["settlement_date"] = pd.to_datetime(bank["settlement_date"], errors="coerce")
    if platform["timestamp"].isna().any():
        raise ValueError("Platform transactions contains invalid timestamp values.")
    if bank["settlement_date"].isna().any():
        raise ValueError("Bank settlements contains invalid settlement_date values.")

    exceptions: list[dict[str, object]] = []
    platform_counts = platform["transaction_id"].value_counts()
    bank_counts = bank["transaction_id"].value_counts()
    duplicate_ids: set[str] = set()

    for source_name, counts, source in (
        ("platform", platform_counts, platform),
        ("bank", bank_counts, bank),
    ):
        for transaction_id, count in counts[counts > 1].items():
            transaction_id = str(transaction_id)
            duplicate_ids.add(transaction_id)
            repeated_rows = source.loc[source["transaction_id"].astype(str) == transaction_id]
            amount_column = "net_amount" if source_name == "platform" else "settled_amount"
            excess = _money(repeated_rows[amount_column].iloc[0]) * (int(count) - 1)
            exceptions.append(
                _exception(
                    "DUPLICATE_ENTRY",
                    transaction_id,
                    excess if source_name == "platform" else -excess,
                    f"{source_name.title()} source contains {count} rows for this transaction; "
                    f"the repeated amount is {abs(excess):.2f}.",
                )
            )

    platform_ids = set(platform["transaction_id"].dropna().astype(str))
    for row in bank.loc[bank["type"].astype(str).str.lower() == "refund"].itertuples():
        original_id = _identifier(row.original_transaction_id)
        if not original_id or original_id not in platform_ids:
            exceptions.append(
                _exception(
                    "ORPHAN_REFUND",
                    row.transaction_id,
                    -_money(row.settled_amount),
                    f"Bank refund references original transaction {original_id or '(blank)'}, "
                    "which is absent from platform transactions.",
                )
            )

    platform_by_id = platform.groupby("transaction_id", dropna=False)["net_amount"].sum()
    bank_by_id = bank.groupby("transaction_id", dropna=False)["settled_amount"].sum()
    all_ids = set(platform_by_id.index.dropna().astype(str)) | set(
        bank_by_id.index.dropna().astype(str)
    )
    rounding_diffs: list[tuple[str, float]] = []
    orphan_ids = {
        str(row.transaction_id)
        for row in bank.loc[bank["type"].astype(str).str.lower() == "refund"].itertuples()
        if not _identifier(row.original_transaction_id)
        or _identifier(row.original_transaction_id) not in platform_ids
    }
    matched_ids: set[str] = set()

    for transaction_id in sorted(all_ids):
        if transaction_id in duplicate_ids or transaction_id in orphan_ids:
            continue
        has_platform = transaction_id in platform_by_id.index
        has_bank = transaction_id in bank_by_id.index
        if not has_platform or not has_bank:
            exceptions.append(
                _exception(
                    "UNMATCHED_OTHER",
                    transaction_id,
                    _money(platform_by_id.get(transaction_id, 0))
                    - _money(bank_by_id.get(transaction_id, 0)),
                    "Transaction appears in only one source.",
                )
            )
            continue

        platform_amount = _money(platform_by_id[transaction_id])
        bank_amount = _money(bank_by_id[transaction_id])
        difference = platform_amount - bank_amount
        platform_rows = platform.loc[platform["transaction_id"].astype(str) == transaction_id]
        bank_rows = bank.loc[bank["transaction_id"].astype(str) == transaction_id]
        timestamps = platform_rows["timestamp"].dropna()
        settlements = bank_rows["settlement_date"].dropna()
        crosses_month = (
            not timestamps.empty
            and not settlements.empty
            and timestamps.dt.to_period("M").eq(pd.Period("2026-10", freq="M")).all()
            and settlements.dt.to_period("M").gt(timestamps.iloc[0].to_period("M")).any()
        )
        if crosses_month:
            exceptions.append(
                _exception(
                    "CROSS_MONTH_TIMING",
                    transaction_id,
                    difference,
                    f"October platform transaction settled on {settlements.max():%Y-%m-%d}, "
                    "after the reporting month closed.",
                )
            )

        if difference == 0:
            matched_ids.add(transaction_id)
        elif abs(difference) < 0.01:
            rounding_diffs.append((transaction_id, difference))
            matched_ids.add(transaction_id)
        else:
            exceptions.append(
                _exception(
                    "UNMATCHED_OTHER",
                    transaction_id,
                    difference,
                    f"Platform and bank amounts differ by {difference:.2f}.",
                )
            )

    if sum(diff for _, diff in rounding_diffs) > 0.01 or sum(
        diff for _, diff in rounding_diffs
    ) < -0.01:
        for transaction_id, difference in rounding_diffs:
            exceptions.append(
                _exception(
                    "CUMULATIVE_ROUNDING_DRIFT",
                    transaction_id,
                    difference,
                    f"Per-row difference is {difference:.4f}; the combined sub-cent "
                    "differences exceed the $0.01 tolerance.",
                )
            )

    exception_frame = pd.DataFrame(exceptions, columns=EXCEPTION_COLUMNS)
    if not exception_frame.empty:
        exception_frame["category"] = pd.Categorical(
            exception_frame["category"], categories=CATEGORIES, ordered=True
        )
        exception_frame = exception_frame.sort_values(
            ["category", "transaction_id"], kind="stable"
        ).reset_index(drop=True)

    platform_total = float(platform["net_amount"].sum())
    bank_total = float(bank["settled_amount"].sum())
    summary: dict[str, float | int] = {
        "platform_total": round(platform_total, 6),
        "bank_total": round(bank_total, 6),
        "net_variance": round(platform_total - bank_total, 6),
        "matched_count": len(matched_ids),
        "exception_count": len(exception_frame),
    }
    return summary, exception_frame


def bridge_breakdown(exceptions: pd.DataFrame) -> dict[str, float]:
    """Return gap contributions by category for the dashboard's bridge chart."""
    if exceptions.empty:
        return {category: 0.0 for category in CATEGORIES}
    values = exceptions.assign(
        variance_amount=pd.to_numeric(exceptions["variance_amount"], errors="coerce").fillna(0)
    )
    return {
        category: round(
            float(values.loc[values["category"].astype(str) == category, "variance_amount"].sum()),
            6,
        )
        for category in CATEGORIES
    }

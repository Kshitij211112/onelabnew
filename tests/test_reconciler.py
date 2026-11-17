from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate_data import build_data
from reconciler import bridge_breakdown, reconcile


def frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return build_data()


def category_rows(exceptions: pd.DataFrame, category: str) -> pd.DataFrame:
    return exceptions.loc[exceptions["category"].astype(str) == category]


def test_normal_t_plus_one_and_two_transactions_reconcile() -> None:
    platform = pd.DataFrame(
        [
            {
                "transaction_id": "TXN-A",
                "timestamp": "2026-10-10 10:00:00",
                "type": "payment",
                "gross_amount": 10.00,
                "net_amount": 9.50,
                "status": "succeeded",
            },
            {
                "transaction_id": "TXN-B",
                "timestamp": "2026-10-11 10:00:00",
                "type": "payment",
                "gross_amount": 20.00,
                "net_amount": 19.00,
                "status": "succeeded",
            },
        ]
    )
    bank = pd.DataFrame(
        [
            {
                "settlement_id": "SET-A",
                "transaction_id": "TXN-A",
                "original_transaction_id": "",
                "settlement_date": "2026-10-11",
                "type": "settlement",
                "settled_amount": 9.50,
            },
            {
                "settlement_id": "SET-B",
                "transaction_id": "TXN-B",
                "original_transaction_id": "",
                "settlement_date": "2026-10-13",
                "type": "settlement",
                "settled_amount": 19.00,
            },
        ]
    )
    summary, exceptions = reconcile(platform, bank)
    assert summary["net_variance"] == 0
    assert summary["matched_count"] == 2
    assert exceptions.empty


def test_generated_dataset_detects_each_planted_anomaly() -> None:
    platform, bank = frames()
    summary, exceptions = reconcile(platform, bank)

    assert set(exceptions["category"].astype(str)) == {
        "CROSS_MONTH_TIMING",
        "CUMULATIVE_ROUNDING_DRIFT",
        "DUPLICATE_ENTRY",
        "ORPHAN_REFUND",
    }
    assert category_rows(exceptions, "CROSS_MONTH_TIMING")["transaction_id"].tolist() == [
        "TXN-1028"
    ]
    drift = category_rows(exceptions, "CUMULATIVE_ROUNDING_DRIFT")
    assert set(drift["transaction_id"]) == {f"TXN-{number}" for number in range(1010, 1020)}
    assert drift["variance_amount"].sum() == pytest.approx(0.04)
    duplicate = category_rows(exceptions, "DUPLICATE_ENTRY")
    assert duplicate["transaction_id"].tolist() == ["TXN-1005"]
    assert duplicate["variance_amount"].iloc[0] == pytest.approx(120.00)
    orphan = category_rows(exceptions, "ORPHAN_REFUND")
    assert orphan["transaction_id"].tolist() == ["REF-9001"]
    assert orphan["variance_amount"].iloc[0] == pytest.approx(45.00)

    assert summary["net_variance"] == pytest.approx(165.04)
    assert sum(bridge_breakdown(exceptions).values()) == pytest.approx(
        summary["net_variance"]
    )
    assert not platform.empty and not bank.empty


def test_empty_frames_return_zero_summary_and_empty_report() -> None:
    platform, bank = frames()
    summary, exceptions = reconcile(platform.iloc[0:0], bank.iloc[0:0])
    assert summary == {
        "platform_total": 0.0,
        "bank_total": 0.0,
        "net_variance": 0.0,
        "matched_count": 0,
        "exception_count": 0,
    }
    assert exceptions.empty


def test_multiple_duplicates_are_reported_with_excess_amount() -> None:
    platform, bank = frames()
    duplicate_row = platform.loc[platform["transaction_id"] == "TXN-1005"].iloc[[0]]
    platform = pd.concat([platform, duplicate_row], ignore_index=True)
    _, exceptions = reconcile(platform, bank)
    duplicate = category_rows(exceptions, "DUPLICATE_ENTRY")
    assert duplicate["transaction_id"].tolist() == ["TXN-1005"]
    assert duplicate["variance_amount"].iloc[0] == pytest.approx(240.00)


def test_zero_amount_rows_match_without_an_exception() -> None:
    platform, bank = frames()
    platform = platform.iloc[0:0].copy()
    bank = bank.iloc[0:0].copy()
    platform.loc[0] = ["TXN-ZERO", "2026-10-10", "payment", 0.0, 0.0, "succeeded"]
    bank.loc[0] = ["SET-ZERO", "TXN-ZERO", "", "2026-10-11", "settlement", 0.0]
    summary, exceptions = reconcile(platform, bank)
    assert summary["net_variance"] == 0
    assert summary["matched_count"] == 1
    assert exceptions.empty


def test_regular_amount_mismatch_is_unmatched_other() -> None:
    platform, bank = frames()
    platform = platform.loc[platform["transaction_id"] == "TXN-1001"].copy()
    bank = bank.loc[bank["transaction_id"] == "TXN-1001"].copy()
    bank.loc[:, "settled_amount"] = 1.0
    _, exceptions = reconcile(platform, bank)
    assert category_rows(exceptions, "UNMATCHED_OTHER")["transaction_id"].tolist() == [
        "TXN-1001"
    ]


def test_invalid_amount_is_reported_instead_of_silently_zeroed() -> None:
    platform, bank = frames()
    platform = platform.loc[platform["transaction_id"] == "TXN-1001"].copy()
    bank = bank.loc[bank["transaction_id"] == "TXN-1001"].copy()
    platform["net_amount"] = "not-a-number"
    with pytest.raises(ValueError, match="invalid net_amount"):
        reconcile(platform, bank)

"""Generate deterministic October 2026 platform and bank reconciliation data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).parent / "data"


def build_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    platform_rows: list[dict[str, object]] = []
    bank_rows: list[dict[str, object]] = []

    for number in range(1001, 1051):
        transaction_id = f"TXN-{number}"
        if number == 1005:
            amount = 120.00
        elif 1010 <= number <= 1019:
            amount = 10.0040
        elif number == 1028:
            amount = 250.00
        else:
            amount = round(24.50 + (number * 7 % 180) + (number % 10) / 100, 2)

        day = 1 + ((number - 1001) % 29)
        hour = 9 + (number % 10)
        minute = (number * 13) % 60
        timestamp = pd.Timestamp(
            year=2026, month=10, day=day, hour=hour, minute=minute
        )
        if number == 1028:
            timestamp = pd.Timestamp("2026-10-31 22:15:00")

        platform_rows.append(
            {
                "transaction_id": transaction_id,
                "timestamp": timestamp,
                "type": "payment",
                "gross_amount": round(amount * 1.03, 2),
                "net_amount": amount,
                "status": "succeeded",
            }
        )

        settlement_date = timestamp.normalize() + pd.Timedelta(
            days=1 + (number % 2)
        )
        if number == 1028:
            settlement_date = pd.Timestamp("2026-11-02")
        settled_amount = round(amount, 2) if 1010 <= number <= 1019 else amount
        bank_rows.append(
            {
                "settlement_id": f"SET-{number}",
                "transaction_id": transaction_id,
                "original_transaction_id": "",
                "settlement_date": settlement_date,
                "type": "settlement",
                "settled_amount": settled_amount,
            }
        )

    platform = pd.DataFrame(platform_rows)
    bank = pd.DataFrame(bank_rows)

    # The source contains two identical platform records but only one settlement.
    platform = pd.concat(
        [platform, platform.loc[platform["transaction_id"] == "TXN-1005"]],
        ignore_index=True,
    )
    bank = pd.concat(
        [
            bank,
            pd.DataFrame(
                [
                    {
                        "settlement_id": "REF-9001",
                        "transaction_id": "REF-9001",
                        "original_transaction_id": "TXN-0000",
                        "settlement_date": pd.Timestamp("2026-10-20"),
                        "type": "refund",
                        "settled_amount": -45.00,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    return platform, bank


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    platform, bank = build_data()
    platform.to_csv(DATA_DIR / "platform_transactions.csv", index=False)
    bank.to_csv(DATA_DIR / "bank_settlements.csv", index=False)
    print(
        f"Generated {len(platform)} platform rows and {len(bank)} bank rows "
        f"in {DATA_DIR}"
    )


if __name__ == "__main__":
    main()

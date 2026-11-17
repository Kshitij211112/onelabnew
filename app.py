"""Interactive Streamlit dashboard for payment reconciliation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from reconciler import CATEGORIES, bridge_breakdown, reconcile


DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(page_title="Payment Reconciliation", page_icon="💸", layout="wide")
st.title("Payment Reconciliation Engine")
st.caption("Compare platform activity with bank settlement records.")

with st.expander("Assumptions & Rules", expanded=False):
    st.markdown(
        """
        - Amounts are compared as **platform net amount minus bank settled amount**.
        - Matched settlements may arrive T+1 or T+2; November settlements for
          October transactions are reported as cross-month timing exceptions.
        - Individual differences below $0.01 are treated as rounding candidates;
          they are exceptions only when their combined signed difference exceeds $0.01.
        - Duplicate source rows, refunds without a known original transaction, and
          all other missing or mismatched records are reported separately.
        - The bridge shows dollar impact by exception type; timing-only items can have
          zero dollar impact. “Reconciled Gap” is the remaining total after attribution.
        """
    )


@st.cache_data
def load_default_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    platform_path = DATA_DIR / "platform_transactions.csv"
    bank_path = DATA_DIR / "bank_settlements.csv"
    if not platform_path.exists() or not bank_path.exists():
        raise FileNotFoundError(
            "Default data is missing. Run `python generate_data.py` from the project folder."
        )
    return pd.read_csv(platform_path), pd.read_csv(bank_path)


default_platform, default_bank = load_default_data()
with st.sidebar:
    st.header("Data sources")
    platform_upload = st.file_uploader("Platform transactions CSV", type="csv")
    bank_upload = st.file_uploader("Bank settlements CSV", type="csv")
    st.caption("Upload both files to replace the bundled October 2026 sample.")

if bool(platform_upload) != bool(bank_upload):
    st.warning("Upload both CSV files, or neither, to run reconciliation.")
    st.stop()

platform_df = (
    pd.read_csv(platform_upload) if platform_upload is not None else default_platform
)
bank_df = pd.read_csv(bank_upload) if bank_upload is not None else default_bank

try:
    summary, exceptions = reconcile(platform_df, bank_df)
except (ValueError, KeyError) as error:
    st.error(f"Could not reconcile the supplied data: {error}")
    st.stop()

gap = float(summary["net_variance"])
bridge = bridge_breakdown(exceptions)
reconciled_gap = gap - sum(bridge.values())

metric_columns = st.columns(4)
metric_columns[0].metric("Platform Total", f"${summary['platform_total']:,.2f}")
metric_columns[1].metric("Bank Total", f"${summary['bank_total']:,.2f}")
metric_columns[2].metric("Total Unexplained Gap", f"${gap:,.2f}")
metric_columns[3].metric("Reconciled Gap", f"${reconciled_gap:,.2f}")
st.caption(
    f"{summary['matched_count']} matched transaction IDs · "
    f"{summary['exception_count']} exception rows"
)

st.subheader("Gap attribution bridge")
labels = ["Net variance"] + list(CATEGORIES) + ["Reconciled gap"]
values = [gap] + [-bridge[category] for category in CATEGORIES] + [reconciled_gap]
measures = ["absolute"] + ["relative"] * len(CATEGORIES) + ["total"]
figure = go.Figure(
    go.Waterfall(
        name="Reconciliation",
        orientation="v",
        measure=measures,
        x=labels,
        y=values,
        text=[f"${value:,.2f}" for value in values],
        textposition="outside",
        connector={"line": {"color": "#9ca3af"}},
        increasing={"marker": {"color": "#dc2626"}},
        decreasing={"marker": {"color": "#16a34a"}},
        totals={"marker": {"color": "#2563eb"}},
    )
)
figure.update_layout(
    showlegend=False,
    height=420,
    yaxis_title="USD",
    margin={"l": 20, "r": 20, "t": 30, "b": 100},
    xaxis={"tickangle": -25},
)
st.plotly_chart(figure, use_container_width=True)

st.download_button(
    "Download exception report as CSV",
    data=exceptions.to_csv(index=False).encode("utf-8"),
    file_name="reconciliation_exceptions.csv",
    mime="text/csv",
    disabled=exceptions.empty,
)

st.subheader("Exceptions by category")
tabs = st.tabs(list(CATEGORIES))
for tab, category in zip(tabs, CATEGORIES):
    with tab:
        category_rows = exceptions.loc[
            exceptions["category"].astype(str) == category
        ].copy()
        if category_rows.empty:
            st.success("No exceptions in this category.")
            continue
        transaction_ids = sorted(category_rows["transaction_id"].astype(str).unique())
        selection = st.selectbox(
            "Filter transaction",
            ["All"] + transaction_ids,
            key=f"filter-{category}",
        )
        if selection != "All":
            category_rows = category_rows.loc[
                category_rows["transaction_id"].astype(str) == selection
            ]
        st.dataframe(category_rows, use_container_width=True, hide_index=True)

"""
Simplified CleanFoam application with improved layout and performance.

This version of the CleanFoam app focuses on an easy‑to‑use interface for
entering worker revenue data and calculating fees according to business
rules.  The design has been updated to use modern Streamlit layout
elements such as columns, expanders and a sidebar to separate primary
inputs from management actions.  Performance improvements include
calculating summaries once per interaction rather than on every widget
change, and better handling of duplicate names when deleting entries.

Use this application when you need a streamlined experience without
exposing internal identifiers or complex configurations.  For a more
feature‑rich version, refer to ``app.py``.
"""

import math
import uuid
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(page_title="CleanFoam", page_icon="✅", layout="wide", initial_sidebar_state="expanded")
st.title("CleanFoam")


# -----------------------------------------------------------------------------
# Session state initialization
# -----------------------------------------------------------------------------
def init_state() -> None:
    if "workers" not in st.session_state:
        st.session_state.workers: List[Dict[str, Any]] = []
    if "report_date" not in st.session_state:
        st.session_state.report_date = date.today()
    if "df" not in st.session_state:
        st.session_state.df: Optional[pd.DataFrame] = None
    if "metrics" not in st.session_state:
        st.session_state.metrics: Optional[Tuple[float, float, float]] = None


init_state()


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def clean_number(n: Any) -> Any:
    """Render integers without .0 and leave other types untouched."""
    try:
        fval = float(n)
        return int(fval) if fval.is_integer() else fval
    except Exception:
        return n


def compute_fee(total_value: float, withdrawn: float, custom_due: Optional[float]) -> float:
    """
    Apply the business fee rules to determine the CleanFoam fee.  If ``custom_due``
    is supplied, it overrides the computed fee.  Otherwise, fees are determined
    using half of the total and a lookup table, plus a fallback for totals
    ending in 5 or a default.
    """
    if custom_due is not None:
        return custom_due

    half_value = total_value / 2.0
    eps = 1e-6
    # Map half of the total to specific fees
    rules_half: Dict[float, float] = {
        40.0: 20.0,
        45.0: 20.0,
        50.0: 25.0,
        52.5: 27.5,
        55.0: 25.0,
    }
    for hv, fee in rules_half.items():
        if math.isclose(half_value, hv, abs_tol=eps):
            return fee
    # Additional rule for a total of 95
    if math.isclose(total_value, 95.0, abs_tol=eps):
        return 22.5
    # Totals ending in 5 trigger a 32.5 fee
    if int(total_value) % 10 == 5:
        return 32.5
    # Default fee
    return 30.0


def update_summary() -> None:
    """Generate the DataFrame and metrics from the workers list."""
    if not st.session_state.workers:
        st.session_state.df = None
        st.session_state.metrics = None
        return
    df = pd.DataFrame(st.session_state.workers)
    # Ensure numeric fields are numeric; fill missing values with zero
    for col in ["Total", "Withdrawn", "Remaining"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    st.session_state.df = df
    total_sum = df["Total"].sum()
    withdrawn_sum = df["Withdrawn"].sum()
    remaining_sum = df["Remaining"].sum()
    for_workers = withdrawn_sum + remaining_sum
    for_cleanfoam = total_sum - for_workers
    st.session_state.metrics = (total_sum, for_workers, for_cleanfoam)


def add_row(name: str, total: float, withdrawn: float, custom_due: Optional[float], note: str, entry_type: str) -> None:
    """
    Append a new entry to the workers list.  The ID is truncated to eight
    characters for simplicity.  After insertion the summary is updated.
    """
    wid = uuid.uuid4().hex[:8]
    if entry_type == "CF":
        row = {
            "ID": wid,
            "Worker": name,
            "Total": total,
            "Due": "",
            "Withdrawn": "",
            "Remaining": "",
            "Note": note,
        }
    else:
        fee = compute_fee(total, withdrawn, custom_due)
        half_value = total / 2.0
        remaining = (half_value - withdrawn) - fee
        row = {
            "ID": wid,
            "Worker": name,
            "Total": total,
            "Due": fee,
            "Withdrawn": withdrawn,
            "Remaining": remaining,
            "Note": note,
        }
    st.session_state.workers.append(row)
    update_summary()


def remove_row(row_id: str) -> None:
    """Delete a worker entry by its ID."""
    st.session_state.workers = [w for w in st.session_state.workers if w["ID"] != row_id]
    update_summary()


def clear_all() -> None:
    """Remove all entries."""
    st.session_state.workers = []
    update_summary()


# -----------------------------------------------------------------------------
# User inputs
# -----------------------------------------------------------------------------
with st.container():
    st.subheader("إدخالات")
    # Four columns for the main inputs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.session_state.report_date = st.date_input("📅 التاريخ", value=st.session_state.report_date)
    with c2:
        name = st.text_input("الاسم", placeholder="اسم العامل")
    with c3:
        total_val = st.number_input("الإجمالي", min_value=0.0, step=0.5, format="%.2f")
    with c4:
        withdrawn_val = st.number_input("المسحوب", min_value=0.0, step=0.5, format="%.2f")

    # Advanced options in a second row
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        custom_due_val = st.number_input("قيمة مستحقة مخصصة", min_value=0.0, step=0.5, format="%.2f")
        custom_due = None if custom_due_val == 0.0 else custom_due_val
    with c6:
        note = st.text_input("ملاحظة", placeholder="اختياري")
    with c7:
        entry_type = st.radio("نوع الإدخال", ("Worker", "CF"), horizontal=True)
    with c8:
        add_clicked = st.button("إضافة", type="primary")

    reset_clicked = st.button("إعادة تعيين العمال", type="secondary")

# Handle button actions
if add_clicked:
    if not name:
        st.error("الاسم مطلوب.")
    elif total_val <= 0 and entry_type == "Worker":
        st.error("يجب أن يكون الإجمالي أكبر من صفر.")
    else:
        add_row(name, total_val, withdrawn_val, custom_due, note, entry_type)
        st.success(f"تمت إضافة {name}")
        st.experimental_rerun()

if reset_clicked:
    clear_all()
    st.info("تم مسح جميع العمال.")
    st.experimental_rerun()


# -----------------------------------------------------------------------------
# Display table and metrics if data exists
# -----------------------------------------------------------------------------
if st.session_state.df is not None:
    df_display = st.session_state.df.copy()
    df_display_display = df_display[["Worker", "Total", "Due", "Withdrawn", "Remaining", "Note"]]
    for col in ["Total", "Due", "Withdrawn", "Remaining"]:
        df_display_display[col] = df_display_display[col].apply(clean_number)
    st.subheader("جدول العمال")
    st.caption(f"تاريخ التقرير: {st.session_state.report_date.strftime('%Y-%m-%d')}")
    # Style the note column to be bold when it contains data
    def highlight_notes(val: Any) -> str:
        return "font-weight: bold" if val else ""
    st.dataframe(
        df_display_display.style.applymap(highlight_notes, subset=["Note"]),
        use_container_width=True,
    )
    # Display metrics
    total_sum, for_workers, for_cleanfoam = st.session_state.metrics
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("الإجمالي", clean_number(total_sum))
    col_b.metric("للعاملين", clean_number(for_workers))
    col_c.metric("لـ CleanFoam", clean_number(for_cleanfoam))

    # Sidebar actions
    with st.sidebar:
        st.header("إجراءات")
        st.markdown("### حذف عامل")
        # Build user-friendly labels; number duplicates
        labels: List[str] = []
        id_map: Dict[str, str] = {}
        counts: Dict[str, int] = {}
        for _, r in df_display.iterrows():
            worker_name = r["Worker"]
            counts[worker_name] = counts.get(worker_name, 0) + 1
            label = f"{worker_name} #{counts[worker_name]}" if counts[worker_name] > 1 else worker_name
            labels.append(label)
            id_map[label] = r["ID"]
        sel = st.selectbox("اختر عاملًا للحذف", labels)
        if st.button("حذف", type="secondary"):
            sel_id = id_map[sel]
            remove_row(sel_id)
            st.success(f"تم حذف {sel}")
            st.experimental_rerun()
        # CSV download
        csv = df_display_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 تحميل CSV",
            data=csv,
            file_name=f"cleanfoam_{st.session_state.report_date.strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
else:
    st.info("لا يوجد عمال مضافون بعد.")
"""
Improved CleanFoam application.

This Streamlit application provides a refined user experience for tracking
worker revenue and fees in the CleanFoam business.  The design has been
improved to better organize inputs and outputs, while performance
enhancements minimize redundant computations and simplify the overall
workflow.  Key features include:

* A clean two–column layout for entering worker data, with optional
  advanced settings hidden behind an expander to avoid clutter.
* A sidebar that houses secondary actions such as deleting workers,
  resetting all data and downloading reports.  Keeping these controls
  separate makes the main interface less overwhelming.
* Computation of totals and metrics only once per run, avoiding
  unnecessary conversions and calculations.
* A helper function for fee calculation that implements the business
  logic from the original version but can be easily extended.
* Better handling of duplicate worker names when selecting entries to
  delete – duplicate names are disambiguated with numbering in the
  dropdown.

The application's text labels are written in Arabic to better serve
users in the Riyadh timezone, though you can adjust these as needed.
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
st.set_page_config(
    page_title="CleanFoam",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("CleanFoam")


# -----------------------------------------------------------------------------
# Session state initialization
# -----------------------------------------------------------------------------
def initialize_state() -> None:
    """Ensure that required keys exist in the Streamlit session state."""
    if "workers" not in st.session_state:
        # This list will hold dictionaries describing each worker row.
        st.session_state.workers: List[Dict[str, Any]] = []
    if "report_date" not in st.session_state:
        st.session_state.report_date = date.today()
    # Optionally cache a DataFrame and metrics so we don't recompute on every
    # render.  These are invalidated whenever the workers list changes.
    if "summary_df" not in st.session_state:
        st.session_state.summary_df: Optional[pd.DataFrame] = None
    if "summary_metrics" not in st.session_state:
        st.session_state.summary_metrics: Optional[Tuple[float, float, float]] = None


initialize_state()


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def clean_number(value: Any) -> Any:
    """
    Convert numeric values to integers when possible and leave non‐numeric
    values unchanged.  This helper is used to display numbers without
    trailing zeros (e.g., 20.0 → 20).  Strings and other types are returned
    untouched.
    """
    try:
        fval = float(value)
        return int(fval) if fval.is_integer() else fval
    except Exception:
        return value


def compute_fee(total_value: float, custom_due: Optional[float]) -> float:
    """
    Determine the fee owed to CleanFoam based on the total value and a
    potential custom fee override.  The logic follows business rules:

    * If a custom fee is provided and greater than zero, it is used.
    * Specific total values map to fixed fees via a lookup table.
    * Totals ending in 5 (e.g., 85, 95, 105) incur a fee of 32.5.
    * All other cases default to a fee of 30.0.

    Args:
        total_value: The total revenue amount for the worker.
        custom_due: A user supplied override for the fee, or ``None``.

    Returns:
        The fee owed to CleanFoam.
    """
    # Use the custom value if provided and positive
    if custom_due is not None and custom_due > 0:
        return custom_due

    # Lookup table for specific totals
    fee_table: Dict[float, float] = {
        80.0: 20.0,
        90.0: 20.0,
        95.0: 22.5,
        100.0: 25.0,
        105.0: 27.5,
        110.0: 25.0,
    }
    fee = fee_table.get(total_value)
    if fee is not None:
        return fee

    # If the total ends with 5 when truncated to an integer
    if int(total_value) % 10 == 5:
        return 32.5

    # Default fee
    return 30.0


def refresh_summary() -> None:
    """
    Build a pandas DataFrame from the current workers list and compute
    aggregated metrics.  The DataFrame and metrics are stored in session state
    for reuse.  Should be called whenever the workers list changes.
    """
    if not st.session_state.workers:
        st.session_state.summary_df = None
        st.session_state.summary_metrics = None
        return

    # Create DataFrame from the workers list
    df = pd.DataFrame(st.session_state.workers)
    # Ensure numeric columns are numeric; missing values become 0
    numeric_cols = ["Total", "Withdrawn", "Remaining"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    st.session_state.summary_df = df

    # Compute financial summary: totals for display
    total_sum = df["Total"].sum()
    withdrawn_sum = df["Withdrawn"].sum()
    remaining_sum = df["Remaining"].sum()
    for_workers = withdrawn_sum + remaining_sum
    for_cleanfoam = total_sum - for_workers
    st.session_state.summary_metrics = (total_sum, for_workers, for_cleanfoam)


def add_worker(
    name: str,
    total_value: float,
    withdrawn_val: float,
    custom_due: Optional[float],
    note: str,
    entry_type: str,
) -> None:
    """
    Append a new worker or CF entry to the session state.  Computes the fee
    and remaining amount based on entry type.  After adding, refreshes the
    cached summary.
    """
    wid = uuid.uuid4().hex[:8]
    if entry_type == "CF":
        new_row = {
            "ID": wid,
            "Worker": name,
            "Total": total_value,
            "Due": "",
            "Withdrawn": "",
            "Remaining": "",
            "Note": note,
            "EntryType": "CF",
        }
    else:
        # Compute fee and remaining values
        fee = compute_fee(total_value, custom_due)
        remaining = (total_value / 2.0) - withdrawn_val - fee
        new_row = {
            "ID": wid,
            "Worker": name,
            "Total": total_value,
            "Due": fee,
            "Withdrawn": withdrawn_val,
            "Remaining": remaining,
            "Note": note,
            "EntryType": "Standard",
        }

    st.session_state.workers.append(new_row)
    refresh_summary()


def delete_worker(worker_id: str) -> None:
    """Remove a worker by their internal ID and refresh the summary."""
    st.session_state.workers = [w for w in st.session_state.workers if w["ID"] != worker_id]
    refresh_summary()


def reset_workers() -> None:
    """Clear all workers and reset the summary."""
    st.session_state.workers = []
    refresh_summary()


# -----------------------------------------------------------------------------
# User interface: Inputs
# -----------------------------------------------------------------------------
# Report date selector at the top of the main page
with st.container():
    col_date, _ = st.columns([1, 5])
    with col_date:
        st.session_state.report_date = st.date_input(
            "📅 تاريخ التقرير",
            value=st.session_state.report_date,
            help="اختر تاريخ التقرير لتمييز التقرير المحمّل",
        )


# Expander for adding a new worker
with st.expander("➕ إضافة مدخل جديد", expanded=True):
    st.write("أدخل تفاصيل العامل أو إدخال CleanFoam (CF)")
    # Lay out the primary input fields in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        name = st.text_input("اسم العامل", placeholder="أدخل اسم العامل", help="حقل إلزامي")
    with col2:
        total_value = st.number_input(
            "الإجمالي", min_value=0.0, step=0.5, format="%.2f", help="القيمة الإجمالية للعمل"
        )
    with col3:
        withdrawn_val = st.number_input(
            "المسحوب", min_value=0.0, step=0.5, format="%.2f", help="المبلغ المسحوب حتى الآن"
        )
    with col4:
        entry_type = st.radio(
            "نوع الإدخال", ("Standard", "CF"), index=0, horizontal=True, help="اختَر ‘CF’ لإدخال CleanFoam"
        )

    # Advanced options hidden under another expander
    with st.expander("⚙️ خيارات متقدمة"):
        adv1, adv2 = st.columns(2)
        with adv1:
            custom_due_val = st.number_input(
                "قيمة مستحقة مخصصة",
                min_value=0.0,
                step=0.5,
                format="%.2f",
                help="يمكنك إدخال قيمة مستحقة مخصصة بدلًا من القواعد الافتراضية",
            )
            custom_due = custom_due_val if custom_due_val > 0.0 else None
        with adv2:
            note_text = st.text_input("ملاحظة", placeholder="أدخل أي ملاحظات إضافية", help="اختياري")

    # Submit button for adding a worker
    add_clicked = st.button("إضافة", type="primary")
    if add_clicked:
        if not name:
            st.error("يجب إدخال اسم العامل.")
        elif total_value <= 0 and entry_type == "Standard":
            st.error("يجب أن تكون القيمة الإجمالية أكبر من صفر.")
        else:
            add_worker(name, total_value, withdrawn_val, custom_due, note_text, entry_type)
            st.success(f"تمت إضافة {name} بنجاح!")
            st.experimental_rerun()


# -----------------------------------------------------------------------------
# Main display: Table and metrics
# -----------------------------------------------------------------------------
if st.session_state.summary_df is None or st.session_state.summary_metrics is None:
    st.info("لا توجد بيانات بعد. استخدم النموذج أعلاه لإضافة مدخلات.")
else:
    df_display = st.session_state.summary_df.copy()
    # Prepare display DataFrame: convert numeric columns for nicer presentation
    for col in ["Total", "Due", "Withdrawn", "Remaining"]:
        df_display[col] = df_display[col].apply(clean_number)
    # Drop internal columns and reorder for display
    display_cols = ["Worker", "Total", "Due", "Withdrawn", "Remaining", "Note"]
    st.subheader("جدول العمال")
    st.caption(f"تاريخ التقرير: {st.session_state.report_date.strftime('%Y-%m-%d')}")
    # Style: bold the Note column if it contains text
    def highlight_note(val: Any) -> str:
        return "font-weight: bold" if val else ""

    st.dataframe(
        df_display[display_cols].style.applymap(highlight_note, subset=["Note"]),
        use_container_width=True,
    )

    # Show metrics in three columns
    total_sum, for_workers, for_cleanfoam = st.session_state.summary_metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("الإجمالي", f"{total_sum:,.2f}")
    m2.metric("للعاملين", f"{for_workers:,.2f}")
    m3.metric("لـ CleanFoam", f"{for_cleanfoam:,.2f}")


# -----------------------------------------------------------------------------
# Sidebar: Settings and actions
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("الإعدادات والأوامر")
    # Delete worker section
    if st.session_state.summary_df is not None:
        st.subheader("حذف مدخل")
        # Build labels for workers; duplicate names get numbered
        labels: List[str] = []
        id_map: Dict[str, str] = {}
        name_counts: Dict[str, int] = {}
        for _, row in st.session_state.summary_df.iterrows():
            worker_name = row["Worker"]
            name_counts[worker_name] = name_counts.get(worker_name, 0) + 1
            label = f"{worker_name} #{name_counts[worker_name]}" if name_counts[worker_name] > 1 else worker_name
            labels.append(label)
            id_map[label] = row["ID"]
        selected_label = st.selectbox("اختر مدخلاً للحذف", options=labels, index=None)
        delete_clicked = st.button("حذف", type="secondary", disabled=selected_label is None)
        if delete_clicked and selected_label:
            delete_worker(id_map[selected_label])
            st.success(f"تم حذف {selected_label}")
            st.experimental_rerun()

    # Divider
    st.markdown("---")
    st.subheader("إجراءات عامة")
    # Reset button
    if st.button("إعادة تعيين الكل"):
        if st.session_state.workers:
            reset_workers()
            st.success("تمت إعادة تعيين جميع البيانات.")
            st.experimental_rerun()

    # Download CSV button
    if st.session_state.summary_df is not None:
        csv_df = st.session_state.summary_df[["Worker", "Total", "Due", "Withdrawn", "Remaining", "Note"]]
        csv = csv_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 تحميل تقرير CSV",
            data=csv,
            file_name=f"cleanfoam_report_{st.session_state.report_date.strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="قم بتحميل تقرير CSV للبيانات الحالية",
        )


if __name__ == "__main__":
    # The Streamlit script executes top‑to‑bottom on page load.  The functions
    # defined above handle all interactions, so no additional logic is needed
    # here.  We include a pass statement for clarity.
    pass
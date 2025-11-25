import uuid
import math
from datetime import date

import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="CleanFoam – Dashboard",
    layout="wide",
    page_icon="🧼",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
        .main-title {
            font-size: 36px; 
            font-weight: 900; 
            margin-bottom: -10px;
        }
        .sub {
            font-size: 16px; 
            opacity: 0.7;
        }
        .box {
            padding: 25px; 
            background: #ffffff; 
            border-radius: 12px; 
            box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        }
        .metric-box {
            padding: 20px;
            border-radius: 12px;
            background: #f7f9fa;
            text-align:center;
            border: 1px solid #eee;
        }
        .metric-value {
            font-size: 28px; 
            font-weight: 700;
        }
        .metric-label {
            font-size: 14px; 
            opacity: 0.6;
        }
        .note-style {
            font-weight:600;
            color:#444;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------
if "workers" not in st.session_state:
    st.session_state.workers = []

if "report_date" not in st.session_state:
    st.session_state.report_date = date.today()


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def clean_number(n):
    if isinstance(n, (int, float)):
        return int(n) if float(n).is_integer() else round(float(n), 2)
    return n


def compute_fee(total_value: float, custom_due: float | None) -> float:
    if custom_due and custom_due > 0:
        return custom_due

    rules = {80: 20, 90: 20, 95: 22.5, 100: 25, 105: 27.5, 110: 25}
    if total_value in rules:
        return rules[total_value]

    if int(total_value) % 10 == 5:
        return 32.5

    return 30.0


# ---------------------------------------------------------
# SIDEBAR – ACTIONS & CONTROL
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Controls")

    st.session_state.report_date = st.date_input(
        "Report Date",
        value=st.session_state.report_date,
    )

    st.divider()

    # delete section
    st.subheader("🗑 Delete Worker")
    if st.session_state.workers:
        labels = [f"{w['Worker']} (Total {w['Total']})" for w in st.session_state.workers]
        selected_to_delete = st.selectbox("Select", labels)

        if st.button("Delete", use_container_width=True):
            index = labels.index(selected_to_delete)
            del st.session_state.workers[index]
            st.success("Worker deleted.")
            st.experimental_rerun()
    else:
        st.caption("No workers yet.")

    st.divider()

    if st.button("🚨 Reset All", use_container_width=True):
        st.session_state.workers = []
        st.experimental_rerun()

    if st.session_state.workers:
        df_csv = pd.DataFrame(st.session_state.workers)
        st.download_button(
            "⬇️ Download CSV",
            df_csv.to_csv(index=False).encode("utf-8"),
            "CleanFoam_Report.csv",
            use_container_width=True
        )


# ---------------------------------------------------------
# MAIN TITLE
# ---------------------------------------------------------
st.markdown('<div class="main-title">CleanFoam Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Efficient Worker Revenue Tracking System</div>', unsafe_allow_html=True)
st.write("")


# ---------------------------------------------------------
# TABS (Input – Table – Summary)
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["➕ Add Worker", "📄 Workers Table", "📊 Summary"])


# ---------------------------------------------------------
# TAB 1 — ADD WORKER
# ---------------------------------------------------------
with tab1:
    st.markdown('<div class="box">', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        name = st.text_input("Worker Name")
        entry_type = st.radio("Entry Type", ["Standard", "CF"], horizontal=True)

    with c2:
        total_value = st.number_input("Total (Required)", min_value=0.0, step=0.5, format="%.2f")
        withdrawn_val = st.number_input("Withdrawn", min_value=0.0, step=0.5, format="%.2f")

    with c3:
        due_custom = st.number_input("Custom Due (Optional)", min_value=0.0, step=0.5, format="%.2f")
        note = st.text_input("Note (Optional)")

    st.write("")
    submit = st.button("Add Worker", type="primary", use_container_width=True)

    if submit:
        if not name:
            st.error("Worker name is required.")
        elif entry_type == "Standard" and total_value <= 0:
            st.error("Total value must be greater than 0.")
        else:
            wid = uuid.uuid4().hex

            if entry_type == "CF":
                row = {
                    "ID": wid,
                    "Worker": name,
                    "Total": total_value,
                    "Due": "",
                    "Withdrawn": "",
                    "Remaining": "",
                    "Note": note
                }
            else:
                fee = compute_fee(total_value, due_custom if due_custom > 0 else None)
                remaining = (total_value / 2) - withdrawn_val - fee

                row = {
                    "ID": wid,
                    "Worker": name,
                    "Total": total_value,
                    "Due": fee,
                    "Withdrawn": withdrawn_val,
                    "Remaining": remaining,
                    "Note": note
                }

            st.session_state.workers.append(row)
            st.success(f"{name} added successfully!")
            st.experimental_rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# TAB 2 — WORKERS TABLE
# ---------------------------------------------------------
with tab2:
    st.markdown('<div class="box">', unsafe_allow_html=True)

    if not st.session_state.workers:
        st.info("No workers yet.")
    else:
        df = pd.DataFrame(st.session_state.workers).copy()

        df_display = df.copy()
        for col in ["Total", "Due", "Withdrawn", "Remaining"]:
            df_display[col] = pd.to_numeric(df_display[col], errors='coerce').fillna("").apply(clean_number)

        st.dataframe(
            df_display[["Worker", "Total", "Due", "Withdrawn", "Remaining", "Note"]],
            use_container_width=True,
            hide_index=True
        )

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# TAB 3 — SUMMARY
# ---------------------------------------------------------
with tab3:
    st.markdown('<div class="box">', unsafe_allow_html=True)

    if not st.session_state.workers:
        st.info("No workers available.")
    else:
        df = pd.DataFrame(st.session_state.workers)
        df_num = df.copy()

        for col in ["Total", "Withdrawn", "Remaining"]:
            df_num[col] = pd.to_numeric(df_num[col], errors="coerce").fillna(0)

        total_sum = df_num["Total"].sum()
        withdrawn_sum = df_num["Withdrawn"].sum()
        remaining_sum = df_num["Remaining"].sum()
        for_workers = withdrawn_sum + remaining_sum
        for_cleanfoam = total_sum - for_workers

        c1, c2, c3 = st.columns(3)

        c1.markdown(f"<div class='metric-box'><div class='metric-value'>{total_sum:.2f}</div><div class='metric-label'>Total Revenue</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-box'><div class='metric-value'>{for_workers:.2f}</div><div class='metric-label'>For Workers</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-box'><div class='metric-value'>{for_cleanfoam:.2f}</div><div class='metric-label'>For CleanFoam</div></div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

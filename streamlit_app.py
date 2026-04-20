import io
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


REQUIRED_COLUMNS = [
    "Item",
    "ProdResourceID",
    "Whse",
    "F Wk3",
    "Sum of SI Wk3",
    "Sum of SI-SS Wk3",
    "Average of SS Wk3",
]


@dataclass
class PriorityRule:
    mode: str   # "SI" or "SS"
    value: float  # SI cover percent (0-100) or SS target percent (e.g. 126 for 126%)


def normalize_pct(v: float) -> float:
    return float(v)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {', '.join(missing)}")


def load_input(path_or_buffer) -> pd.DataFrame:
    df = pd.read_excel(path_or_buffer, sheet_name="Sheet1")
    df = normalize_columns(df)
    validate_columns(df)
    return df


def find_vendor_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if "vendor" in c.lower()]


def compute_ss_ratio(si: float, ss: float) -> float:
    if ss == 0:
        return 0.0
    return si / ss


def compute_priority_target_f_after(row: pd.Series, rule: PriorityRule) -> float:
    f_orig = float(row["F Wk3 Original"])
    current_si = float(row["Current SI"])
    ss = float(row["Average of SS Wk3"])

    if rule.mode == "SI":
        cover = max(0.0, min(100.0, float(rule.value))) / 100.0
        target = f_orig - (current_si * cover)
    else:
        target_ratio = max(0.0, float(rule.value)) / 100.0
        target_si_after = target_ratio * ss
        target = f_orig + (target_si_after - current_si)

    return max(0.0, target)


def allocate_item(group: pd.DataFrame, rules: Dict[str, PriorityRule]) -> pd.DataFrame:
    group = group.copy()
    group["Whse"] = group["Whse"].astype(str)

    group["F Wk3"] = pd.to_numeric(group["F Wk3"], errors="coerce").fillna(0)
    group["Sum of SI Wk3"] = pd.to_numeric(group["Sum of SI Wk3"], errors="coerce").fillna(0)
    group["Sum of SI-SS Wk3"] = pd.to_numeric(group["Sum of SI-SS Wk3"], errors="coerce").fillna(0)
    group["Average of SS Wk3"] = pd.to_numeric(group["Average of SS Wk3"], errors="coerce").fillna(0)

    result = group.copy()
    result["F Wk3 Original"] = result["F Wk3"].round().astype(int)
    result["Current SI"] = result["Sum of SI Wk3"]
    result["Current SS%"] = result.apply(
        lambda r: compute_ss_ratio(float(r["Current SI"]), float(r["Average of SS Wk3"])), axis=1
    )

    total_firm = int(result["F Wk3 Original"].sum())
    result["Firm PO Total"] = total_firm

    result["F Wk3 After Destination Change"] = 0
    result["Priority Rule Mode"] = ""
    result["Priority Rule Value"] = np.nan
    result["Priority Target F After"] = np.nan
    result["Remaining Unallocated PO"] = 0

    # Start from zero firm everywhere, then redistribute the full firm pool
    remaining = total_firm

    # Priority first
    priority_rows = []
    for idx in result.index:
        wh = str(result.at[idx, "Whse"])
        if wh in rules:
            rule = rules[wh]
            result.at[idx, "Priority Rule Mode"] = rule.mode
            result.at[idx, "Priority Rule Value"] = float(rule.value)
            result.at[idx, "Priority Target F After"] = compute_priority_target_f_after(result.loc[idx], rule)
            priority_rows.append(idx)

    # Allocate 1 PCS at a time to priority warehouses until each reaches target or remaining hits 0
    while remaining > 0 and priority_rows:
        pending = []
        for idx in priority_rows:
            current_after = float(result.at[idx, "F Wk3 After Destination Change"])
            target_after = float(result.at[idx, "Priority Target F After"])
            if current_after < target_after:
                current_si_after = float(result.at[idx, "Current SI"]) + (current_after - float(result.at[idx, "F Wk3 Original"]))
                ss_after = compute_ss_ratio(current_si_after, float(result.at[idx, "Average of SS Wk3"]))
                pending.append((ss_after, str(result.at[idx, "Whse"]), idx))
        if not pending:
            break
        pending.sort(key=lambda x: (x[0], x[1]))
        idx = pending[0][2]
        result.at[idx, "F Wk3 After Destination Change"] += 1
        remaining -= 1

    # Non-priority phase 1: pull lowest SI toward 0
    non_priority_rows = [idx for idx in result.index if idx not in priority_rows]
    while remaining > 0 and non_priority_rows:
        pending = []
        for idx in non_priority_rows:
            f_after = float(result.at[idx, "F Wk3 After Destination Change"])
            si_after = float(result.at[idx, "Current SI"]) + (f_after - float(result.at[idx, "F Wk3 Original"]))
            if si_after < 0:
                pending.append((si_after, str(result.at[idx, "Whse"]), idx))
        if not pending:
            break
        pending.sort(key=lambda x: (x[0], x[1]))
        idx = pending[0][2]
        result.at[idx, "F Wk3 After Destination Change"] += 1
        remaining -= 1

    # Non-priority phase 2: lowest SS% first
    while remaining > 0 and non_priority_rows:
        pending = []
        for idx in non_priority_rows:
            f_after = float(result.at[idx, "F Wk3 After Destination Change"])
            si_after = float(result.at[idx, "Current SI"]) + (f_after - float(result.at[idx, "F Wk3 Original"]))
            ss_after = compute_ss_ratio(si_after, float(result.at[idx, "Average of SS Wk3"]))
            pending.append((ss_after, str(result.at[idx, "Whse"]), idx))
        if not pending:
            break
        pending.sort(key=lambda x: (x[0], x[1]))
        idx = pending[0][2]
        result.at[idx, "F Wk3 After Destination Change"] += 1
        remaining -= 1

    # If only priority rows exist and there is still remaining, keep allocating to lowest SS% among priority
    while remaining > 0 and priority_rows and not non_priority_rows:
        pending = []
        for idx in priority_rows:
            f_after = float(result.at[idx, "F Wk3 After Destination Change"])
            si_after = float(result.at[idx, "Current SI"]) + (f_after - float(result.at[idx, "F Wk3 Original"]))
            ss_after = compute_ss_ratio(si_after, float(result.at[idx, "Average of SS Wk3"]))
            pending.append((ss_after, str(result.at[idx, "Whse"]), idx))
        pending.sort(key=lambda x: (x[0], x[1]))
        idx = pending[0][2]
        result.at[idx, "F Wk3 After Destination Change"] += 1
        remaining -= 1

    result["Net Destination Change"] = result["F Wk3 After Destination Change"] - result["F Wk3 Original"]
    result["Current SI After"] = result["Current SI"] + result["Net Destination Change"]
    result["SS % After"] = result.apply(
        lambda r: compute_ss_ratio(float(r["Current SI After"]), float(r["Average of SS Wk3"])), axis=1
    )

    # Hard check: total before = total after
    if int(result["F Wk3 Original"].sum()) != int(result["F Wk3 After Destination Change"].sum()):
        raise ValueError("Tổng F Wk3 trước/sau không bằng nhau.")

    return result


def build_detail_output(detail: pd.DataFrame) -> pd.DataFrame:
    vendor_cols = find_vendor_columns(detail)

    preferred = [
        "Item",
        "ProdResourceID",
        "Whse",
        "F Wk3",
        "Sum of SI Wk3",
        "Sum of SI-SS Wk3",
        "Average of SS Wk3",
        "Current SI",
        "Current SS%",
        "Firm PO Total",
        "F Wk3 Original",
        "F Wk3 After Destination Change",
        "Net Destination Change",
        "Current SI After",
        "SS % After",
        "Remaining Unallocated PO",
        "Priority Rule Mode",
        "Priority Rule Value",
        "Priority Target F After",
    ]

    cols = []
    for c in preferred:
        if c in detail.columns and c not in cols:
            cols.append(c)

    for c in detail.columns:
        if c not in cols and c not in vendor_cols:
            cols.append(c)

    for c in vendor_cols:
        if c not in cols:
            cols.append(c)

    return detail[cols].copy()


def build_summary(detail: pd.DataFrame) -> pd.DataFrame:
    return (
        detail.groupby("Item", as_index=False)
        .agg(
            **{
                "Firm PO Total": ("Firm PO Total", "max"),
                "Original Total F": ("F Wk3 Original", "sum"),
                "After Total F": ("F Wk3 After Destination Change", "sum"),
                "Remaining Unallocated PO": ("Remaining Unallocated PO", "sum"),
            }
        )
    )


st.set_page_config(page_title="Destination Change Optimizer", layout="wide")
st.title("Destination Change Optimizer")
st.caption("%SS = SHIPPABLE INV / SAFETY STK = Current SI / Average of SS Wk3")

uploaded = st.file_uploader("Upload file input (.xlsx)", type=["xlsx"])

if uploaded is None:
    st.info("Tải file Excel lên để bắt đầu.")
    st.stop()

try:
    df = load_input(uploaded)
except Exception as e:
    st.error(f"Không đọc được file input: {e}")
    st.stop()

st.subheader("Preview input")
st.dataframe(df.head(50), use_container_width=True)

items = sorted(df["Item"].dropna().astype(str).unique().tolist())
selected_items = st.multiselect(
    "Chọn item cần áp priority rule",
    options=items,
    default=[],
)

item_rules: Dict[str, Dict[str, PriorityRule]] = {}
for item in selected_items:
    st.markdown(f"### Item {item}")
    sub = df[df["Item"].astype(str) == item].copy()
    whs = sorted(sub["Whse"].dropna().astype(str).unique().tolist())
    selected_whs = st.multiselect(
        f"Chọn 1 hoặc nhiều kho priority cho item {item}",
        options=whs,
        default=[],
        key=f"wh_{item}",
    )
    item_rules[item] = {}
    for wh in selected_whs:
        c1, c2 = st.columns(2)
        with c1:
            mode = st.selectbox(
                f"Mode - item {item} - WH {wh}",
                options=["SI", "SS"],
                key=f"mode_{item}_{wh}",
            )
        with c2:
            if mode == "SI":
                value = st.number_input(
                    f"% cover SI=0 - item {item} - WH {wh}",
                    min_value=0.0,
                    max_value=100.0,
                    value=100.0,
                    step=1.0,
                    key=f"value_{item}_{wh}",
                )
            else:
                value = st.number_input(
                    f"Target SS% - item {item} - WH {wh}",
                    min_value=0.0,
                    value=100.0,
                    step=1.0,
                    key=f"value_{item}_{wh}",
                )
        item_rules[item][wh] = PriorityRule(mode=mode, value=normalize_pct(value))

if st.button("Run Destination Change", type="primary"):
    try:
        detail_parts = []
        for item, group in df.groupby("Item", sort=False):
            rules = item_rules.get(str(item), {})
            detail_parts.append(allocate_item(group.copy(), rules))

        detail = pd.concat(detail_parts, ignore_index=True) if detail_parts else pd.DataFrame()
        detail_out = build_detail_output(detail)
        summary = build_summary(detail)

        output_buffer = io.BytesIO()
        with pd.ExcelWriter(output_buffer, engine="openpyxl") as writer:
            detail_out.to_excel(writer, index=False, sheet_name="Optimized Data")
            summary.to_excel(writer, index=False, sheet_name="Summary")
        output_buffer.seek(0)

        st.success("Đã chạy xong.")
        st.subheader("Optimized Data")
        st.dataframe(detail_out, use_container_width=True)

        st.subheader("Summary")
        st.dataframe(summary, use_container_width=True)

        st.download_button(
            "Download output Excel",
            data=output_buffer.getvalue(),
            file_name="destination_change_optimized.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as e:
        st.exception(e)

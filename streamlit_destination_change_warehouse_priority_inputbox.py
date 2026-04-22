import io
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


INPUT_SHEET_MAIN = "Sheet1"
VENDOR_FALLBACK_COL = "Vendor"


@dataclass
class PriorityRule:
    whse: str
    mode: str  # SI | SS
    value: float  # SI = % cover toward SI=0 as decimal; SS = target SS ratio as decimal


def normalize_whse(v) -> str:
    if pd.isna(v):
        return ""
    try:
        f = float(v)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return str(v).strip()


def normalize_pct(value) -> float:
    if value is None:
        return 0.0
    value = float(value)
    if value > 1:
        value = value / 100.0
    return value


def round_to_int_units(value: float) -> int:
    return int(round(float(value)))


def safe_ss_ratio(current_si: float, ss_target: float) -> float:
    """
    %SS = SHIPPABLE INV / SAFETY STK = Current SI / Average of SS Wk3
    """
    if ss_target <= 0:
        if current_si > 0:
            return math.inf
        if current_si < 0:
            return -math.inf
        return 0.0
    return current_si / ss_target


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def validate_columns(df: pd.DataFrame) -> None:
    required = [
        "Item",
        "ProdResourceID",
        "Whse",
        "F Wk3",
        "Sum of SI Wk3",
        "Average of SS Wk3",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột ở Sheet1: {missing}")


def find_vendor_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if "vendor" in c.lower()]


def load_input(uploaded_file) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file, sheet_name=INPUT_SHEET_MAIN)
    df = normalize_columns(df)
    validate_columns(df)

    # Keep Item as text so alphanumeric item codes are preserved and processed correctly.
    df["Item"] = df["Item"].map(lambda x: "" if pd.isna(x) else str(x).strip())
    df = df[df["Item"] != ""].copy()
    df["Whse"] = df["Whse"].map(normalize_whse)

    for c in ["F Wk3", "Sum of SI Wk3", "Average of SS Wk3"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    if "Sum of SI-SS Wk3" in df.columns:
        df["Sum of SI-SS Wk3"] = pd.to_numeric(df["Sum of SI-SS Wk3"], errors="coerce").fillna(0)
    else:
        df["Sum of SI-SS Wk3"] = pd.NA

    if not find_vendor_columns(df):
        df[VENDOR_FALLBACK_COL] = ""

    df["Current SI"] = df["Sum of SI Wk3"]
    df["Current SS%"] = df.apply(
        lambda r: safe_ss_ratio(float(r["Current SI"]), float(r["Average of SS Wk3"])),
        axis=1,
    )

    totals = (
        df.groupby("Item", as_index=False)["F Wk3"]
        .sum()
        .rename(columns={"F Wk3": "Firm PO Total"})
    )
    df = df.merge(totals, on="Item", how="left")
    return df


def current_si_after(row: dict) -> int:
    return int(row["current_si"] + (row["final_f"] - row["orig_f"]))


def current_ss_after(row: dict) -> float:
    return safe_ss_ratio(current_si_after(row), row["ss_target"])


def compute_priority_target_final(row: dict, rule: PriorityRule) -> int:
    orig_f = row["orig_f"]
    current_si = row["current_si"]
    ss_target = row["ss_target"]

    if rule.mode == "SI":
        pct = max(0.0, min(1.0, float(rule.value)))
        target_si_after = current_si * (1.0 - pct)
        final_f = orig_f + (target_si_after - current_si)
        return max(0, round_to_int_units(final_f))

    if rule.mode == "SS":
        target_ratio = max(0.0, float(rule.value))
        target_si_after = ss_target * target_ratio
        final_f = orig_f + (target_si_after - current_si)
        return max(0, round_to_int_units(final_f))

    return int(orig_f)


def build_rows(group: pd.DataFrame, item_rules: Dict[str, PriorityRule]) -> Tuple[List[dict], int]:
    rows: List[dict] = []
    for _, r in group.iterrows():
        row = {
            "item": r["Item"],
            "prod": r["ProdResourceID"],
            "whse": normalize_whse(r["Whse"]),
            "orig_f": round_to_int_units(r["F Wk3"]),
            "current_si": round_to_int_units(r["Current SI"]),
            "ss_target": float(r["Average of SS Wk3"]),
            "final_f": 0,
            "priority_rule_mode": "",
            "priority_rule_value": None,
            "priority_target_f_after": None,
        }
        rule = item_rules.get(row["whse"])
        if rule is not None:
            row["priority_rule_mode"] = rule.mode
            row["priority_rule_value"] = rule.value
            row["priority_target_f_after"] = compute_priority_target_final(row, rule)
        rows.append(row)

    total_f = round_to_int_units(group["Firm PO Total"].iloc[0])
    return rows, total_f


def choose_priority_recipient(rows: List[dict], priority_indices: List[int]) -> Optional[int]:
    candidates = []
    for idx in priority_indices:
        row = rows[idx]
        target = row.get("priority_target_f_after")
        if target is None:
            continue
        if row["final_f"] >= target:
            continue

        gap = target - row["final_f"]
        if row["priority_rule_mode"] == "SI":
            primary_metric = current_si_after(row)
            secondary_metric = current_ss_after(row)
        else:
            primary_metric = current_ss_after(row)
            secondary_metric = current_si_after(row)

        candidates.append((gap, primary_metric, secondary_metric, row["whse"], idx))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
    return candidates[0][4]


def choose_phase1_recipient(rows: List[dict], candidate_indices: List[int]) -> Optional[int]:
    candidates = []
    for idx in candidate_indices:
        after_si = current_si_after(rows[idx])
        if after_si < 0:
            ratio = safe_ss_ratio(after_si, rows[idx]["ss_target"])
            candidates.append((after_si, ratio, rows[idx]["whse"], idx))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    return candidates[0][3]


def choose_phase2_recipient(rows: List[dict], candidate_indices: List[int]) -> Optional[int]:
    candidates = []
    for idx in candidate_indices:
        after_si = current_si_after(rows[idx])
        ratio = safe_ss_ratio(after_si, rows[idx]["ss_target"])
        candidates.append((ratio, after_si, rows[idx]["whse"], idx))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    return candidates[0][3]


def allocate_item(group: pd.DataFrame, item_rules: Dict[str, PriorityRule]) -> pd.DataFrame:
    rows, total_f = build_rows(group, item_rules)
    remaining = total_f

    priority_indices = [i for i, r in enumerate(rows) if r["priority_rule_mode"]]
    non_priority_indices = [i for i, r in enumerate(rows) if not r["priority_rule_mode"]]

    # 1) Priority first
    while remaining > 0:
        idx = choose_priority_recipient(rows, priority_indices)
        if idx is None:
            break
        rows[idx]["final_f"] += 1
        remaining -= 1

    # 2) Non-priority phase 1: bring SI up to 0
    allocation_pool = non_priority_indices[:] if non_priority_indices else priority_indices[:]
    while remaining > 0:
        idx = choose_phase1_recipient(rows, allocation_pool)
        if idx is None:
            break
        rows[idx]["final_f"] += 1
        remaining -= 1

    # 3) Non-priority phase 2: balance by lowest SS%
    while remaining > 0:
        idx = choose_phase2_recipient(rows, allocation_pool)
        if idx is None:
            break
        rows[idx]["final_f"] += 1
        remaining -= 1

    out = group.copy().reset_index(drop=True)
    out["F Wk3 Original"] = out["F Wk3"].round().astype(int)
    out["F Wk3 After Destination Change"] = [r["final_f"] for r in rows]
    out["Net Destination Change"] = out["F Wk3 After Destination Change"] - out["F Wk3 Original"]
    out["Current SI After"] = out["Current SI"] + out["Net Destination Change"]
    out["SS % After"] = out.apply(
        lambda r: safe_ss_ratio(float(r["Current SI After"]), float(r["Average of SS Wk3"])),
        axis=1,
    )
    out["Remaining Unallocated PO"] = total_f - int(out["F Wk3 After Destination Change"].sum())
    out["Priority Rule Mode"] = [r["priority_rule_mode"] for r in rows]
    out["Priority Rule Value"] = [r["priority_rule_value"] for r in rows]
    out["Priority Target F After"] = [r["priority_target_f_after"] for r in rows]

    before_total = int(out["F Wk3 Original"].sum())
    after_total = int(out["F Wk3 After Destination Change"].sum())
    if before_total != after_total:
        raise ValueError(
            f"Item {out['Item'].iloc[0]}: tổng firm không bảo toàn, before={before_total}, after={after_total}."
        )

    return out


def build_detail_output(detail: pd.DataFrame) -> pd.DataFrame:
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
    vendor_cols = find_vendor_columns(detail)
    if not vendor_cols and "Vendor" in detail.columns:
        vendor_cols = ["Vendor"]

    final_cols = [c for c in preferred if c in detail.columns]
    for c in detail.columns:
        if c not in final_cols and c not in vendor_cols:
            final_cols.append(c)
    for c in vendor_cols:
        if c not in final_cols:
            final_cols.append(c)

    return detail[final_cols].copy()


def build_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, g in detail.groupby("Item", sort=True):
        rows.append(
            {
                "Item": g["Item"].iloc[0],
                "ProdResourceID": g["ProdResourceID"].iloc[0],
                "Firm PO Total": int(g["Firm PO Total"].iloc[0]),
                "Total F Before": int(g["F Wk3 Original"].sum()),
                "Total F After": int(g["F Wk3 After Destination Change"].sum()),
                "Min SI After": int(g["Current SI After"].min()),
                "Max SI After": int(g["Current SI After"].max()),
                "Min SS % After": float(g["SS % After"].min()),
                "Max SS % After": float(g["SS % After"].max()),
            }
        )
    return pd.DataFrame(rows)


def render_excel(detail: pd.DataFrame, summary: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        detail.to_excel(writer, index=False, sheet_name="Optimized Data")
        summary.to_excel(writer, index=False, sheet_name="Summary")
        logic = pd.DataFrame(
            [
                ["Current SI", "= Sum of SI Wk3"],
                ["Current SS%", "= Current SI / Average of SS Wk3"],
                ["F Wk3 After", "Firm cuối cùng của kho sau destination change"],
                ["Net Destination Change", "= F Wk3 After - F Wk3 Original"],
                ["Current SI After", "= Current SI + (F Wk3 After - F Wk3 Original)"],
                ["SS % After", "= Current SI After / Average of SS Wk3"],
                ["Priority mode SI", "Target là cover theo % hướng tới SI = 0"],
                ["Priority mode SS", "Target %SS dùng công thức SI / SS"],
                ["Multi-priority", "Nhiều kho priority được cấp từng 1 PCS cho đến khi đạt target hoặc hết firm"],
            ],
            columns=["Field", "Meaning"],
        )
        logic.to_excel(writer, index=False, sheet_name="Logic")
    output.seek(0)
    return output.getvalue()


def build_priority_rule_ui(warehouses: List[str]) -> Dict[str, PriorityRule]:
    st.subheader("Chọn kho ưu tiên")
    st.caption("Chọn kho, rồi nhập trực tiếp tỷ lệ % mong muốn cho từng kho.")
    selected_whs = st.multiselect(
        "Chọn một hoặc nhiều kho priority",
        options=warehouses,
        default=[],
    )

    rules: Dict[str, PriorityRule] = {}
    if selected_whs:
        st.markdown("### Thiết lập rule cho từng kho priority")
        for wh in selected_whs:
            with st.expander(f"Kho {wh}", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    mode = st.selectbox(
                        f"Mode cho kho {wh}",
                        options=["SI", "SS"],
                        key=f"mode_{wh}",
                    )
                with c2:
                    if mode == "SI":
                        value = st.number_input(
                            f"% cover tới SI=0 cho kho {wh}",
                            min_value=0.0,
                            max_value=100.0,
                            value=100.0,
                            step=1.0,
                            key=f"value_{wh}",
                            format="%.1f",
                        )
                    else:
                        value = st.number_input(
                            f"Target SS% cho kho {wh}",
                            min_value=0.0,
                            max_value=300.0,
                            value=100.0,
                            step=1.0,
                            key=f"value_{wh}",
                            format="%.1f",
                        )
                rules[str(wh)] = PriorityRule(
                    whse=str(wh),
                    mode=mode,
                    value=normalize_pct(value),
                )
                st.write(f"Rule hiện tại: **{mode}** — **{value:.1f}%**")
    return rules


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

warehouses = sorted(df["Whse"].dropna().astype(str).unique().tolist())
item_count = df["Item"].nunique()
st.write(f"Số item: **{item_count}** | Số kho phát hiện: **{len(warehouses)}**")

priority_rules = build_priority_rule_ui(warehouses)

if st.button("Run Destination Change", type="primary"):
    try:
        detail_parts = []
        for item, group in df.groupby("Item", sort=False):
            item_rules = {wh: rule for wh, rule in priority_rules.items() if wh in set(group["Whse"].astype(str).tolist())}
            detail_parts.append(allocate_item(group.copy(), item_rules))

        detail_full = pd.concat(detail_parts, ignore_index=True) if detail_parts else pd.DataFrame()
        detail = build_detail_output(detail_full)
        summary = build_summary(detail_full)

        before_all = int(detail_full["F Wk3 Original"].sum())
        after_all = int(detail_full["F Wk3 After Destination Change"].sum())
        if before_all != after_all:
            raise ValueError(f"Sai tổng Firm PO toàn file: before={before_all}, after={after_all}")

        output_bytes = render_excel(detail, summary)

        st.success("Đã chạy xong.")
        st.subheader("Optimized Data")
        st.dataframe(detail, use_container_width=True)

        st.subheader("Summary")
        st.dataframe(summary, use_container_width=True)

        st.download_button(
            "Download output Excel",
            data=output_bytes,
            file_name="destination_change_optimized.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as e:
        st.exception(e)

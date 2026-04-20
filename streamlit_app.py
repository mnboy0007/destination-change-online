import io
import importlib.util
from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
ENGINE_PATH = BASE_DIR / "destination_change_optimizer_ssratio.py"

spec = importlib.util.spec_from_file_location("dc_engine", ENGINE_PATH)
dc_engine = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(dc_engine)

st.set_page_config(page_title="Destination Change Optimizer", layout="wide")
st.title("Destination Change Optimizer")
st.caption("%SS = SHIPPABLE INV / SAFETY STK = Current SI / Average of SS Wk3")

uploaded = st.file_uploader("Upload file input (.xlsx)", type=["xlsx"])

if uploaded is not None:
    temp_input = BASE_DIR / "_streamlit_uploaded_input.xlsx"
    temp_input.write_bytes(uploaded.getvalue())

    try:
        df = dc_engine.load_input(str(temp_input))
    except Exception as e:
        st.error(f"Không đọc được file input: {e}")
        st.stop()

    st.subheader("Preview input")
    st.dataframe(df.head(50), use_container_width=True)

    all_wh = sorted(df["Whse"].dropna().astype(str).unique().tolist())
    priority_whs = st.multiselect(
        "Chọn 1 hoặc nhiều kho priority",
        options=all_wh,
        default=[],
    )

    rules: Dict[str, object] = {}
    if priority_whs:
        st.subheader("Thiết lập rule cho từng kho priority")
        for wh in priority_whs:
            c1, c2 = st.columns(2)
            with c1:
                mode = st.selectbox(
                    f"Mode - WH {wh}",
                    options=["SI", "SS"],
                    key=f"mode_{wh}"
                )
            with c2:
                if mode == "SI":
                    value = st.number_input(
                        f"% cover SI=0 - WH {wh}",
                        min_value=0.0,
                        max_value=100.0,
                        value=100.0,
                        step=1.0,
                        key=f"value_{wh}",
                    )
                else:
                    value = st.number_input(
                        f"Target SS% - WH {wh}",
                        min_value=0.0,
                        value=100.0,
                        step=1.0,
                        key=f"value_{wh}",
                    )
            rules[str(wh)] = dc_engine.PriorityRule(
                whse=str(wh),
                mode=mode,
                value=dc_engine.normalize_pct(value)
            )

    if st.button("Run Destination Change", type="primary"):
        try:
            detail_parts = []
            for _, group in df.groupby("Item", sort=False):
                detail_parts.append(dc_engine.allocate_item(group.copy(), rules))

            detail = pd.concat(detail_parts, ignore_index=True) if detail_parts else pd.DataFrame()
            detail_out = dc_engine.build_detail_output(detail)
            summary = dc_engine.build_summary(detail)

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
else:
    st.info("Tải file Excel lên để bắt đầu.")

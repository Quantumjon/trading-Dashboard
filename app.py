# -----start section 1 - Setup & Page Navigation-----

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("🧠 TBR Dashboard")

# Page Navigation
page = st.sidebar.radio("Select Page", ["Range Finder", "DCA Risk Calculator"])

# -----end section 1 - Setup & Page Navigation-----
# -----start section 2 - Range Finder Uploads + Filters-----

if page == "Range Finder":
    st.header("📊 Range Finder")

    st.sidebar.header("📂 CSV File Upload")
    num_instruments = st.sidebar.number_input("Number of Instruments", 1, 10, 1)
    timeframes = ["1 Year", "6 Months", "3 Months", "1 Month"]

    instrument_data = {}

    for i in range(1, num_instruments + 1):
        with st.sidebar.expander(f"Instrument {i}"):
            instrument_name = st.text_input(f"Name for Instrument {i}", value=f"Instrument_{i}")
            instrument_data[instrument_name] = {}

            for tf in timeframes:
                uploaded = st.file_uploader(f"{instrument_name} - {tf}", type="csv", key=f"{instrument_name}_{tf}")
                if uploaded:
                    df = pd.read_csv(uploaded)
                    if "Date" in df.columns:
                        df.rename(columns={"Date": "DayOfWeek"}, inplace=True)
                    instrument_data[instrument_name][tf] = df

    st.subheader("📖 Refined Table View")
    min_strike = st.number_input("Min Strike Rate", 0, 100, 75)
    risk_filter = st.radio("Risk Filter", ["All", "Low", "Moderate", "High"], horizontal=True)
    combined = []

# -----end section 2 - Range Finder Uploads + Filters-----
# -----start section 3 - Range Finder Core Logic (Merging, Metrics, Risk Score)-----

    for instrument, data in instrument_data.items():
        if all(tf in data for tf in timeframes):
            df = data["1 Month"].copy()
            df.rename(columns={
                "StrikeRate": "Strike_1M", "AvgMAE": "MAE_1M", "AvgMFE": "MFE_1M", "AvgRangePerc": "Range_1M"
            }, inplace=True)

            rename_map = {
                "3 Months": {"StrikeRate": "Strike_3M", "AvgMAE": "MAE_3M", "AvgMFE": "MFE_3M", "AvgRangePerc": "Range_3M"},
                "6 Months": {"StrikeRate": "Strike_6M", "AvgMAE": "MAE_6M", "AvgMFE": "MFE_6M", "AvgRangePerc": "Range_6M"},
                "1 Year": {"StrikeRate": "Strike_1Y", "AvgMAE": "MAE_1Y", "AvgMFE": "MFE_1Y", "AvgRangePerc": "Range_1Y"},
            }

            for tf, cols in rename_map.items():
                df_temp = data[tf].rename(columns=cols)

                # Keep only required columns: key join columns + renamed metric columns
                required = ["DayOfWeek", "RangeStart", "RangeEnd"] + list(cols.values())
                df_temp = df_temp.loc[:, required]

                df = df.merge(df_temp, on=["DayOfWeek", "RangeStart", "RangeEnd"], how="left")

            weights_strike = {"1M": 0.1, "3M": 0.2, "6M": 0.3, "1Y": 0.4}
            weights_mfe_mae = {"1M": 0.4, "3M": 0.3, "6M": 0.2, "1Y": 0.1}

            df["Weighted_Strike"] = sum(df[f"Strike_{k}"] * w for k, w in weights_strike.items())
            df["Weighted_MAE"] = sum(df[f"MAE_{k}"] * w for k, w in weights_mfe_mae.items())
            df["Weighted_MFE"] = sum(df[f"MFE_{k}"] * w for k, w in weights_mfe_mae.items())

            df["Reward/Risk Ratio"] = df["Weighted_MFE"] / df["Weighted_MAE"]
            df["Reward/Risk Ratio"] = df["Reward/Risk Ratio"].replace([np.inf, -np.inf], np.nan).fillna(0)

            df["Risk_Score"] = sum(
                (df[f"MAE_{k}"] / df[f"Range_{k}"]) * 100 * weights_strike[k] for k in weights_strike
            )
            df["Risk_Level"] = df["Risk_Score"].apply(lambda x: "Low" if x < 80 else "Moderate" if x <= 120 else "High")

            df["Instrument"] = instrument
            combined.append(df)

# -----end section 3 - Range Finder Core Logic (Merging, Metrics, Risk Score)-----
# -----start section 4 - Range Finder Display + Tooltip + Color Grading-----

    if combined:
        final_df = pd.concat(combined).reset_index(drop=True)

        with st.expander("ℹ️ Column Guide & Metric Logic"):
            st.markdown("""
### **Metric Definitions & Calculation Logic**

**Weighted Strike Rate**  
> Combines strike rate from all timeframes using:  
`(1Y × 40%) + (6M × 30%) + (3M × 20%) + (1M × 10%)`

**Weighted MFE & MAE**  
> Use recency-weighted logic to reflect current market behavior:  
`(1M = 40%, 3M = 30%, 6M = 20%, 1Y = 10%)`

**Reward/Risk Ratio**  
> `Weighted MFE ÷ Weighted MAE`  
- ≥ 1.80: Excellent (green)  
- 1.30–1.79: Tradable (yellow)  
- 1.00–1.29: Unreliable (orange)  
- < 1.00: Fail (red)

**Risk Level**  
> Based on drawdown vs. range (%):  
- Low: < 80  
- Moderate: 80–120  
- High: > 120
""")

        st.markdown("### 📊 Combined Refined Table (All Instruments)")
        display_cols = [
            "Instrument", "DayOfWeek", "RangeStart", "RangeEnd",
            "Weighted_Strike", "Weighted_MFE", "Weighted_MAE",
            "Reward/Risk Ratio", "Risk_Level"
        ]

        filtered = final_df[final_df["Weighted_Strike"] >= min_strike]

        if risk_filter != "All":
            filtered = filtered[filtered["Risk_Level"] == risk_filter]

        # Remove Totals rows with case-agnostic filter
        filtered = filtered[~filtered["Instrument"].astype(str).str.strip().str.lower().eq("totals")]

        # Apply color grading to Reward/Risk Ratio
        def color_rr(val):
            try:
                val = float(val)
                if val >= 1.80:
                    return "background-color: #4caf50;"  # Green
                elif val >= 1.30:
                    return "background-color: #fbc02d;"  # Yellow-Gold
                elif val >= 1.00:
                    return "background-color: #ff9800;"  # Orange
                else:
                    return "background-color: #f44336;"  # Red
            except:
                return ""

        styled_combined = filtered[display_cols].style.applymap(color_rr, subset=["Reward/Risk Ratio"])
        st.dataframe(styled_combined, use_container_width=True)

        st.markdown("### 📈 Individual Instrument Tables")
        for inst in filtered["Instrument"].unique():
            inst_df = filtered[filtered["Instrument"] == inst]
            st.markdown(f"#### {inst}")
            styled_inst = inst_df[display_cols].style.applymap(color_rr, subset=["Reward/Risk Ratio"])
            st.dataframe(styled_inst, use_container_width=True)

    else:
        st.info("📥 Please upload all 4 timeframes for at least one instrument to begin analysis.")

# -----end section 4 - Range Finder Display + Tooltip + Color Grading-----
# -----start section 5 - DCA CALCULATOR: UI, BLENDED ENTRY, AND SUMMARY TABLE-----

elif page == "DCA Risk Calculator":
    st.title("📐 DCA Risk Calculator")

    tick_values = {
        "NQ": 5.00, "YM": 5.00, "ES": 12.5, "GC": 10.00, "6E": 6.25,
        "CL": 10.00, "RTY": 5.00, "6B": 6.25, "6J": 12.5, "6A": 10.0,
        "6C": 10.0, "6N": 10.0, "6S": 12.5, "SI": 50.0, "HG": 25.0,
        "MGC": 1.00, "MES": 1.25, "MNQ": 0.50, "MYM": 0.50, "M2K": 0.50,
        "M6E": 1.25, "M6B": 1.25, "M6A": 1.00, "MCL": 1.00, "PA": 50.0, "PL": 50.0
    }

    col1, col2 = st.columns(2)
    with col1:
        symbol = st.selectbox("Instrument", list(tick_values.keys()))
        ref_price = st.number_input("Reference Price", min_value=0.0, format="%.2f")
        max_mae_pct = st.number_input("Max MAE % (Total Stop)", min_value=0.01, format="%.2f")
        initial_contracts = st.number_input("Initial Entry Contracts", min_value=1, step=1)
    with col2:
        mfe_1 = st.number_input("MFE Target 1 (%)", min_value=0.01, format="%.2f")
        mfe_2 = st.number_input("MFE Target 2 (%)", min_value=0.00, format="%.2f")

    tick_val = tick_values[symbol]
    dca_levels = []

    st.subheader("➕ DCA Levels")
    for i in range(1, 4):
        c1, c2 = st.columns(2)
        with c1:
            mae = st.number_input(f"MAE {i} (%)", min_value=0.00, format="%.2f", key=f"mae_{i}")
        with c2:
            qty = st.number_input(f"Contracts @ MAE {i}", min_value=0, step=1, key=f"qty_{i}")
        if mae > 0 and qty > 0:
            dca_levels.append((mae, qty))

    all_orders = [(0.0, initial_contracts)] + dca_levels
    total_qty = sum(qty for _, qty in all_orders)

    blended_entry = sum((ref_price * (1 - mae / 100)) * qty for mae, qty in all_orders) / total_qty
    stop_price = ref_price * (1 - max_mae_pct / 100)
    tick_dist_to_stop = (blended_entry - stop_price) / (tick_val / 100)
    dollar_risk = tick_dist_to_stop * tick_val * total_qty
    breakeven = blended_entry + (dollar_risk / (tick_val * total_qty)) * (tick_val / 100)

    summary_data = {
        "Blended Entry": [round(blended_entry, 4)],
        "Stop Price": [round(stop_price, 4)],
        "Breakeven Price": [round(breakeven, 4)],
        "Total Contracts": [total_qty],
        "Dollar Risk": [round(dollar_risk, 2)]
    }

    st.subheader("📊 DCA Summary")
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

# -----end section 5 - DCA CALCULATOR: UI, BLENDED ENTRY, AND SUMMARY TABLE-----
# -----start section 6 - DCA CALCULATOR: RISK/REWARD TABLE-----

    results = []
    for mfe in [mfe_1, mfe_2]:
        if mfe > 0:
            tp_price = ref_price * (1 + mfe / 100)
            tick_dist_to_tp = (tp_price - blended_entry) / (tick_val / 100)
            profit = tick_dist_to_tp * tick_val * total_qty
            rr = round(profit / dollar_risk, 2) if dollar_risk > 0 else "N/A"
            results.append({
                "Profit $": round(profit, 2),
                "Dollar Risk": round(dollar_risk, 2),
                "RR": rr,
                "MFE %": mfe,
                "Max MAE %": max_mae_pct
            })

    if results:
        st.subheader("📈 Risk/Reward Table")
        rr_df = pd.DataFrame(results)[["Profit $", "Dollar Risk", "RR", "MFE %", "Max MAE %"]]
        st.dataframe(rr_df, use_container_width=True)

# -----end section 6 - DCA CALCULATOR: RISK/REWARD TABLE-----

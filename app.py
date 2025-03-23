# -----start section 1-----
# RANGE FINDER + DCA DASHBOARD SETUP

import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Select Page", ["Range Finder", "DCA Risk Calculator"])
# -----end section 1-----
# -----start section 2-----
# RANGE FINDER: CSV UPLOADS + DATA PREP

if page == "Range Finder":
    st.title("📊 Range Finder")

    num_instruments = st.sidebar.number_input("Number of Instruments", 1, 20, 1)
    timeframes = ["1 Year", "6 Months", "3 Months", "1 Month"]
    instrument_data = {}

    for i in range(1, num_instruments + 1):
        with st.sidebar.expander(f"Instrument {i}"):
            name = st.text_input(f"Rename Instrument {i}", f"Instrument_{i}")
            instrument_data[name] = {}
            for tf in timeframes:
                uploaded = st.file_uploader(f"{name} - {tf}", type="csv", key=f"{name}_{tf}")
                if uploaded:
                    df = pd.read_csv(uploaded)
                    expected_cols = ["DayOfWeek", "RangeStart", "RangeEnd", "StrikeRate", "AvgMAE", "AvgMFE", "AvgRangePerc"]
                    df = df[[col for col in expected_cols if col in df.columns]]
                    instrument_data[name][tf] = df
# -----end section 2-----
# -----start section 3-----
# RANGE FINDER: METRICS, RISK SCORE, RECOMMENDED MFE

    st.subheader("📖 Refined Table View")
    min_strike = st.number_input("Min Strike Rate", 0, 100, 75)
    risk_filter = st.radio("Risk Filter", ["All", "Low", "Moderate", "High"], horizontal=True)
    combined = []

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
                df = df.merge(data[tf].rename(columns=cols), on=["DayOfWeek", "RangeStart", "RangeEnd"], how="left")

            weights = {"1M": 0.1, "3M": 0.2, "6M": 0.3, "1Y": 0.4}
            df["Weighted_Strike"] = sum(df[f"Strike_{k}"] * w for k, w in weights.items())
            df["Weighted_MAE"] = sum(df[f"MAE_{k}"] * w for k, w in weights.items())
            df["Weighted_MFE"] = sum(df[f"MFE_{k}"] * w for k, w in weights.items())
            df["Weighted_MFE_to_MAE"] = df["Weighted_MFE"] / df["Weighted_MAE"]

            df["Risk_Score"] = sum(
                (df[f"MAE_{k}"] / df[f"Range_{k}"]) * 100 * w for k, w in weights.items()
            )
            df["Risk_Level"] = df["Risk_Score"].apply(lambda x: "Low" if x < 80 else "Moderate" if x <= 120 else "High")

            for k in ["1M", "3M", "6M", "1Y"]:
                df[f"Ratio_{k}"] = df[f"MFE_{k}"] / df[f"MAE_{k}"]

            def pick_mfe(row):
                candidates = []
                for k in ["1M", "3M", "6M", "1Y"]:
                    if row[f"Ratio_{k}"] >= 1.5 and row[f"Strike_{k}"] >= 70:
                        candidates.append((k, row[f"Strike_{k}"], row[f"MFE_{k}"]))
                if not candidates:
                    return f"{round(row['Weighted_MFE'], 3)}% (Weighted)"
                best = sorted(candidates, key=lambda x: (-x[1], -x[2]))[0]
                return f"{round(best[2], 3)}% ({best[0]})"

            df["Recommended_MFE"] = df.apply(pick_mfe, axis=1)
# -----end section 3-----
# -----start section 4-----
# RANGE FINDER: TARGET ZONE, MAE QUALITY, TABLE FINALIZATION

            def target_zone(row):
                valid = [row[f"MFE_{k}"] for k in ["1M", "3M", "6M", "1Y"] if row[f"Ratio_{k}"] >= 1.5]
                if not valid:
                    return f"{round(row['Weighted_MFE'], 3)}%"
                return f"{round(min(valid), 3)}% – {round(max(valid), 3)}%"

            df["MFE_Target_Zone"] = df.apply(target_zone, axis=1)
            df["MAE_Threshold"] = df["Range_1M"] * 2.5

            def mae_grade(x):
                if x >= 1.8: return "Excellent"
                elif x >= 1.3: return "Tradable"
                elif x >= 1.0: return "Unreliable"
                else: return "Fail"

            df["MAE_Quality"] = df["Weighted_MFE_to_MAE"].apply(mae_grade)
            df.insert(0, "Instrument", instrument)
            combined.append(df)
# -----end section 4-----
# -----start section 5-----
# RANGE FINDER: FILTERED TABLE VIEW WITH TOOLTIP EXPANDER

    if combined:
        final_df = pd.concat(combined).reset_index(drop=True)

        st.markdown("### Refined Range Table")

        with st.expander("ℹ️ Column Guide"):
            st.markdown("""
            - **Weighted_Strike**: Weighted strike rate from 1Y (40%) to 1M (10%).
            - **Recommended MFE**: Best-performing timeframe with high MFE/MAE ratio and strike rate.
            - **Target Zone**: MFE range (min to max) from qualifying timeframes.
            - **Weighted MAE**: Avg downside (risk) across all timeframes.
            - **MAE Quality**:
                - Excellent ≥ 1.8
                - Tradable 1.3–1.79
                - Unreliable 1.0–1.29
                - Fail < 1.0
            - **Risk Level**: Based on MAE relative to range size.
            """)

        table_cols = [
            "Instrument", "DayOfWeek", "RangeStart", "RangeEnd",
            "Weighted_Strike", "Recommended_MFE", "MFE_Target_Zone",
            "Weighted_MAE", "MAE_Quality", "Risk_Level"
        ]

        df_filtered = final_df[final_df["Weighted_Strike"] >= min_strike]
        if risk_filter != "All":
            df_filtered = df_filtered[df_filtered["Risk_Level"] == risk_filter]

        # Remove "Totals" row if present
        df_filtered = df_filtered[df_filtered["Instrument"] != "Totals"]

        st.dataframe(df_filtered[table_cols].sort_values("Weighted_Strike", ascending=False), use_container_width=True)
    else:
        st.info("Upload valid CSVs for all 4 timeframes to begin analysis.")
# -----end section 5-----
# -----start section 6-----
# DCA CALCULATOR: UI, BLENDED ENTRY, AND SUMMARY TABLE

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
# -----end section 6-----
# -----start section 7-----
# DCA CALCULATOR: RISK/REWARD TABLE

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
# -----end section 7-----

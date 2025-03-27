# -----start section 1 - Setup & Page Navigation-----

import streamlit as st
import pandas as pd
import numpy as np

# Page config and sidebar selection
st.set_page_config(layout="wide")
page = st.sidebar.radio("Select Page", ["Range Finder"])

# -----end section 1 - Setup & Page Navigation-----

# -----start section 2 - Bulk Uploads + Filters Only-----

if page == "Range Finder":
    st.header("📊 Range Finder")

    # --- 2.1 Bulk Upload Block ---
    st.sidebar.markdown("### 📂 Upload")
    st.sidebar.markdown("Eg: CSVs : `XYZ_GC_6M.csv`")
    bulk_files = st.sidebar.file_uploader(
        "Upload CSVs", type="csv", accept_multiple_files=True, key="bulk_upload"
    )

    # --- 2.2 Filters ---
    st.subheader("🔍 Filters")
    col1, col2 = st.columns(2)

    with col1:
        min_strike = st.number_input("Min Strike Rate", min_value=0, max_value=100, value=75)

    with col2:
        max_stdev = st.number_input("Max Strike Rate Deviation (Stdev)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)

    setup_health_radio = st.radio(
        "Setup Health",
        ["All", "✅ Core Setup", "⚠️ Watch List", "❌ Decaying"],
        index=0,
        horizontal=True
    )

    if setup_health_radio == "All":
        setup_health_filter = ["✅ Core Setup", "⚠️ Watch List", "❌ Decaying"]
    else:
        setup_health_filter = [setup_health_radio]

    # --- 2.3 Metric Definitions ---
    with st.expander("📘 Column Definitions & Calculation Logic", expanded=False):
        st.markdown("""### **Metric Logic Overview**

- **Strike**  
  Recency-weighted average of strike rate across 1Y, 6M, 3M, and 1M. Emphasizes long-term consistency while giving weight to recent data.

- **MAE / MFE**  
  Recency-weighted average of Maximum Adverse Excursion (MAE) and Maximum Favorable Excursion (MFE), representing average drawdown and run-up per setup.

- **Range %**  
  Weighted average of price range (as % of price) for each setup window, favoring recent time periods.

- **MFE/MAE**  
  Reward-to-risk ratio. A higher value means the setup tends to deliver more reward relative to its risk.

- **Range/MAE (Base Hit Ratio)**  
  Measures how efficiently price reverts to the otherside of the range. Higher values suggest stronger mean-reverting behavior.

- **Avg Duration**  
  Recency-weighted average time the setup lasts, formatted as `1h 12m`, `43m`, etc.

- **Strike Decay**  
  Difference between 1M and 1Y strike rate (`1M - 1Y`). A positive value indicates improving recent performance.

- **Strike Stdev**  
  Standard deviation of the strike rates across all timeframes. Lower values suggest more consistent historical performance.

- **Setup Health**  
  Combines Strike Decay + Stdev to classify setups:
  - ✅ Core Setup: Strike is improving and stable (low stdev)
  - ⚠️ Watch List: Improving strike, but inconsistent
  - ❌ Decaying: Strike performance is declining
        """)

    instrument_data = {}
    combined = []
    upload_status = {}

# -----end section 2 - Bulk Uploads + Filters Only-----
# -----start section 3 - Range Finder Core Logic (Bulk Upload Processing)-----

    timeframe_labels = {"1Y": "1 Year", "6M": "6 Months", "3M": "3 Months", "1M": "1 Month"}
    expected_timeframes = set(timeframe_labels.values())
    missing_timeframes_by_instrument = {}

    if bulk_files:
        for file in bulk_files:
            filename = file.name.replace(".csv", "")
            parts = filename.split("_")
            if len(parts) >= 2:
                tf_code = parts[-1].upper()
                tf_label = timeframe_labels.get(tf_code)
                if tf_label:
                    instrument = "_".join(parts[1:-1]) if len(parts) > 2 else parts[0]
                    df = pd.read_csv(file)
                    if "Date" in df.columns:
                        df.rename(columns={"Date": "DayOfWeek"}, inplace=True)
                    if instrument not in instrument_data:
                        instrument_data[instrument] = {}
                    instrument_data[instrument][tf_label] = df

    # --- Display Grouped Uploads (Compact Sidebar) ---
    if instrument_data:
        st.sidebar.markdown("### 📁 Uploaded Files")
        for inst, timeframes in instrument_data.items():
            tf_list = sorted(timeframes.keys(), key=lambda x: ["1 Month", "3 Months", "6 Months", "1 Year"].index(x))
            compact_tf = ", ".join([tf.split()[0] for tf in tf_list])
            st.sidebar.markdown(f"**{inst}**  • {compact_tf}")

# -----end section 3 - Range Finder Core Logic (Bulk Upload Processing)-----
# -----start section 4 - Range Finder Display + Table Output-----

    if instrument_data:
        for instrument, data in instrument_data.items():
            if all(tf in data for tf in expected_timeframes):
                df = data["1 Month"].copy()
                df.rename(columns={
                    "StrikeRate": "Strike_1M", "AvgMAE": "MAE_1M", "AvgMFE": "MFE_1M",
                    "AvgRangePerc": "Range_1M", "AvgDuration": "Duration_1M"
                }, inplace=True)

                rename_map = {
                    "3 Months": {"StrikeRate": "Strike_3M", "AvgMAE": "MAE_3M", "AvgMFE": "MFE_3M", "AvgRangePerc": "Range_3M", "AvgDuration": "Duration_3M"},
                    "6 Months": {"StrikeRate": "Strike_6M", "AvgMAE": "MAE_6M", "AvgMFE": "MFE_6M", "AvgRangePerc": "Range_6M", "AvgDuration": "Duration_6M"},
                    "1 Year": {"StrikeRate": "Strike_1Y", "AvgMAE": "MAE_1Y", "AvgMFE": "MFE_1Y", "AvgRangePerc": "Range_1Y", "AvgDuration": "Duration_1Y"},
                }

                for tf, cols in rename_map.items():
                    df_temp = data[tf].rename(columns=cols)
                    required = ["DayOfWeek", "RangeStart", "RangeEnd"] + list(cols.values())
                    df_temp = df_temp.loc[:, required]
                    df = df.merge(df_temp, on=["DayOfWeek", "RangeStart", "RangeEnd"], how="left")

                w_strike = {"1M": 0.1, "3M": 0.2, "6M": 0.3, "1Y": 0.4}
                w_recent = {"1M": 0.4, "3M": 0.3, "6M": 0.2, "1Y": 0.1}

                df["Strike"] = sum(df[f"Strike_{k}"] * w for k, w in w_strike.items())
                df["MAE"] = sum(df[f"MAE_{k}"] * w for k, w in w_recent.items())
                df["MFE"] = sum(df[f"MFE_{k}"] * w for k, w in w_recent.items())
                df["Range %"] = sum(df[f"Range_{k}"] * w for k, w in w_recent.items())
                df["Avg Duration"] = sum(df[f"Duration_{k}"] * w for k, w in w_recent.items())

                df["MFE/MAE"] = df["MFE"] / df["MAE"]
                df["Range/MAE"] = df["Range %"] / df["MAE"]

                df["Strike_Stdev"] = df[[f"Strike_{k}" for k in w_strike]].std(axis=1)
                df["Strike_Decay"] = df["Strike_1M"] - df["Strike_1Y"]

                def classify_setup(decay, stdev):
                    if decay > 0 and stdev <= max_stdev:
                        return "✅ Core Setup"
                    elif decay > 0:
                        return "⚠️ Watch List"
                    else:
                        return "❌ Decaying"

                df["Setup Health"] = df.apply(lambda row: classify_setup(row["Strike_Decay"], row["Strike_Stdev"]), axis=1)
                df["Instrument"] = instrument
                combined.append(df)

    if combined:
        final_df = pd.concat(combined).reset_index(drop=True)

        # Filter and clean
        filtered = final_df[final_df["Strike"] >= min_strike]
        filtered = filtered[filtered["Setup Health"].isin(setup_health_filter)]
        filtered = filtered[filtered["Strike_Stdev"] <= max_stdev]
        filtered = filtered[
            ~filtered["Instrument"].astype(str).str.strip().str.lower().eq("totals") &
            ~filtered["DayOfWeek"].astype(str).str.strip().str.lower().eq("totals")
        ]

        def format_duration(minutes):
            h = int(minutes) // 60
            m = int(minutes) % 60
            return f"{h}h {m}m" if h else f"{m}m"

        # Format and center display
        display_cols = [
            "Instrument", "DayOfWeek", "RangeStart", "RangeEnd",
            "Strike", "MFE", "MAE", "Range %", "Range/MAE", "MFE/MAE",
            "Avg Duration", "Setup Health"
        ]
        numeric_cols = ["Strike", "MFE", "MAE", "Range %", "Range/MAE", "MFE/MAE"]

        rounded = filtered[display_cols].copy()
        for col in numeric_cols:
            rounded[col] = rounded[col].apply(lambda x: f"{x:.2f}")
        rounded["Avg Duration"] = filtered["Avg Duration"].apply(format_duration)

        # Combined Table
        st.markdown("### 📊 Combined Playbook ")
        st.dataframe(rounded.style.set_properties(**{"text-align": "center"}), use_container_width=True)

        # Per Instrument
        st.markdown("### 📈 Individual Playbook")
        for inst in filtered["Instrument"].unique():
            inst_df = filtered[filtered["Instrument"] == inst]
            inst_rounded = inst_df[display_cols].copy()
            for col in numeric_cols:
                inst_rounded[col] = inst_rounded[col].apply(lambda x: f"{x:.2f}")
            inst_rounded["Avg Duration"] = inst_df["Avg Duration"].apply(format_duration)
            st.markdown(f"#### {inst}")
            st.dataframe(inst_rounded.style.set_properties(**{"text-align": "center"}), use_container_width=True)

    else:

        st.info("📥 Please upload all 4 time periods for at least one instrument to begin analysis.")

# -----end section 4 - Range Finder Display + Table Output-----

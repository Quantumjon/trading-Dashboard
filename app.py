# ----start section 1 - Setup & Page Navigation-----

import streamlit as st
import pandas as pd
import numpy as np

# Page config and sidebar selection
st.set_page_config(layout="wide")
page = st.sidebar.radio("Select Page", ["Range Finder", "Backtest Analyzer"])


# -----end section 1 - Setup & Page Navigation-----

# -----start section 2 - Bulk Uploads + Filters Only-----

if page == "Range Finder":
    st.header("📊 Range Finder")

    # --- 2.1 Bulk Upload Block ---
    st.sidebar.markdown("### 📂 Upload")
    st.sidebar.markdown("CSVs : `XYZ_GC_6M.csv`")
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

#-------- end dashboard 1 -----------------------------------------------------
#-------- start dashboard 2 ---------------------------------------------------

# -----start section X - MAE/MFE Analyzer Integration -----
if page == "Backtest Analyzer":
    import plotly.express as px

    st.header("📊 Backtest Analyzer")

    uploaded_file = st.sidebar.file_uploader("📁 1Year Backtest CSV", type=["csv"], key="upload")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df["DayOfWeek"] = df["DayOfWeek"].str.strip().str.title()
        valid_days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]
        df = df[df["DayOfWeek"].isin(valid_days)]

        st.sidebar.subheader("📅 Filter by Recency")
        num_trades = st.sidebar.number_input("Trades per weekday (MAE/MFE charts)", min_value=5, max_value=200, value=30, step=5)

        def filter_last_n_per_day(df, n):
            return df.groupby("DayOfWeek", group_keys=False).apply(lambda g: g.sort_values("Datetime").tail(n)).reset_index(drop=True)

        strategy = st.radio("Choose Strategy:", ["MAE Optimizer (Base-Hit TP)", "MFE Strategy (Fixed 0.3% MAE)"])

        # === MAE STRATEGY ===
        if strategy == "MAE Optimizer (Base-Hit TP)":
            st.subheader("🟩 MAE Optimizer: Base-Hit Strategy")
            df["Direction"] = np.where(df["BreakoutDirection"] == "High", "Short", "Long")
            df["RewardPct"] = np.where(
                df["Direction"] == "Short",
                (df["BreakoutClose"] - df["Range_Low"]) / df["BreakoutClose"],
                (df["Range_High"] - df["BreakoutClose"]) / df["BreakoutClose"],
            )

            mae_steps = np.round(np.arange(0.01, 0.31, 0.01), 3)
            min_sr_threshold = 60
            results = []

            for day, group in df.groupby("DayOfWeek"):
                best_row = None
                best_expectancy = -np.inf
                for mae in mae_steps:
                    eligible = group[group["MAE"] <= mae]
                    if eligible.empty:
                        continue
                    wins = eligible["Hit"].sum()
                    total = len(group)
                    sr = (wins / total) * 100 if total else 0
                    if sr < min_sr_threshold:
                        continue
                    rr = eligible["RewardPct"] / (mae / 100)
                    median_rr = np.median(rr)
                    exp = (sr / 100) * median_rr - (1 - sr / 100)

                    recent = group.sort_values("Datetime").iloc[-30:]
                    historic = group.sort_values("Datetime").iloc[:-30]
                    mae_shift = (recent["MAE"].mean() - historic["MAE"].mean()) / historic["MAE"].mean() * 100 if not historic.empty else 0
                    range_shift = (recent["RangePercentage"].mean() - historic["RangePercentage"].mean()) / historic["RangePercentage"].mean() * 100 if not historic.empty else 0
                    dur_shift = (recent["Duration"].mean() - historic["Duration"].mean()) / historic["Duration"].mean() * 100 if not historic.empty else 0

                    if exp > best_expectancy:
                        best_row = {
                            "DayOfWeek": day,
                            "MAE_Stop": mae,
                            "Strike Rate (%)": round(sr, 2),
                            "R:R": round(median_rr, 2),
                            "Expectancy": round(exp, 4),
                            "MAE Change (%)": round(mae_shift, 1),
                            "Range% Change (%)": round(range_shift, 1),
                            "Duration Change (%)": round(dur_shift, 1),
                        }
                        best_expectancy = exp
                if best_row:
                    results.append(best_row)

            mae_df = pd.DataFrame(results)

            def score_confidence(row):
                score = 0
                if row["Strike Rate (%)"] >= 75:
                    score += 1
                if row["Expectancy"] >= 0.15:
                    score += 1
                return score

            mae_df["ConfidenceScore"] = mae_df.apply(score_confidence, axis=1)
            mae_df["Confidence"] = mae_df["ConfidenceScore"].apply(lambda s: "High" if s == 2 else "Medium" if s == 1 else "Low")

            st.subheader("🧠 Trader’s Assistance (MAE Strategy)")
            for _, row in mae_df.iterrows():
                label = {
                    "High": "(core setup, confidence: High)",
                    "Medium": "(watchlist, confidence: Medium)",
                    "Low": "(risky setup, confidence: Low)"
                }[row["Confidence"]]
                st.markdown(
                    f"- **{row['DayOfWeek']} {label}** → Optimal stop at `{row['MAE_Stop']:.2f}%` "
                    f"→ SR `{row['Strike Rate (%)']:.1f}%`, R:R `{row['R:R']:.2f}`, EV `{row['Expectancy']:.2f}` | "
                    f"Volatility shift: MAE `{row['MAE Change (%)']:+.1f}%`, Range `{row['Range% Change (%)']:+.1f}%`, Duration `{row['Duration Change (%)']:+.1f}%`."
                )

            st.subheader("📊 Optimized MAE Summary")
            st.dataframe(mae_df, use_container_width=True)

            # -------- MAE SCATTER (Interactive + Reference Lines) --------
            st.subheader("📉 MAE Distribution (Last Trades per Day)")
            df_recent = filter_last_n_per_day(df, num_trades)
            fig = px.scatter(df_recent, x="Datetime", y="MAE", color="DayOfWeek", title="MAE Scatter Plot by Day", height=600)
            fig.update_traces(marker=dict(size=6, opacity=0.7))
            fig.update_layout(yaxis_title="MAE (%)", xaxis_title="Date", legend_title="Day")
            for level in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
                fig.add_hline(y=level, line_dash="dash", line_color="gray", opacity=0.5)
            st.plotly_chart(fig, use_container_width=True)
            # -------- END MAE SCATTER --------

        # === MFE STRATEGY ===
        elif strategy == "MFE Strategy (Fixed 0.3% MAE)":
            st.subheader("🟦 MFE Strategy: Target Optimization (0.3% MAE Stop)")
            df_fixed = df[df["MAE"] <= 0.3]
            mfe_steps = np.round(np.arange(0.01, 0.31, 0.01), 3)
            results = []
            for day, group in df_fixed.groupby("DayOfWeek"):
                valid = group[group["Hit"] == 1]
                best_ev = -999
                best_row = None
                for mfe in mfe_steps:
                    hit_count = (valid["MFE"] >= mfe).sum()
                    total = len(valid)
                    hit_rate = hit_count / total if total else 0
                    rr = mfe / 0.3
                    ev = hit_rate * rr - (1 - hit_rate)
                    if ev > best_ev:
                        best_row = {
                            "DayOfWeek": day,
                            "MFE_Target": mfe,
                            "Hit Rate (%)": round(hit_rate * 100, 2),
                            "R:R": round(rr, 2),
                            "Expectancy": round(ev, 4),
                            "Trades": total
                        }
                        best_ev = ev
                if best_row:
                    results.append(best_row)

            mfe_df = pd.DataFrame(results)

            st.subheader("🧠 Trader’s Assistance (MFE Strategy)")
            for _, row in mfe_df.iterrows():
                st.markdown(
                    f"- **{row['DayOfWeek']}** → Optimal MFE Target: `{row['MFE_Target']:.2f}` "
                    f"→ SR `{row['Hit Rate (%)']:.1f}%`, R:R `{row['R:R']:.2f}`, EV `{row['Expectancy']:.2f}` "
                    f"from `{row['Trades']}` qualifying trades."
                )

            st.subheader("📊 MFE Target Summary")
            st.dataframe(mfe_df, use_container_width=True)

            # -------- MFE SCATTER (Interactive + Reference Lines) --------
            st.subheader("📈 MFE Distribution (Last Trades per Day)")
            df_mfe_recent = filter_last_n_per_day(df_fixed, num_trades)
            fig2 = px.scatter(df_mfe_recent, x="Datetime", y="MFE", color="DayOfWeek", title="MFE Scatter Plot by Day", height=600)
            fig2.update_traces(marker=dict(size=6, opacity=0.7))
            fig2.update_layout(yaxis_title="MFE (%)", xaxis_title="Date", legend_title="Day")
            for level in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
                fig2.add_hline(y=level, line_dash="dash", line_color="gray", opacity=0.5)
            st.plotly_chart(fig2, use_container_width=True)
            # -------- END MFE SCATTER --------
# -----end section X - MAE/MFE Analyzer Integration -----

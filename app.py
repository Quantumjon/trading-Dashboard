# ----start section 1 - Setup & Page Navigation-----
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")
page = st.sidebar.radio("Select Page", ["Range Finder", "Backtest Analyzer"])
# ----end section 1 - Setup & Page Navigation-----

# ----start section 2 - Range Finder -----
if page == "Range Finder":
    st.header("📊 Range Finder")
    st.sidebar.markdown("### 📂 Upload")
    st.sidebar.markdown("CSVs : `XYZ_GC_6M.csv`")
    bulk_files = st.sidebar.file_uploader("Upload CSVs", type="csv", accept_multiple_files=True, key="bulk_upload")

    st.subheader("🔍 Filters")
    col1, col2 = st.columns(2)
    with col1:
        min_strike = st.number_input("Min Strike Rate", min_value=0, max_value=100, value=75)
    with col2:
        max_stdev = st.number_input("Max Strike Rate Deviation (Stdev)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)

    setup_health_radio = st.radio(
        "Setup Health", ["All", "✅ Core Setup", "⚠️ Watch List", "❌ Decaying"],
        index=0, horizontal=True
    )
    setup_health_filter = (
        ["✅ Core Setup", "⚠️ Watch List", "❌ Decaying"]
        if setup_health_radio == "All" else [setup_health_radio]
    )

    with st.expander("📘 Column Definitions & Calculation Logic", expanded=False):
        st.markdown("""### **Metric Logic Overview**
- **Strike**: Recency-weighted average of strike rate across timeframes.
- **MAE / MFE**: Avg drawdown / run-up per setup.
- **Range %**: Avg price range in percent.
- **MFE/MAE**: Reward-to-risk ratio.
- **Range/MAE**: Measures mean-reversion power.
- **Avg Duration**: Avg time for setup completion.
- **Strike Decay**: Recent improvement (`1M - 1Y`).
- **Strike Stdev**: Variability across timeframes.
- **Setup Health**: Combines decay & stdev for quality label.""")

    instrument_data = {}
    combined = []
    timeframe_labels = {"1Y": "1 Year", "6M": "6 Months", "3M": "3 Months", "1M": "1 Month"}
    expected_timeframes = set(timeframe_labels.values())

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

    if instrument_data:
        st.sidebar.markdown("### 📁 Uploaded Files")
        for inst, timeframes in instrument_data.items():
            tf_list = sorted(timeframes.keys(), key=lambda x: ["1 Month", "3 Months", "6 Months", "1 Year"].index(x))
            compact_tf = ", ".join([tf.split()[0] for tf in tf_list])
            st.sidebar.markdown(f"**{inst}**  • {compact_tf}")

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

                def classify(decay, stdev):
                    if decay > 0 and stdev <= max_stdev: return "✅ Core Setup"
                    elif decay > 0: return "⚠️ Watch List"
                    else: return "❌ Decaying"

                df["Setup Health"] = df.apply(lambda row: classify(row["Strike_Decay"], row["Strike_Stdev"]), axis=1)
                df["Instrument"] = instrument
                combined.append(df)

    if combined:
        final_df = pd.concat(combined).reset_index(drop=True)
        filtered = final_df[
            (final_df["Strike"] >= min_strike) &
            (final_df["Strike_Stdev"] <= max_stdev) &
            (final_df["Setup Health"].isin(setup_health_filter))
        ]
        filtered = filtered[
            ~filtered["Instrument"].str.lower().eq("totals") &
            ~filtered["DayOfWeek"].str.lower().eq("totals")
        ]

        def format_minutes(mins):
            h = int(mins) // 60
            m = int(mins) % 60
            return f"{h}h {m}m" if h else f"{m}m"

        display_cols = ["Instrument", "DayOfWeek", "RangeStart", "RangeEnd",
                        "Strike", "MFE", "MAE", "Range %", "Range/MAE", "MFE/MAE", "Avg Duration", "Setup Health"]
        numeric_cols = ["Strike", "MFE", "MAE", "Range %", "Range/MAE", "MFE/MAE"]

        view = filtered[display_cols].copy()
        for col in numeric_cols:
            view[col] = view[col].apply(lambda x: f"{x:.2f}")
        view["Avg Duration"] = filtered["Avg Duration"].apply(format_minutes)

        st.markdown("### 📊 Combined Playbook")
        st.dataframe(view.style.set_properties(**{"text-align": "center"}), use_container_width=True)

        st.markdown("### 📈 Individual Playbook")
        for inst in filtered["Instrument"].unique():
            inst_df = filtered[filtered["Instrument"] == inst][display_cols].copy()
            for col in numeric_cols:
                inst_df[col] = inst_df[col].apply(lambda x: f"{x:.2f}")
            inst_df["Avg Duration"] = inst_df["Avg Duration"].apply(format_minutes)
            st.markdown(f"#### {inst}")
            st.dataframe(inst_df.style.set_properties(**{"text-align": "center"}), use_container_width=True)
    else:
        st.info("📥 Please upload all 4 timeframes per instrument.")
# ----end section 2 - Range Finder -----
# -----start section 3 - Backtest Analyzer (MAE/MFE Strategy) -----
if page == "Backtest Analyzer":
    st.header("📊 Backtest Analyzer")
    uploaded_file = st.sidebar.file_uploader("📁 Upload 1Y Backtest CSV", type=["csv"], key="upload_bt")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df["DayOfWeek"] = df["DayOfWeek"].str.strip().str.title()
        df = df[df["DayOfWeek"].isin(["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"])]

        st.sidebar.subheader("📅 Filter by Recency")
        num_trades = st.sidebar.number_input("Trades per weekday (MAE/MFE charts)", 5, 200, 30, 5)

        def filter_last_n(df, n):
            return df.groupby("DayOfWeek", group_keys=False).apply(lambda g: g.sort_values("Datetime").tail(n)).reset_index(drop=True)

        strategy = st.radio("Choose Strategy:", ["MAE Optimizer (Base-Hit TP)", "MFE Strategy (Fixed 0.3% MAE)"])

        with st.expander("📘 How are these values calculated?"):
            if strategy == "MAE Optimizer (Base-Hit TP)":
                st.markdown("""
                **MAE Strategy Logic:**
                - Calculates strike rate based on TP hit before MAE breach.
                - Tests all MAE thresholds (0.01 to 0.3) and picks the level with highest expectancy.
                - Expectancy = `SR * R:R - (1 - SR)`
                - R:R = distance to opposite range side ÷ MAE stop level.
                - Also evaluates recent vs historical MAE, Range%, Duration for volatility shifts.
                """)
            else:
                st.markdown("""
                **MFE Strategy Logic (0.3% MAE Stop):**
                - Filters trades that did not exceed 0.3 MAE.
                - Calculates hit rate for different MFE targets.
                - Expectancy = `Hit Rate * (MFE / 0.3) - (1 - Hit Rate)`
                - Selects MFE level with highest expectancy.
                """)

        # === MAE Strategy ===
        if strategy == "MAE Optimizer (Base-Hit TP)":
            st.subheader("🟩 MAE Optimizer: Base-Hit Strategy")

            df["Direction"] = np.where(df["BreakoutDirection"] == "High", "Short", "Long")
            df["RewardPct"] = np.where(
                df["Direction"] == "Short",
                (df["BreakoutClose"] - df["Range_Low"]) / df["BreakoutClose"],
                (df["Range_High"] - df["BreakoutClose"]) / df["BreakoutClose"],
            )

            results = []
            mae_levels = np.round(np.arange(0.01, 0.31, 0.01), 3)

            for day, group in df.groupby("DayOfWeek"):
                best_row = None
                best_exp = -999
                for mae in mae_levels:
                    eligible = group[group["MAE"] <= mae]
                    if eligible.empty:
                        continue
                    wins = eligible["Hit"].sum()
                    sr = wins / len(group) * 100
                    if sr < 60: continue
                    rr = eligible["RewardPct"] / (mae / 100)
                    median_rr = np.median(rr)
                    ev = (sr / 100) * median_rr - (1 - sr / 100)
                    recent = group.sort_values("Datetime").iloc[-30:]
                    historic = group.sort_values("Datetime").iloc[:-30]
                    mae_shift = (recent["MAE"].mean() - historic["MAE"].mean()) / historic["MAE"].mean() * 100 if not historic.empty else 0
                    range_shift = (recent["RangePercentage"].mean() - historic["RangePercentage"].mean()) / historic["RangePercentage"].mean() * 100 if not historic.empty else 0
                    dur_shift = (recent["Duration"].mean() - historic["Duration"].mean()) / historic["Duration"].mean() * 100 if not historic.empty else 0

                    if ev > best_exp:
                        best_row = {
                            "DayOfWeek": day, "MAE_Stop": mae, "Strike Rate (%)": round(sr, 2),
                            "R:R": round(median_rr, 2), "Expectancy": round(ev, 4),
                            "MAE Change (%)": round(mae_shift, 1), "Range% Change (%)": round(range_shift, 1),
                            "Duration Change (%)": round(dur_shift, 1)
                        }
                        best_exp = ev
                if best_row: results.append(best_row)

            mae_df = pd.DataFrame(results)

            def confidence(row):
                score = int(row["Strike Rate (%)"] >= 75) + int(row["Expectancy"] >= 0.15)
                return ["Low", "Medium", "High"][score]

            mae_df["Confidence"] = mae_df.apply(confidence, axis=1)

            st.subheader("🧠 Trader’s Assistance (MAE Strategy)")
            for _, row in mae_df.iterrows():
                label = {
                    "High": "(core setup, confidence: High)",
                    "Medium": "(watchlist, confidence: Medium)",
                    "Low": "(risky setup, confidence: Low)"
                }[row["Confidence"]]
                st.markdown(
                    f"- **{row['DayOfWeek']} {label}** → Optimal stop at `{row['MAE_Stop']:.2f}%`, "
                    f"SR `{row['Strike Rate (%)']:.1f}%`, R:R `{row['R:R']:.2f}`, EV `{row['Expectancy']:.2f}` | "
                    f"Volatility shift: MAE `{row['MAE Change (%)']:+.1f}%`, Range `{row['Range% Change (%)']:+.1f}%`, Duration `{row['Duration Change (%)']:+.1f}%`."
                )

            st.subheader("📊 Optimized MAE Summary")
            st.dataframe(mae_df, use_container_width=True)

            st.subheader("📉 MAE Distribution")
            df_recent = filter_last_n(df, num_trades)
            fig = px.scatter(df_recent, x="Datetime", y="MAE", color="DayOfWeek", title="MAE Scatter Plot", height=600)
            for level in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]:
                fig.add_hline(y=level, line_dash="dash", line_color="gray", opacity=0.4)
            st.plotly_chart(fig, use_container_width=True)

        # === MFE Strategy ===
        elif strategy == "MFE Strategy (Fixed 0.3% MAE)":
            st.subheader("🟦 MFE Strategy: Target Optimization (0.3% MAE Stop)")
            df_fixed = df[df["MAE"] <= 0.3]
            mfe_levels = np.round(np.arange(0.01, 0.31, 0.01), 3)
            results = []

            for day, group in df_fixed.groupby("DayOfWeek"):
                best_row = None
                best_ev = -999
                valid = group[group["Hit"] == 1]
                for mfe in mfe_levels:
                    hit_rate = (valid["MFE"] >= mfe).sum() / len(valid) if len(valid) else 0
                    rr = mfe / 0.3
                    ev = hit_rate * rr - (1 - hit_rate)
                    if ev > best_ev:
                        best_row = {
                            "DayOfWeek": day, "MFE_Target": mfe,
                            "Hit Rate (%)": round(hit_rate * 100, 2),
                            "R:R": round(rr, 2), "Expectancy": round(ev, 4),
                            "Trades": len(valid)
                        }
                        best_ev = ev
                if best_row: results.append(best_row)

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

            st.subheader("📈 MFE Distribution")
            df_mfe_recent = filter_last_n(df_fixed, num_trades)
            fig2 = px.scatter(df_mfe_recent, x="Datetime", y="MFE", color="DayOfWeek", title="MFE Scatter Plot", height=600)
            for level in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]:
                fig2.add_hline(y=level, line_dash="dash", line_color="gray", opacity=0.4)
            st.plotly_chart(fig2, use_container_width=True)
# -----end section 3 - Backtest Analyzer (MAE/MFE Strategy) -----

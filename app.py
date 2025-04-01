import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# Page layout
st.set_page_config(layout="wide")

# Sidebar dashboard selector
st.sidebar.title("📊 Select Dashboard")
dashboard_page = st.sidebar.radio("Go to:", ["Range Finder", "Backtest Analyzer"])

# ----- START SECTION: RANGE FINDER -----
if dashboard_page == "Range Finder":
    st.header("📊 Range Finder")

    # --- 2.1 Bulk Upload Block ---
    st.sidebar.markdown("### 📂 Upload")
    st.sidebar.markdown("CSVs : `XYZ GC_6M.csv`")
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

    # Sidebar Summary of Uploaded Files
    if instrument_data:
        st.sidebar.markdown("### 📁 Uploaded Files")
        for inst, timeframes in instrument_data.items():
            tf_list = sorted(timeframes.keys(), key=lambda x: ["1 Month", "3 Months", "6 Months", "1 Year"].index(x))
            compact_tf = ", ".join([tf.split()[0] for tf in tf_list])
            st.sidebar.markdown(f"**{inst}**  • {compact_tf}")

    # ----- Core Processing + Summary Display -----
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
# ----- END SECTION: RANGE FINDER -----

# ----- BACKTEST ANALYZER SECTION -----
elif dashboard_page == "Backtest Analyzer":
    st.header("📈 Backtest Analyzer")

    uploaded_files = st.sidebar.file_uploader(
        "Upload Backtest CSVs", type="csv", accept_multiple_files=True, key="backtest_upload"
    )

    if uploaded_files:
        all_data = []
        for file in uploaded_files:
            df = pd.read_csv(file)
            instrument = "_".join(file.name.replace(".csv", "").split("_")[:-1])
            df["Instrument"] = instrument
            all_data.append(df)

        data = pd.concat(all_data, ignore_index=True)
        data["Datetime"] = pd.to_datetime(data["Datetime"])
        data["DayOfWeek"] = pd.Categorical(
            data["DayOfWeek"],
            categories=["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            ordered=True
        )

        strategy = st.radio("Select Strategy", ["MAE Optimizer", "MFE Strategy"])
        st.subheader("Trade Recency Filter")
        trade_limit = st.number_input("Number of recent trades per weekday", min_value=10, value=30)

        recent = data.sort_values("Datetime").groupby("DayOfWeek").tail(trade_limit)

        def calculate_efficiency(reward, risk):
            return reward / risk if risk > 0 else 0

        def assign_volatility_tier(mae, rng, dur):
            vol = mae + rng + dur / 100
            if vol < 0.5:
                return "Low"
            elif vol < 1:
                return "Medium"
            else:
                return "High"

        def get_streaks(result_series):
            results = result_series.tolist()
            streaks = {
                "Current Win Streak": 0,
                "Current Loss Streak": 0,
                "Max Win Streak": 0,
                "Max Loss Streak": 0,
                "Avg Win Streak": 0,
                "Avg Loss Streak": 0,
            }
            current_streak = 0
            last_result = None
            win_streaks = []
            loss_streaks = []

            for r in results:
                if r == last_result:
                    current_streak += 1
                else:
                    if last_result == 1:
                        win_streaks.append(current_streak)
                    elif last_result == 0:
                        loss_streaks.append(current_streak)
                    current_streak = 1
                last_result = r
            if last_result == 1:
                win_streaks.append(current_streak)
            elif last_result == 0:
                loss_streaks.append(current_streak)

            streaks["Current Win Streak"] = current_streak if last_result == 1 else 0
            streaks["Current Loss Streak"] = current_streak if last_result == 0 else 0
            streaks["Max Win Streak"] = max(win_streaks) if win_streaks else 0
            streaks["Max Loss Streak"] = max(loss_streaks) if loss_streaks else 0
            streaks["Avg Win Streak"] = round(np.mean(win_streaks), 2) if win_streaks else 0
            streaks["Avg Loss Streak"] = round(np.mean(loss_streaks), 2) if loss_streaks else 0
            return streaks

        def default_streaks():
            return {
                "Current Win Streak": 0,
                "Current Loss Streak": 0,
                "Max Win Streak": 0,
                "Max Loss Streak": 0,
                "Avg Win Streak": 0,
                "Avg Loss Streak": 0,
            }





# ----- START SECTION: MAE STRATEGY -----
        if strategy == "MAE Optimizer":
            st.markdown("## 🛡️ MAE Strategy – Base-Hit Optimizer")

            summary_rows = []
            ordered_days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

            for day in [d for d in ordered_days if d in recent["DayOfWeek"].unique()]:
                df_day = recent[recent["DayOfWeek"] == day]
                if df_day.empty:
                    continue

                total = len(df_day)
                best_row = None

                for stop in np.round(np.arange(0.05, 0.31, 0.01), 3):
                    df_day["ResultSim"] = np.where((df_day["MAE"] <= stop) & (df_day["Hit"] == 1), 1, 0)
                    sr = df_day["ResultSim"].mean()
                    rr = df_day["RangePercentage"].mean() / stop if stop > 0 else 0
                    ev = sr * rr - (1 - sr)

                    if sr >= 0.68:
                        candidate = {
                            "Day": day,
                            "Stop %": round(stop, 3),
                            "StrikeRate": f"{sr:.0%}",
                            "RR": round(rr, 2),
                            "EV": round(ev, 2),
                        }

                        if not best_row:
                            best_row = candidate
                        elif best_row["EV"] <= 0 and ev > 0:
                            best_row = candidate
                        elif best_row["EV"] > 0 and ev > 0 and sr > float(best_row["StrikeRate"].strip('%')) / 100:
                            best_row = candidate

                if best_row:
                    df_day["Filtered"] = (df_day["MAE"] <= best_row["Stop %"]) & (df_day["Hit"] == 1)
                    streaks = get_streaks(df_day["Filtered"].astype(int))
                    mae = df_day["MAE"].mean()
                    rng = df_day["RangePercentage"].mean()
                    dur = df_day["Duration"].mean()
                    eff = calculate_efficiency(rng, mae)
                    tier = assign_volatility_tier(mae, rng, dur)
                    label = "Core Setup" if float(best_row["StrikeRate"].strip('%')) >= 68 and best_row["EV"] > 0 else "Watchlist"
                else:
                    best_row = {
                        "Day": day,
                        "Stop %": np.nan,
                        "StrikeRate": "0%",
                        "RR": 0,
                        "EV": 0,
                    }
                    streaks = default_streaks()
                    tier = "High"
                    eff = 0
                    label = "Risky"

                best_row.update({
                    "Confidence": label,
                    "Volatility": tier,
                    "Efficiency": round(eff, 2),
                    **streaks,
                    "Total Trades": total
                })
                summary_rows.append(best_row)

            df_summary = pd.DataFrame(summary_rows).sort_values("Day", key=lambda x: pd.Categorical(x, categories=ordered_days, ordered=True))


            # Summary table
            st.markdown("### 📋 MAE Strategy Summary")
            st.dataframe(df_summary.set_index("Day"), use_container_width=True)

            # MAE Plotly Scatter
            st.markdown("### 📈 MAE Distribution")
            mae_plot_data = recent[recent["DayOfWeek"].isin(df_summary["Day"])]

            fig = px.scatter(
                mae_plot_data,
                x="Datetime",
                y="MAE",
                color="DayOfWeek",
                category_orders={"DayOfWeek": ordered_days},
                title="MAE Distribution by Date",
                labels={"MAE": "MAE (%)"},
                height=500
            )
            fig.update_traces(marker=dict(size=8, opacity=1.0))
            fig.update_layout(legend_title_text="Weekday", legend=dict(itemsizing="constant"))
            st.plotly_chart(fig, use_container_width=True)


# ----- START SECTION: MFE STRATEGY -----
        elif strategy == "MFE Strategy":
            st.markdown("## 🎯 MFE Strategy – Target Zone Analyzer")

            summary_rows = []
            ordered_days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

            for day in [d for d in ordered_days if d in recent["DayOfWeek"].unique()]:
                df_day = recent[recent["DayOfWeek"] == day]
                if df_day.empty:
                    continue

                total = len(df_day)

                # Stopped trades: those that exceeded 0.3% MAE
                df_day["Stopped"] = df_day["MAE"] > 0.3

                # MFE Target: towards opposite side of range
                df_day["TP_Hit"] = df_day["MFE"] >= df_day["RangePercentage"]
                
                # Qualified trades are those that didn't stop out and hit target
                df_day["Qualified"] = (~df_day["Stopped"]) & (df_day["TP_Hit"])
                df_day["ResultSim"] = df_day["Qualified"].astype(int)

                sr = df_day["ResultSim"].mean()
                ev = sr - (1 - sr)
                mae = df_day["MAE"].mean()
                rng = df_day["RangePercentage"].mean()
                dur = df_day["Duration"].mean()
                eff = calculate_efficiency(rng, 0.3)

                # Calculate edge distance if range boundaries are provided
                if "Range_High" in df_day.columns and "Range_Low" in df_day.columns:
                    edge = np.abs(df_day["BreakoutClose"] - np.where(
                        df_day["BreakoutDirection"] == "Long",
                        df_day["Range_High"],
                        df_day["Range_Low"]
                    )).mean()
                    edge_distance = round(df_day["RangePercentage"].mean() - edge, 3)
                else:
                    edge_distance = np.nan

                streaks = get_streaks(df_day["ResultSim"])
                tier = assign_volatility_tier(mae, rng, dur)
                label = "Core Setup" if sr >= 0.68 else "Watchlist" if sr >= 0.7 else "Risky"

                summary_rows.append({
                    "Day": day,
                    "Target %": round(rng, 3),
                    "StrikeRate": f"{sr:.0%}",
                    "RR": 1,
                    "EV": round(ev, 2),
                    "Efficiency": round(eff, 2),
                    "Volatility": tier,
                    "TP Speed": round(dur, 2),
                    "Entry Edge": edge_distance,
                    "Confidence": label,
                    **streaks,
                    "Total Trades": total
                })

            # Sort summary rows based on ordered weekday sequence
            df_summary = pd.DataFrame(summary_rows).sort_values("Day", key=lambda x: pd.Categorical(x, categories=ordered_days, ordered=True))

            # Summary table
            st.markdown("### 📋 MFE Strategy Summary")
            st.dataframe(df_summary.set_index("Day"), use_container_width=True)

            # MFE Plotly Scatter
            st.markdown("### 📈 MFE Distribution")
            mfe_plot_data = recent[recent["DayOfWeek"].isin(df_summary["Day"])]

            fig = px.scatter(
                mfe_plot_data,
                x="Datetime",
                y="MFE",
                color="DayOfWeek",
                category_orders={"DayOfWeek": ordered_days},
                title="MFE Distribution by Date",
                labels={"MFE": "MFE (%)"},
                height=500
            )
            fig.update_traces(marker=dict(size=8, opacity=1.0))
            fig.update_layout(legend_title_text="Weekday", legend=dict(itemsizing="constant"))
            st.plotly_chart(fig, use_container_width=True)

# ----- END SECTION: MFE STRATEGY -----

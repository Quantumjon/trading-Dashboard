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
This module tests a range of stop-loss levels (0.01% to 0.30%) to identify the most effective stop for each weekday.

- A **Win** is counted only if TP is hit **before** the MAE threshold is breached.
- A **Loss** is recorded as soon as MAE exceeds the stop level.
- The bot selects the best stop based on **Strike Rate** and **Expectancy**.

**Metrics**:
- **R:R** = Reward (range size) ÷ stop level
- **Expectancy** = (Strike Rate × R:R) − (1 − Strike Rate)
- **Volatility Shifts** = Changes in MAE, range %, and duration
- **Confidence**: Core Setup / Watchlist / Risky, based on strike and EV
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

####################################################################


# ----- START SECTION: Backtest Analyzer Setup -----
if page == "Backtest Analyzer":
    import plotly.express as px
    import plotly.io as pio

    st.header("📊 Backtest Analyzer")

    uploaded_file = st.sidebar.file_uploader("📁 Upload Tactix 1Y Backtest CSV", type=["csv"], key="upload")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df["DayOfWeek"] = df["DayOfWeek"].str.strip().str.title()
        df = df[df["DayOfWeek"].isin(["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])]
# ----- END SECTION: Backtest Analyzer Setup -----


# ----- START SECTION: Recency Filter -----
        st.sidebar.subheader("📅 Recency Filter")
        num_trades = st.sidebar.number_input("Last N Trades per Day", 5, 200, 30, step=5)

        def filter_last_n_per_day(df, n):
            return df.groupby("DayOfWeek", group_keys=False).apply(
                lambda g: g.sort_values("Datetime").tail(n)
            ).reset_index(drop=True)
# ----- END SECTION: Recency Filter -----


# ----- START SECTION: Strategy Selection -----
        strategy = st.radio("Choose Strategy:", ["MAE Optimizer (Base-Hit TP)", "MFE Strategy (Fixed 0.3% MAE)"], key="strategy_selector")
# ----- END SECTION: Strategy Selection -----

# ----- START SECTION: MAE Strategy -----
        if strategy == "MAE Optimizer (Base-Hit TP)":
            st.subheader("🟥 MAE Optimizer: Base-Hit Strategy")

            with st.expander("📘 Strategy Logic & Metrics"):
                st.markdown("""
This module tests a range of stop-loss levels (0.01% to 0.30%) to identify the most effective stop for each weekday.

- A **Win** is counted only if TP is hit **before** the MAE threshold is breached.
- A **Loss** is recorded as soon as MAE exceeds the stop level.
- The bot selects the best stop based on **Strike Rate** and **Expectancy**.

**Metrics**:
- **R:R** = Reward (range size) ÷ stop level
- **Expectancy** = (Strike Rate × R:R) − (1 − Strike Rate)
- **Volatility Shifts** = Changes in MAE, range %, and duration
- **Confidence**: Core Setup / Watchlist / Risky, based on strike and EV
""")

            df["Direction"] = np.where(df["BreakoutDirection"] == "High", "Short", "Long")
            df["RewardPct"] = np.where(
                df["Direction"] == "Short",
                (df["BreakoutClose"] - df["Range_Low"]) / df["BreakoutClose"],
                (df["Range_High"] - df["BreakoutClose"]) / df["BreakoutClose"]
            )

            mae_levels = np.round(np.arange(0.01, 0.31, 0.01), 3)
            results = []
            min_sr = 60

            for day in ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
                group = df[df["DayOfWeek"] == day].copy()
                best_row = None
                best_ev = -np.inf

                for mae in mae_levels:
                    group["Sim_Result"] = np.where((group["MAE"] <= mae) & (group["Hit"] == 1), 1, 0)
                    sr = group["Sim_Result"].mean() * 100
                    if sr < min_sr: continue

                    rr = group[group["Sim_Result"] == 1]["RewardPct"] / (mae / 100)
                    ev = (sr / 100) * np.median(rr) - (1 - sr / 100)

                    recent = group.sort_values("Datetime").tail(num_trades)
                    historic = group.sort_values("Datetime").iloc[:-num_trades]
                    mae_shift = (recent["MAE"].mean() - historic["MAE"].mean()) / historic["MAE"].mean() * 100 if not historic.empty else 0
                    range_shift = (recent["RangePercentage"].mean() - historic["RangePercentage"].mean()) / historic["RangePercentage"].mean() * 100 if not historic.empty else 0
                    dur_shift = (recent["Duration"].mean() - historic["Duration"].mean()) / historic["Duration"].mean() * 100 if not historic.empty else 0

                    if ev > best_ev:
                        best_row = {
                            "DayOfWeek": day, "MAE_Stop": mae, "Strike Rate (%)": round(sr, 2),
                            "R:R": round(np.median(rr), 2), "Expectancy": round(ev, 4),
                            "MAE Change (%)": round(mae_shift, 1), "Range% Change (%)": round(range_shift, 1),
                            "Duration Change (%)": round(dur_shift, 1)
                        }
                        best_ev = ev

                if best_row: results.append(best_row)

            mae_df = pd.DataFrame(results)

            def get_confidence(row):
                score = int(row["Strike Rate (%)"] >= 75) + int(row["Expectancy"] >= 0.15)
                return ["Risky", "Watchlist", "Core Setup"][score]

            mae_df["Confidence"] = mae_df.apply(get_confidence, axis=1)


# ----- START SECTION: MAE Trader Assistant -----
            df_recent = filter_last_n_per_day(df, num_trades)
            summaries = []

            for _, row in mae_df.iterrows():
                day = row["DayOfWeek"]
                stop = row["MAE_Stop"]
                label = row["Confidence"]
                tag = f"{label}"

                filtered = df_recent[df_recent["DayOfWeek"] == day].copy()
                filtered["Sim_Result"] = np.where((filtered["MAE"] <= stop) & (filtered["Hit"] == 1), 1, 0)

                # Win/Loss classification
                filtered["TradeResult"] = np.where(filtered["Sim_Result"] == 1, "Win", "Loss")
                total = len(filtered)
                wins = (filtered["TradeResult"] == "Win").sum()
                losses = (filtered["TradeResult"] == "Loss").sum()

                # Streak counters
                streaks = filtered["TradeResult"].tolist()
                current_win = current_loss = max_win = max_loss = win_streak = loss_streak = 0
                win_runs = []
                loss_runs = []

                for result in streaks:
                    if result == "Win":
                        current_loss = 0
                        win_streak += 1
                        max_win = max(max_win, win_streak)
                        current_win = win_streak
                        if loss_streak:
                            loss_runs.append(loss_streak)
                            loss_streak = 0
                    else:
                        current_win = 0
                        loss_streak += 1
                        max_loss = max(max_loss, loss_streak)
                        current_loss = loss_streak
                        if win_streak:
                            win_runs.append(win_streak)
                            win_streak = 0

                avg_win = round(np.mean(win_runs), 1) if win_runs else 0
                avg_loss = round(np.mean(loss_runs), 1) if loss_runs else 0

                # Volatility + Duration shift
                insight = []
                if row["MAE Change (%)"] > 0: insight.append(f"MAE up {row['MAE Change (%)']:+.1f}%")
                if row["Range% Change (%)"] < 0: insight.append(f"Range down {row['Range% Change (%)']:+.1f}%")
                if row["Duration Change (%)"] > 0: insight.append(f"Duration up {row['Duration Change (%)']:+.1f}%")

                # DCA suitability: low range + high MAE = poor fit
                dca_friendly = "Yes" if row["Range% Change (%)"] > 0 and row["MAE Change (%)"] <= 0 else "No"

                summaries.append({
                    "DayOfWeek": day, "Confidence": label,
                    "Wins": wins, "Losses": losses, "Total": total,
                    "WinStreak": current_win, "LossStreak": current_loss,
                    "MaxWin": max_win, "MaxLoss": max_loss,
                    "AvgWin": avg_win, "AvgLoss": avg_loss,
                    "DCA Friendly": dca_friendly,
                    "Volatility Insight": "; ".join(insight) if insight else "Stable"
                })

            st.subheader("🧠 Trader Assistant – MAE Strategy")
            for row in summaries:
                st.markdown(
                    f"**{row['DayOfWeek']} – {row['Confidence']}**  \n"
                    f"Optimal Stop: `{mae_df.loc[mae_df['DayOfWeek'] == row['DayOfWeek'], 'MAE_Stop'].values[0]:.2f}%` "
                    f"→ SR: `{mae_df.loc[mae_df['DayOfWeek'] == row['DayOfWeek'], 'Strike Rate (%)'].values[0]:.1f}%`, "
                    f"R:R: `{mae_df.loc[mae_df['DayOfWeek'] == row['DayOfWeek'], 'R:R'].values[0]:.2f}`, "
                    f"EV: `{mae_df.loc[mae_df['DayOfWeek'] == row['DayOfWeek'], 'Expectancy'].values[0]:.2f}`  \n"
                    f"{row['Volatility Insight']}  \n"
                    f"Recent: {row['Total']} trades → Wins: {row['Wins']}, Losses: {row['Losses']} | "
                    f"WinStreak: {row['WinStreak']}, Max: {row['MaxWin']}, LossStreak: {row['LossStreak']}  \n"
                    f"DCA Friendly: {row['DCA Friendly']}"
                )

            st.subheader("📊 Optimized MAE Summary")
            st.dataframe(mae_df, use_container_width=True)

            st.subheader("📉 MAE Distribution (Last Trades per Day)")
            fig = px.scatter(df_recent, x="Datetime", y="MAE", color="DayOfWeek", height=600,
                             title="MAE Scatter Plot", category_orders={"DayOfWeek": ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]})
            fig.update_traces(marker=dict(size=8, opacity=1))
            st.plotly_chart(fig, use_container_width=True)
# ----- END SECTION: MAE Strategy -----

        elif strategy == "MFE Strategy (Fixed 0.3% MAE)":
            st.subheader("🟦 MFE Strategy: Target Optimization (0.3% MAE Stop)")

            with st.expander("📘 Strategy Logic & Metrics"):
                st.markdown("""
This module identifies the most profitable take-profit (MFE) targets for trades with a fixed 0.3% stop.

- Only trades that hit TP **and stayed within** 0.3% MAE are considered.
- The bot tests MFE targets from 0.01% to 0.30% and selects the best by **Expectancy**.

**Metrics**:
- **Hit Rate** = % of trades hitting the target
- **R:R** = MFE Target ÷ 0.3
- **Expectancy** = (Hit Rate × R:R) − (1 − Hit Rate)
- **TP Speed** = Minutes to reach TP
- **Entry Edge** = Proximity of entry to range edge
- **Confidence**: Core Setup / Watchlist / Risky
""")

            df_fixed = df[(df["MAE"] <= 0.3) & (df["Hit"] == 1)].copy()
            df_fixed["Hit_dt"] = pd.to_datetime(df_fixed["Hit_dt"])
            df_fixed["TP_Minutes"] = (df_fixed["Hit_dt"] - df_fixed["Datetime"]).dt.total_seconds() / 60
            df_fixed["Direction"] = np.where(df_fixed["BreakoutDirection"] == "High", "Short", "Long")
            df_fixed["EntryEdge"] = np.where(
                df_fixed["Direction"] == "Short",
                (df_fixed["BreakoutClose"] - df_fixed["Range_High"]) / df_fixed["BreakoutClose"],
                (df_fixed["Range_Low"] - df_fixed["BreakoutClose"]) / df_fixed["BreakoutClose"]
            ) * 100

            mfe_steps = np.round(np.arange(0.01, 0.31, 0.01), 3)
            results = []

            for day in ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
                group = df_fixed[df_fixed["DayOfWeek"] == day]
                if group.empty: continue

                best_ev, best_row = -999, None
                for mfe in mfe_steps:
                    hits = (group["MFE"] >= mfe).sum()
                    total = len(group)
                    hr = hits / total if total else 0
                    rr = mfe / 0.3
                    ev = hr * rr - (1 - hr)
                    if ev > best_ev:
                        best_row = {
                            "DayOfWeek": day,
                            "MFE_Target": mfe,
                            "Hit Rate (%)": round(hr * 100, 2),
                            "R:R": round(rr, 2),
                            "Expectancy": round(ev, 4),
                            "Trades": total,
                            "TP Speed (min)": round(group["TP_Minutes"].mean(), 1),
                            "Entry Edge (%)": round(group["EntryEdge"].abs().mean(), 3)
                        }
                        best_ev = ev
                if best_row:
                    results.append(best_row)

            mfe_df = pd.DataFrame(results)

            # Confidence label
            def confidence_label(row):
                score = 0
                if row["Hit Rate (%)"] >= 75: score += 1
                if row["Expectancy"] >= 0.1: score += 1
                return {2: "Core Setup", 1: "Watchlist", 0: "Risky"}[score]

            mfe_df["Confidence"] = mfe_df.apply(confidence_label, axis=1)

            # Win/loss streaks
            def compute_streaks(group, target):
                streaks = []
                curr = []
                for val in (group["MFE"] >= target).astype(int):
                    if not curr or curr[-1] == val:
                        curr.append(val)
                    else:
                        streaks.append((curr[0], len(curr)))
                        curr = [val]
                if curr:
                    streaks.append((curr[0], len(curr)))
                wins = sum(1 for s in streaks if s[0] == 1)
                losses = sum(1 for s in streaks if s[0] == 0)
                win_lengths = [s[1] for s in streaks if s[0] == 1]
                loss_lengths = [s[1] for s in streaks if s[0] == 0]
                return {
                    "Wins": sum(group["MFE"] >= target),
                    "Losses": len(group) - sum(group["MFE"] >= target),
                    "Current Win Streak": streaks[-1][1] if streaks and streaks[-1][0] == 1 else 0,
                    "Current Loss Streak": streaks[-1][1] if streaks and streaks[-1][0] == 0 else 0,
                    "Max Win Streak": max(win_lengths) if win_lengths else 0,
                    "Avg Win Streak": round(np.mean(win_lengths), 1) if win_lengths else 0,
                    "Max Loss Streak": max(loss_lengths) if loss_lengths else 0,
                    "Avg Loss Streak": round(np.mean(loss_lengths), 1) if loss_lengths else 0,
                    "Total Trades": len(group)
                }

            streak_metrics = []
            for _, row in mfe_df.iterrows():
                group = df_fixed[df_fixed["DayOfWeek"] == row["DayOfWeek"]]
                streak_metrics.append(compute_streaks(group, row["MFE_Target"]))

            final_mfe = pd.concat([mfe_df.reset_index(drop=True), pd.DataFrame(streak_metrics)], axis=1)

            # ----- Trader Assistant -----
            st.subheader("🧠 Trader Assistant")
            for _, row in final_mfe.iterrows():
                tag = row["Confidence"]
                st.markdown(
                    f"**{row['DayOfWeek']} – {tag}**  \n"
                    f"Target: `{row['MFE_Target']:.2f}%` → Hit Rate: `{row['Hit Rate (%)']:.1f}%`, "
                    f"R:R: `{row['R:R']:.2f}`, EV: `{row['Expectancy']:.2f}`  \n"
                    f"TP Speed: `{row['TP Speed (min)']}` min | Entry Edge: `{row['Entry Edge (%)']:.2f}%`  \n"
                    f"Recent: {row['Total Trades']} trades → {row['Wins']} Wins / {row['Losses']} Losses | "
                    f"WinStreak: {row['Current Win Streak']}, Max: {row['Max Win Streak']}, "
                    f"LossStreak: {row['Current Loss Streak']}"
                )

            # ----- Final Combined Table -----
            st.subheader("📊 MFE Strategy Summary")
            display_cols = [
                "DayOfWeek", "Confidence", "MFE_Target", "Hit Rate (%)", "R:R", "Expectancy",
                "TP Speed (min)", "Entry Edge (%)", "Wins", "Losses", "Current Win Streak", "Current Loss Streak",
                "Max Win Streak", "Max Loss Streak", "Avg Win Streak", "Avg Loss Streak", "Total Trades"
            ]
            st.dataframe(final_mfe[display_cols].set_index("DayOfWeek"), use_container_width=True)

            # ----- MFE Scatter Plot -----
            st.subheader("📈 MFE Distribution (Last Trades per Day)")
            df_mfe_recent = filter_last_n_per_day(df_fixed, num_trades)
            fig2 = px.scatter(
                df_mfe_recent, x="Datetime", y="MFE", color="DayOfWeek",
                title="MFE Scatter Plot", height=600,
                category_orders={"DayOfWeek": ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]}
            )
            fig2.update_traces(marker=dict(size=8, opacity=1))
            st.plotly_chart(fig2, use_container_width=True)

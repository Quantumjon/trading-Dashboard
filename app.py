# ----- START SECTION: SETUP -----
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Set wide layout once at the top
st.set_page_config(layout="wide")

# Title
st.title("Trading Dashboard")

# Sidebar Strategy Selector
dashboard_page = st.sidebar.radio("Select Dashboard", ["Range Finder", "Backtest Analyzer"])

# Utility functions
def extract_instrument_name(filename):
    base = os.path.basename(filename)
    name = base.split("_")[0]
    return name

def calculate_expectancy(df):
    return (df["StrikeRate"] * df["RR"] + (1 - df["StrikeRate"]) * -1).round(2)

def calculate_efficiency(reward_pct, mae_pct):
    return (reward_pct / mae_pct).round(2) if mae_pct > 0 else np.nan

def assign_volatility_tier(mae, range_pct, duration):
    score = (mae + range_pct) * duration
    if score < 10:
        return "Low"
    elif score < 20:
        return "Medium"
    else:
        return "High"

# For win/loss streaks
def get_streaks(results):
    current_win, current_loss, max_win, max_loss = 0, 0, 0, 0
    win_streaks, loss_streaks = [], []
    for r in results:
        if r == 1:
            current_win += 1
            max_loss = max(max_loss, current_loss)
            if current_loss > 0:
                loss_streaks.append(current_loss)
            current_loss = 0
        else:
            current_loss += 1
            max_win = max(max_win, current_win)
            if current_win > 0:
                win_streaks.append(current_win)
            current_win = 0
    # Final streaks
    if current_win > 0:
        win_streaks.append(current_win)
    if current_loss > 0:
        loss_streaks.append(current_loss)

    return {
        "Current Win Streak": current_win,
        "Current Loss Streak": current_loss,
        "Max Win Streak": max(win_streaks) if win_streaks else 0,
        "Max Loss Streak": max(loss_streaks) if loss_streaks else 0,
        "Avg Win Streak": round(np.mean(win_streaks), 1) if win_streaks else 0,
        "Avg Loss Streak": round(np.mean(loss_streaks), 1) if loss_streaks else 0,
    }

# ----- END SECTION: SETUP -----
# ----- START SECTION: RANGE FINDER -----
if dashboard_page == "Range Finder":
    st.header("Range Finder")

    uploaded_files = st.sidebar.file_uploader("Upload Timeframe CSVs", accept_multiple_files=True, type="csv")

    if uploaded_files:
        instrument_data = {}

        for file in uploaded_files:
            df_temp = pd.read_csv(file)
            instrument = extract_instrument_name(file.name)

            required = [
                "Timeframe", "Setup Count", "StrikeRate", "RR", "Expectancy", 
                "Range Avg", "MAE Avg", "Duration Avg"
            ]
            # ---- KeyError fix: Check for required columns ----
            missing_cols = [col for col in required if col not in df_temp.columns]
            if missing_cols:
                st.error(f"Missing columns in {file.name}: {', '.join(missing_cols)}")
                continue

            df_temp = df_temp.loc[:, required]
            instrument_data[instrument] = df_temp

        if instrument_data:
            st.subheader("Filtered Instrument Setups")
            min_strike = st.sidebar.slider("Minimum Weighted Strike Rate (%)", 0, 100, 75)

            weight_map = {"1M": 0.4, "3M": 0.3, "6M": 0.2, "1Y": 0.1}
            results = []

            for instrument, df in instrument_data.items():
                if not all(tf in df["Timeframe"].values for tf in weight_map.keys()):
                    continue

                df.set_index("Timeframe", inplace=True)
                weighted_sr = sum(df.at[tf, "StrikeRate"] * weight_map[tf] for tf in weight_map)
                weighted_rr = sum(df.at[tf, "RR"] * weight_map[tf] for tf in weight_map)
                weighted_ev = sum(df.at[tf, "Expectancy"] * weight_map[tf] for tf in weight_map)

                label = "Core Setup" if weighted_sr >= min_strike else "Watchlist" if weighted_sr >= (min_strike - 5) else "Risky"
                
                results.append({
                    "Instrument": instrument,
                    "Weighted SR": round(weighted_sr, 2),
                    "Weighted RR": round(weighted_rr, 2),
                    "Expectancy": round(weighted_ev, 2),
                    "Label": label
                })

            if results:
                df_results = pd.DataFrame(results)
                st.dataframe(df_results)
            else:
                st.warning("No instruments met the required conditions.")
# ----- END SECTION: RANGE FINDER -----
# ----- START SECTION: BACKTEST ANALYZER - MAE STRATEGY -----
elif dashboard_page == "Backtest Analyzer":
    st.header("Backtest Analyzer")

    uploaded_files = st.sidebar.file_uploader("Upload Backtest CSVs", type="csv", accept_multiple_files=True)

    if uploaded_files:
        all_data = []
        for file in uploaded_files:
            df = pd.read_csv(file)
            instrument = extract_instrument_name(file.name)
            df["Instrument"] = instrument
            all_data.append(df)

        data = pd.concat(all_data, ignore_index=True)
        data["Datetime"] = pd.to_datetime(data["Datetime"])
        data["DayOfWeek"] = pd.Categorical(data["DayOfWeek"], 
                                           categories=["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                                           ordered=True)

        # Strategy selector
        strategy = st.radio("Select Strategy", ["MAE Optimizer", "MFE Strategy"])

        # Recency filter
        st.subheader("Trade Recency Filter")
        trade_limit = st.number_input("Number of recent trades per weekday", min_value=10, value=30)

        if strategy == "MAE Optimizer":
            st.subheader("MAE Strategy: Optimal Stop Placement")

            # ----- Tooltip -----
            with st.expander("**How this works**"):
                st.markdown("""
- Trades are grouped by original range formation day.
- Bot simulates different MAE stop levels (e.g., 0.05% to 0.3%) to find best strike rate and expectancy.
- Confidence labels:
  - **Core Setup** = Strong strike rate + EV
  - **Watchlist** = Near-threshold setups
  - **Risky** = Below 75% strike rate
- Regime shift and streak insights are based on recent vs historical samples.
                """)

            # Filter to last N trades per weekday
            data = data.sort_values("Datetime")
            recent = data.groupby("DayOfWeek").tail(trade_limit)

            summary_rows = []
            for day, df_day in recent.groupby("DayOfWeek"):
                total = len(df_day)
                best_sr = 0
                best_row = None
                for stop in np.arange(0.05, 0.31, 0.01):
                    df_day["Stopped"] = df_day["MAE"] > stop
                    df_day["ResultSim"] = np.where(df_day["Stopped"], 0, df_day["Hit"])
                    sr = df_day["ResultSim"].mean()
                    rr = df_day["RangePercentage"].mean() / stop if stop > 0 else np.nan
                    ev = sr * rr - (1 - sr)
                    if sr >= 0.75 and ev > 0:
                        if sr > best_sr:
                            best_sr = sr
                            best_row = {
                                "Day": day,
                                "Stop %": round(stop, 3),
                                "StrikeRate": round(sr, 2),
                                "RR": round(rr, 2),
                                "EV": round(ev, 2),
                            }

                if best_row:
                    df_day["Filtered"] = df_day["MAE"] <= best_row["Stop %"]
                    streak_data = get_streaks(df_day[df_day["Filtered"]]["ResultSim"])
                    mae = df_day["MAE"].mean()
                    r_pct = df_day["RangePercentage"].mean()
                    dur = df_day["Duration"].mean()
                    tier = assign_volatility_tier(mae, r_pct, dur)
                    eff = calculate_efficiency(df_day["RangePercentage"].mean(), mae)
                    label = "Core Setup"
                else:
                    best_row = {
                        "Day": day, "Stop %": np.nan, "StrikeRate": 0, "RR": 0, "EV": 0
                    }
                    tier = "High"
                    eff = 0
                    label = "Risky"
                    streak_data = {
                        "Current Win Streak": 0,
                        "Current Loss Streak": 0,
                        "Max Win Streak": 0,
                        "Max Loss Streak": 0,
                        "Avg Win Streak": 0,
                        "Avg Loss Streak": 0,
                    }

                best_row.update({
                    "Confidence": label,
                    "Volatility": tier,
                    "Efficiency": eff,
                    **streak_data,
                    "Total Trades": total
                })
                summary_rows.append(best_row)

            df_summary = pd.DataFrame(summary_rows)
            df_summary = df_summary.sort_values("Day")

            st.markdown("### Trader Assistant")
            for _, row in df_summary.iterrows():
                st.markdown(f"**{row['Confidence']} – {row['Day']}**  \n"
                            f"Stop: `{row['Stop %']}%` | SR: `{row['StrikeRate']}` | R:R: `{row['RR']}` | EV: `{row['EV']}`  \n"
                            f"Volatility: `{row['Volatility']}` | Efficiency: `{row['Efficiency']}`  \n"
                            f"Wins: `{row['Current Win Streak']}` | Losses: `{row['Current Loss Streak']}` | Max Win: `{row['Max Win Streak']}` | Max Loss: `{row['Max Loss Streak']}`  \n"
                            f"Avg Win: `{row['Avg Win Streak']}` | Avg Loss: `{row['Avg Loss Streak']}`  \n"
                            f"Sample: `{row['Total Trades']}` trades")

            st.markdown("### MAE Strategy Summary")
            st.dataframe(df_summary.set_index("Day"))

            # Scatter plot
            st.markdown("### MAE Distribution")
            fig, ax = plt.subplots(figsize=(12, 6))
            for day, group in recent.groupby("DayOfWeek"):
                ax.scatter(group["Datetime"], group["MAE"], label=day, s=8)
            ax.set_ylabel("MAE %")
            ax.set_title("MAE Distribution by Date")
            ax.legend()
            st.pyplot(fig)

# ----- START SECTION: BACKTEST ANALYZER - MFE STRATEGY -----
        elif strategy == "MFE Strategy":
            st.subheader("MFE Strategy: TP Target Performance (Fixed Stop)")

            # ----- Tooltip -----
            with st.expander("**How this works**"):
                st.markdown("""
- Uses a fixed MAE stop of 0.3% to evaluate take-profit levels.
- Evaluates how often MFE targets are reached and how quickly.
- Efficiency = MFE ÷ 0.3.
- Entry Edge = Distance from breakout close to opposite range side.
- Clustering = TP speed + volatility pattern by weekday.
                """)

            data = data.sort_values("Datetime")
            recent = data.groupby("DayOfWeek").tail(trade_limit)

            summary_rows = []
            for day, df_day in recent.groupby("DayOfWeek"):
                total = len(df_day)
                target_hit = (df_day["MFE"] >= df_day["RangePercentage"]).sum()
                sr = round(target_hit / total, 2)
                rr = 1  # Since target is always Range %
                ev = round(sr * rr - (1 - sr), 2)
                mfe = df_day["MFE"].mean()
                mae = df_day["MAE"].mean()
                eff = calculate_efficiency(df_day["RangePercentage"].mean(), 0.3)
                speed = df_day["Duration"].mean()
                edge = df_day["RangePercentage"].mean() - df_day["BreakoutClose"].sub(
                    np.where(df_day["BreakoutDirection"] == "Long", df_day["RangeHigh"], df_day["RangeLow"])
                ).abs().mean()
                tier = assign_volatility_tier(mae, df_day["RangePercentage"].mean(), speed)

                # Simulated fixed stop streaks
                df_day["Stopped"] = df_day["MAE"] > 0.3
                df_day["ResultSim"] = np.where(df_day["Stopped"], 0, df_day["Hit"])
                streak_data = get_streaks(df_day["ResultSim"])

                label = "Core Setup" if sr >= 0.75 else "Watchlist" if sr >= 0.7 else "Risky"

                summary_rows.append({
                    "Day": day,
                    "Target %": round(df_day["RangePercentage"].mean(), 3),
                    "StrikeRate": sr,
                    "RR": rr,
                    "EV": ev,
                    "Efficiency": round(eff, 2),
                    "Volatility": tier,
                    "TP Speed": round(speed, 2),
                    "Entry Edge": round(edge, 3),
                    "Confidence": label,
                    **streak_data,
                    "Total Trades": total
                })

            df_summary = pd.DataFrame(summary_rows)
            df_summary = df_summary.sort_values("Day")

            st.markdown("### Trader Assistant")
            for _, row in df_summary.iterrows():
                st.markdown(f"**{row['Confidence']} – {row['Day']}**  \n"
                            f"Target: `{row['Target %']}%` | SR: `{row['StrikeRate']}` | R:R: `{row['RR']}` | EV: `{row['EV']}`  \n"
                            f"Volatility: `{row['Volatility']}` | Efficiency: `{row['Efficiency']}`  \n"
                            f"TP Speed: `{row['TP Speed']}` | Entry Edge: `{row['Entry Edge']}`  \n"
                            f"Wins: `{row['Current Win Streak']}` | Losses: `{row['Current Loss Streak']}` | Max Win: `{row['Max Win Streak']}` | Max Loss: `{row['Max Loss Streak']}`  \n"
                            f"Avg Win: `{row['Avg Win Streak']}` | Avg Loss: `{row['Avg Loss Streak']}`  \n"
                            f"Sample: `{row['Total Trades']}` trades")

            st.markdown("### MFE Strategy Summary")
            st.dataframe(df_summary.set_index("Day"))

            # Scatter plot
            st.markdown("### MFE Distribution")
            fig, ax = plt.subplots(figsize=(12, 6))
            for day, group in recent.groupby("DayOfWeek"):
                ax.scatter(group["Datetime"], group["MFE"], label=day, s=8)
            ax.set_ylabel("MFE %")
            ax.set_title("MFE Distribution by Date")
            ax.legend()
            st.pyplot(fig)
# ----- END SECTION: BACKTEST ANALYZER - MFE STRATEGY -----
# ----- START SECTION: SCRIPT END -----
    else:
        st.warning("Please upload at least one backtest CSV file to begin analysis.")

else:
    st.info("Upload CSVs from your backtests or timeframe exports using the sidebar to begin.")
# ----- END SECTION: SCRIPT END -----



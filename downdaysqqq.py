import pandas as pd
import yfinance as yf

# -----------------------------
# Settings
# -----------------------------
ticker = "QQQ"
start_date = "2010-01-01"   # change as needed
end_date = None             # None = up to latest available data

# -----------------------------
# Download daily data
# yfinance download() supports interval='1d'
# start is inclusive, end is exclusive
# -----------------------------
df = yf.download(
    ticker,
    start=start_date,
    end=end_date,
    interval="1d",
    auto_adjust=False,
    progress=False
)

if df.empty:
    raise ValueError("No data downloaded. Check ticker/date range.")

# Handle possible multi-index columns from yfinance
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Keep only needed columns
df = df[["Open", "Close"]].dropna().copy()

# -----------------------------
# Define candle direction
# Red/down day = Close < Open
# Green/up day = Close > Open
# Doji = Close == Open
# -----------------------------
df["weekday_num"] = df.index.weekday   # Monday=0 ... Friday=4
df["weekday"] = df.index.day_name()

df["red"] = df["Close"] < df["Open"]
df["green"] = df["Close"] > df["Open"]
df["doji"] = df["Close"] == df["Open"]

# -----------------------------
# Aggregate by weekday
# -----------------------------
summary = (
    df.groupby(["weekday_num", "weekday"])
      .agg(
          total_days=("red", "size"),
          red_days=("red", "sum"),
          green_days=("green", "sum"),
          doji_days=("doji", "sum")
      )
      .reset_index()
)

summary["red_pct"] = (summary["red_days"] / summary["total_days"] * 100).round(2)
summary["green_pct"] = (summary["green_days"] / summary["total_days"] * 100).round(2)
summary["avg_oc_return_pct"] = (
    df.assign(oc_return_pct=(df["Close"] / df["Open"] - 1) * 100)
      .groupby([df["weekday_num"], df["weekday"]])["oc_return_pct"]
      .mean()
      .round(4)
      .values
)

# Monday -> Friday order
summary = summary.sort_values("weekday_num").drop(columns="weekday_num")

print(f"\nWeekday candle stats for {ticker} since {start_date}\n")
print(summary.to_string(index=False))

# -----------------------------
# Best short candidates
# 1) highest % of red candles
# 2) lowest average open->close return
# -----------------------------
best_by_red_pct = summary.sort_values("red_pct", ascending=False).iloc[0]
best_by_avg_return = summary.sort_values("avg_oc_return_pct", ascending=True).iloc[0]

print("\nMost frequently red weekday:")
print(
    f"{best_by_red_pct['weekday']} "
    f"({best_by_red_pct['red_days']}/{best_by_red_pct['total_days']} red, "
    f"{best_by_red_pct['red_pct']}%)"
)

print("\nWeakest average open->close weekday:")
print(
    f"{best_by_avg_return['weekday']} "
    f"(avg open->close return = {best_by_avg_return['avg_oc_return_pct']}%)"
)
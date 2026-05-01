import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import skew, kurtosis
from datetime import datetime

# Optional libraries
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except Exception:
    HMM_AVAILABLE = False

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except Exception:
    RUPTURES_AVAILABLE = False


# =========================
# COLORS (ANSI)
# =========================
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"     # bright green
YELLOW = "\033[93m"    # bright yellow
RED = "\033[91m"       # bright red
CYAN = "\033[96m"
MAGENTA = "\033[95m"


# =========================
# CONFIG
# =========================
TICKER = "QQQ"
START_DATE = "2010-01-01"
HMM_STATES = 3
HMM_LOOKBACK = 252 * 3        # fit HMM on ~3 years of daily data
CHANGEPOINT_LOOKBACK = 252    # detect recent structural changes on ~1 year
SHORT_WINDOWS = [20, 63, 126] # ~1m, 3m, 6m
LONG_WINDOWS = [252]          # ~12m

# Monte Carlo config
MC_HORIZON_DAYS = 20          # forecast horizon
MC_NUM_PATHS = 10000          # number of simulated paths

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


# =========================
# DATA DOWNLOAD
# =========================
def download_data(ticker=TICKER, start=START_DATE):
    df = yf.download(
        ticker,
        start=start,
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise ValueError("No data downloaded.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    needed = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df = df[[c for c in needed if c in df.columns]].copy()
    df = df.dropna().copy()

    df["ret_cc"] = df["Adj Close"].pct_change()
    df["log_ret"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))
    df["ret_oc"] = df["Close"] / df["Open"] - 1
    df["ret_co"] = df["Open"] / df["Close"].shift(1) - 1
    df["hl_range"] = df["High"] / df["Low"] - 1
    df["weekday"] = df.index.day_name()
    df["month"] = df.index.month
    df["year"] = df.index.year
    return df.dropna()


# =========================
# HELPERS
# =========================
def fmt_pct(x, digits=2):
    if pd.isna(x):
        return "nan"
    return f"{x * 100:.{digits}f}%"

def fmt_num(x, digits=4):
    if pd.isna(x):
        return "nan"
    return f"{x:.{digits}f}"

def annualized_sharpe(returns):
    r = pd.Series(returns).dropna()
    if len(r) < 2 or r.std() == 0:
        return np.nan
    return (r.mean() / r.std()) * np.sqrt(252)

def max_drawdown(series):
    s = pd.Series(series).dropna()
    if s.empty:
        return np.nan
    cum = (1 + s).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1
    return dd.min()

def calc_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = down.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def trend_strength(series, window):
    s = pd.Series(series).dropna().tail(window)
    if len(s) < window:
        return np.nan
    x = np.arange(len(s))
    y = np.log(s.values)
    return np.polyfit(x, y, 1)[0]

def hurst_exponent(ts, max_lag=20):
    ts = pd.Series(ts).dropna()
    if len(ts) < max_lag + 10:
        return np.nan
    lags = range(2, max_lag)
    tau = [np.std(ts.diff(lag).dropna()) for lag in lags]
    if any(t <= 0 for t in tau):
        return np.nan
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def downside_upside_capture(returns, benchmark=None):
    r = pd.Series(returns).dropna()
    if benchmark is None:
        benchmark = r
    b = pd.Series(benchmark).dropna()
    idx = r.index.intersection(b.index)
    r = r.loc[idx]
    b = b.loc[idx]
    up = r[b > 0].mean()
    down = r[b < 0].mean()
    return up, down


# =========================
# SNAPSHOT STATS
# =========================
def window_stats(df, window):
    d = df.tail(window).copy()
    if len(d) < 20:
        return None

    price = d["Adj Close"]
    ret = d["ret_cc"]
    intraday = d["ret_oc"]
    overnight = d["ret_co"]

    out = {
        "window": window,
        "start": d.index[0].date(),
        "end": d.index[-1].date(),
        "total_return": price.iloc[-1] / price.iloc[0] - 1,
        "ann_vol": ret.std() * np.sqrt(252),
        "sharpe": annualized_sharpe(ret),
        "max_dd": max_drawdown(ret),
        "pct_up_days": (ret > 0).mean(),
        "pct_red_intraday": (intraday < 0).mean(),
        "avg_intraday": intraday.mean(),
        "avg_overnight": overnight.mean(),
        "rsi_14": calc_rsi(price, 14).iloc[-1],
        "distance_20dma": price.iloc[-1] / price.rolling(20).mean().iloc[-1] - 1,
        "distance_50dma": price.iloc[-1] / price.rolling(50).mean().iloc[-1] - 1,
        "distance_200dma": price.iloc[-1] / price.rolling(200).mean().iloc[-1] - 1,
        "trend_20": trend_strength(price, min(20, len(price))),
        "trend_63": trend_strength(price, min(63, len(price))),
        "hurst": hurst_exponent(price.pct_change().dropna(), max_lag=20),
    }
    return out


# =========================
# WEEKDAY + MONTH EFFECTS
# =========================
def weekday_stats(df, window=252):
    d = df.tail(window).copy()
    g = (
        d.groupby("weekday")
         .agg(
             total_days=("ret_oc", "size"),
             red_days=("ret_oc", lambda x: (x < 0).sum()),
             avg_intraday=("ret_oc", "mean"),
             avg_overnight=("ret_co", "mean"),
             avg_close_close=("ret_cc", "mean")
         )
         .reset_index()
    )
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    g["weekday"] = pd.Categorical(g["weekday"], categories=order, ordered=True)
    g = g.sort_values("weekday")
    g["red_pct"] = g["red_days"] / g["total_days"]
    return g

def month_stats(df, window=252*5):
    d = df.tail(window).copy()
    g = (
        d.groupby("month")
         .agg(
             obs=("ret_cc", "size"),
             avg_close_close=("ret_cc", "mean"),
             avg_intraday=("ret_oc", "mean"),
             avg_overnight=("ret_co", "mean")
         )
         .reset_index()
         .sort_values("month")
    )
    return g


# =========================
# VOLATILITY REGIMES
# =========================
def volatility_regime_stats(df, vol_window=20, analysis_window=252*3):
    d = df.tail(analysis_window).copy()
    d["rv20"] = d["ret_cc"].rolling(vol_window).std() * np.sqrt(252)
    d = d.dropna()

    if len(d) < 100:
        return None, None

    d["vol_bucket"] = pd.qcut(d["rv20"], q=3, labels=["LowVol", "MidVol", "HighVol"])

    summary = (
        d.groupby("vol_bucket")
         .agg(
             obs=("ret_cc", "size"),
             avg_ret=("ret_cc", "mean"),
             avg_intraday=("ret_oc", "mean"),
             avg_overnight=("ret_co", "mean"),
             std_ret=("ret_cc", "std"),
             pct_up=("ret_cc", lambda x: (x > 0).mean())
         )
         .reset_index()
    )

    latest = d.iloc[-1][["rv20", "vol_bucket"]]
    return summary, latest


# =========================
# HMM REGIME
# =========================
def hmm_regime_analysis(df, n_states=HMM_STATES, lookback=HMM_LOOKBACK):
    if not HMM_AVAILABLE:
        return None, "hmmlearn not installed"

    d = df.tail(lookback).copy()
    d["vol20"] = d["log_ret"].rolling(20).std()
    d = d.dropna()

    if len(d) < 120:
        return None, "not enough data for HMM"

    X = d[["log_ret", "vol20"]].replace([np.inf, -np.inf], np.nan).dropna().values
    if len(X) < 120:
        return None, "not enough clean HMM feature rows"

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=300,
        random_state=42
    )

    model.fit(X)
    states = model.predict(X)
    d = d.iloc[-len(states):].copy()
    d["state"] = states

    state_summary = (
        d.groupby("state")
         .agg(
             obs=("ret_cc", "size"),
             mean_ret=("ret_cc", "mean"),
             mean_intraday=("ret_oc", "mean"),
             mean_overnight=("ret_co", "mean"),
             mean_vol20=("vol20", "mean")
         )
         .reset_index()
         .sort_values("mean_ret", ascending=False)
         .reset_index(drop=True)
    )

    regime_labels = {}
    ordered_states = state_summary["state"].tolist()

    if len(ordered_states) == 3:
        regime_labels[ordered_states[0]] = "Bull"
        regime_labels[ordered_states[1]] = "Neutral"
        regime_labels[ordered_states[2]] = "Bear"
    else:
        for i, s in enumerate(ordered_states):
            regime_labels[s] = f"Rank{i+1}"

    d["regime"] = d["state"].map(regime_labels)
    current_state = int(d["state"].iloc[-1])
    current_regime = regime_labels[current_state]

    recent_regime_counts = d["regime"].tail(63).value_counts(normalize=True)

    out = {
        "current_state": current_state,
        "current_regime": current_regime,
        "state_summary": state_summary,
        "recent_regime_mix_63d": recent_regime_counts,
        "last_10_regimes": d["regime"].tail(10)
    }
    return out, None


# =========================
# CHANGE-POINT
# =========================
def changepoint_analysis(df, lookback=CHANGEPOINT_LOOKBACK):
    if not RUPTURES_AVAILABLE:
        return None, "ruptures not installed"

    d = df.tail(lookback).copy()
    d = d.dropna()

    if len(d) < 80:
        return None, "not enough data for changepoint detection"

    signal = d["log_ret"].values.reshape(-1, 1)

    try:
        algo = rpt.Pelt(model="rbf").fit(signal)
        bkps = algo.predict(pen=10)
    except Exception as e:
        return None, f"changepoint error: {e}"

    break_ix = [b for b in bkps if b < len(d)]
    break_dates = [d.index[b - 1].date() for b in break_ix if b - 1 < len(d)]

    return {
        "lookback_days": lookback,
        "break_count": len(break_dates),
        "break_dates": break_dates[-10:]
    }, None


# =========================
# Z-SCORE STUDIES
# =========================
def zscore_reversion_stats(df, ma_windows=(20, 50), forward_days=(1, 5, 10), lookback=252*5):
    d = df.tail(lookback).copy()
    results = []

    for ma in ma_windows:
        roll_mean = d["Adj Close"].rolling(ma).mean()
        roll_std = d["Adj Close"].rolling(ma).std()
        z = (d["Adj Close"] - roll_mean) / roll_std
        d[f"z_{ma}"] = z

        for fwd in forward_days:
            d[f"fwd_{fwd}"] = d["Adj Close"].shift(-fwd) / d["Adj Close"] - 1

            subsets = {
                "z<=-2": d.loc[d[f"z_{ma}"] <= -2, f"fwd_{fwd}"],
                "-2<z<=-1": d.loc[(d[f"z_{ma}"] > -2) & (d[f"z_{ma}"] <= -1), f"fwd_{fwd}"],
                "|z|<1": d.loc[d[f"z_{ma}"].abs() < 1, f"fwd_{fwd}"],
                "1<=z<2": d.loc[(d[f"z_{ma}"] >= 1) & (d[f"z_{ma}"] < 2), f"fwd_{fwd}"],
                "z>=2": d.loc[d[f"z_{ma}"] >= 2, f"fwd_{fwd}"],
            }

            for bucket, vals in subsets.items():
                vals = vals.dropna()
                results.append({
                    "ma": ma,
                    "fwd_days": fwd,
                    "bucket": bucket,
                    "obs": len(vals),
                    "avg_fwd_return": vals.mean() if len(vals) else np.nan,
                    "median_fwd_return": vals.median() if len(vals) else np.nan,
                    "pct_up": (vals > 0).mean() if len(vals) else np.nan
                })

    return pd.DataFrame(results)


# =========================
# AUTOCORR & MA STACK
# =========================
def autocorr_stats(df, lags=(1, 2, 3, 5, 10, 20), window=252):
    d = df.tail(window)["ret_cc"].dropna()
    rows = []
    for lag in lags:
        rows.append({"lag": lag, "autocorr": d.autocorr(lag=lag)})
    return pd.DataFrame(rows)

def rolling_relative_position(df):
    d = df.copy()
    d["ma20"] = d["Adj Close"].rolling(20).mean()
    d["ma50"] = d["Adj Close"].rolling(50).mean()
    d["ma200"] = d["Adj Close"].rolling(200).mean()

    latest = d.iloc[-1]
    regime = []
    if latest["Adj Close"] > latest["ma200"]:
        regime.append("Above 200DMA")
    else:
        regime.append("Below 200DMA")

    if latest["ma20"] > latest["ma50"] > latest["ma200"]:
        regime.append("Bull stack")
    elif latest["ma20"] < latest["ma50"] < latest["ma200"]:
        regime.append("Bear stack")
    else:
        regime.append("Mixed stack")

    return latest, regime


# =========================
# MONTE CARLO SIMULATION
# =========================
def monte_carlo_simulation(df, horizon=MC_HORIZON_DAYS, n_paths=MC_NUM_PATHS):
    daily_returns = df["ret_cc"].dropna()
    if len(daily_returns) < 100:
        return None

    last_price = df["Adj Close"].iloc[-1]
    rand_rets = np.random.choice(daily_returns.values, size=(n_paths, horizon), replace=True)
    cum_rets = (1 + rand_rets).prod(axis=1) - 1
    future_prices = last_price * (1 + cum_rets)

    summary = {
        "last_price": last_price,
        "horizon": horizon,
        "n_paths": n_paths,
        "mean_return": np.mean(cum_rets),
        "median_return": np.median(cum_rets),
        "pct5": np.percentile(cum_rets, 5),
        "pct25": np.percentile(cum_rets, 25),
        "pct75": np.percentile(cum_rets, 75),
        "pct95": np.percentile(cum_rets, 95),
        "mean_price": np.mean(future_prices),
        "pct5_price": np.percentile(future_prices, 5),
        "pct95_price": np.percentile(future_prices, 95),
    }
    return summary


# =========================
# PRINT HELPERS
# =========================
def print_header(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

def print_window_table(stats_list):
    rows = []
    for s in stats_list:
        rows.append({
            "Window": s["window"],
            "Start": str(s["start"]),
            "End": str(s["end"]),
            "TotRet": fmt_pct(s["total_return"]),
            "AnnVol": fmt_pct(s["ann_vol"]),
            "Sharpe": fmt_num(s["sharpe"], 2),
            "MaxDD": fmt_pct(s["max_dd"]),
            "UpDays": fmt_pct(s["pct_up_days"]),
            "RedIntra": fmt_pct(s["pct_red_intraday"]),
            "AvgIntra": fmt_pct(s["avg_intraday"], 3),
            "AvgON": fmt_pct(s["avg_overnight"], 3),
            "RSI14": fmt_num(s["rsi_14"], 1),
            "Dist20": fmt_pct(s["distance_20dma"]),
            "Dist50": fmt_pct(s["distance_50dma"]),
            "Dist200": fmt_pct(s["distance_200dma"]),
            "Hurst": fmt_num(s["hurst"], 3),
        })
    print(pd.DataFrame(rows).to_string(index=False))

def print_weekday_stats(title, df_wd):
    print(f"\n{title}")
    tmp = df_wd.copy()
    tmp["red_pct"] = tmp["red_pct"].map(lambda x: f"{x*100:.2f}%")
    tmp["avg_intraday"] = tmp["avg_intraday"].map(lambda x: f"{x*100:.3f}%")
    tmp["avg_overnight"] = tmp["avg_overnight"].map(lambda x: f"{x*100:.3f}%")
    tmp["avg_close_close"] = tmp["avg_close_close"].map(lambda x: f"{x*100:.3f}%")
    print(tmp.to_string(index=False))

def print_df_pct(df_in, pct_cols):
    df = df_in.copy()
    for c in pct_cols:
        if c in df.columns:
            df[c] = df[c].map(lambda x: f"{x*100:.3f}%")
    print(df.to_string(index=False))


# =========================
# COLORED SUMMARY
# =========================
def print_colored_summary(df, stats_list, hmm_out, vol_latest, wd_6m, zr):
    w20 = next((s for s in stats_list if s["window"] == 20), None)
    w63 = next((s for s in stats_list if s["window"] == 63), None)
    w252 = next((s for s in stats_list if s["window"] == 252), None)

    latest_price = df["Adj Close"].iloc[-1]
    rsi = w20["rsi_14"] if w20 else np.nan
    dist20 = w20["distance_20dma"] if w20 else np.nan
    dist50 = w63["distance_50dma"] if w63 else np.nan if w63 else np.nan
    dist200 = w252["distance_200dma"] if w252 else np.nan

    print_header("COLORED SUMMARY (HUMAN-READABLE)")

    # 1) Short-term trend
    line = "Short-term (20d): "
    if w20:
        if w20["total_return"] > 0.10 and rsi >= 70 and dist20 > 0.03:
            line += "Strong uptrend, overbought, momentum regime."
            print(GREEN + BOLD + line + RESET)
        elif w20["total_return"] > 0.05 and rsi >= 60:
            line += "Uptrend with healthy momentum."
            print(GREEN + line + RESET)
        elif w20["total_return"] < -0.05 and rsi <= 40:
            line += "Short-term downtrend / oversold."
            print(RED + BOLD + line + RESET)
        else:
            line += "Mixed / neutral short-term."
            print(line)
    else:
        print("Short-term (20d): not enough data.")

    # 2) Medium-term trend
    line = "Medium-term (3–12m): "
    if w63 and w252:
        if w252["total_return"] > 0.20 and w252["sharpe"] > 1.5 and dist200 > 0.05:
            line += "Bullish medium-term trend, strong last year."
            print(GREEN + BOLD + line + RESET)
        elif w252["total_return"] > 0.05:
            line += "Mild to moderate uptrend over past year."
            print(GREEN + line + RESET)
        elif w252["total_return"] < -0.05:
            line += "Medium-term drawdown / weak year."
            print(RED + line + RESET)
        else:
            line += "Sideways medium-term behavior."
            print(line)
    else:
        print("Medium-term (3–12m): insufficient data.")

    # 3) Environment for short bias
    short_line = "Environment for short intraday plays: "
    th_6m = wd_6m[wd_6m["weekday"] == "Thursday"].iloc[0] if not wd_6m.empty else None

    bearish_regime = hmm_out and hmm_out["current_regime"] == "Bear"
    if th_6m is not None:
        th_red = th_6m["red_days"] / th_6m["total_days"]
        th_intra = th_6m["avg_intraday"]
    else:
        th_red, th_intra = np.nan, np.nan

    if w252 and w252["total_return"] > 0.20 and not bearish_regime:
        short_line += "Overall bull year; short setups work best as tactical fades."
        print(YELLOW + short_line + RESET)
    elif bearish_regime or (th_red > 0.6 and th_intra < 0):
        short_line += "Favorable for tactical shorts on weak days (e.g., Thursdays recently)."
        print(GREEN + short_line + RESET)
    else:
        short_line += "No strong statistical edge for aggressive shorting."
        print(short_line)

    # 4) Regime / risk notes
    line = "Regime / risk: "
    if vol_latest is not None:
        vol_val = vol_latest["rv20"] * 100
        vol_bucket = vol_latest["vol_bucket"]
    else:
        vol_val, vol_bucket = np.nan, "Unknown"

    if hmm_out:
        reg = hmm_out["current_regime"]
        line += f"HMM={reg}, vol20≈{vol_val:.1f}%, bucket={vol_bucket}."
        if reg == "Bull" and vol_bucket == "LowVol":
            line += " Calm bull regime; dips tend to be bought."
            print(GREEN + line + RESET)
        elif reg == "Bear" and vol_bucket in ["MidVol", "HighVol"]:
            line += " Choppier / risk-off flavor; watch position sizing."
            print(RED + line + RESET)
        else:
            print(YELLOW + line + RESET)
    else:
        print("Regime / risk: HMM unavailable.")

    # 5) Mean-reversion/trend takeaway
    mr_line = "Mean-reversion / trend: "
    if not zr.empty:
        deep_os = zr[(zr["ma"] == 20) & (zr["fwd_days"] == 1) & (zr["bucket"] == "z<=-2")]
        mod_ob = zr[(zr["ma"] == 20) & (zr["fwd_days"] == 10) & (zr["bucket"] == "1<=z<2")]
        if not deep_os.empty and not mod_ob.empty:
            os_ret = deep_os["avg_fwd_return"].iloc[0]
            ob_ret = mod_ob["avg_fwd_return"].iloc[0]
            mr_line += (
                f"Deep oversold (20d z<=-2) tends to bounce next day (~{fmt_pct(os_ret)}), "
                f"while moderate overbought often trends higher over ~10d (~{fmt_pct(ob_ret)})."
            )
            print(CYAN + mr_line + RESET)
        else:
            print(mr_line + "patterns exist but sample is thin.")
    else:
        print(mr_line + "no Z-score data.")


# =========================
# DAY-OF-WEEK FOCUS (TODAY)
# =========================
def get_today_weekday_summary(df, wd_long, wd_12m):
    """
    Use the system's current weekday (calendar day) instead of the last bar's date.
    If it's Sat/Sun, fall back to last trading day in data.
    """
    today_idx = datetime.today().weekday()  # Monday=0 ... Sunday=6

    if today_idx > 4:  # Saturday or Sunday → use last trading day's weekday
        weekday_name = df.index[-1].day_name()
    else:
        weekday_name = WEEKDAY_ORDER[today_idx]

    row_long = wd_long[wd_long["weekday"] == weekday_name]
    row_12m = wd_12m[wd_12m["weekday"] == weekday_name]

    long_msg = "No long-sample data for this weekday."
    short_msg = "No 12-month data for this weekday."

    if not row_long.empty:
        r = row_long.iloc[0]
        long_msg = (
            f"Long sample ({r['total_days']} {weekday_name}s): "
            f"avg intraday={fmt_pct(r['avg_intraday'], 3)}, "
            f"avg overnight={fmt_pct(r['avg_overnight'], 3)}, "
            f"avg close→close={fmt_pct(r['avg_close_close'], 3)}, "
            f"red_intraday={fmt_pct(r['red_days']/r['total_days'], 2)}"
        )

    if not row_12m.empty:
        r = row_12m.iloc[0]
        short_msg = (
            f"Last 12 months ({r['total_days']} {weekday_name}s): "
            f"avg intraday={fmt_pct(r['avg_intraday'], 3)}, "
            f"avg overnight={fmt_pct(r['avg_overnight'], 3)}, "
            f"avg close→close={fmt_pct(r['avg_close_close'], 3)}, "
            f"red_intraday={fmt_pct(r['red_days']/r['total_days'], 2)}"
        )

    return weekday_name, long_msg, short_msg


def print_today_weekday_behavior_block(weekday_name, long_msg, short_msg):
    print_header(f"{weekday_name.upper()} HISTORICAL BEHAVIOR (FOCUS)")
    print(long_msg)
    print(short_msg)
    print(
        "\nInterpretation: this tells you if today's weekday has historically "
        "been slightly bullish or bearish intraday and overnight, both long-term and in the last year."
    )


# =========================
# MAIN
# =========================
def main():
    df = download_data()

    latest, regime_flags = rolling_relative_position(df)

    print_header(f"{TICKER} DAILY STATISTICAL DASHBOARD")
    print(f"Last date in data:  {df.index[-1].date()}")
    print(f"Run date (system):  {datetime.today().date()}")
    print(f"Last Adj Close:     {latest['Adj Close']:.2f}")
    print(f"20DMA / 50DMA / 200DMA: {latest['ma20']:.2f} / {latest['ma50']:.2f} / {latest['ma200']:.2f}")
    print(f"Trend structure:    {', '.join(regime_flags)}")

    print_header("MULTI-HORIZON SNAPSHOT")
    stats_list = []
    for w in SHORT_WINDOWS + LONG_WINDOWS:
        s = window_stats(df, w)
        if s:
            stats_list.append(s)
    print_window_table(stats_list)

    print_header("WEEKDAY EFFECTS")
    wd_6m = weekday_stats(df, window=126)
    wd_12m = weekday_stats(df, window=252)
    wd_all = weekday_stats(df, window=min(len(df), 252*10))
    print_weekday_stats("Last 6 months (~126 trading days)", wd_6m)
    print_weekday_stats("Last 12 months (~252 trading days)", wd_12m)
    print_weekday_stats("Long sample", wd_all)

    # Get today's weekday summary using calendar weekday
    weekday_name, long_msg, short_msg = get_today_weekday_summary(df, wd_all, wd_12m)
    print_today_weekday_behavior_block(weekday_name, long_msg, short_msg)

    print_header("MONTH EFFECTS")
    ms = month_stats(df, window=min(len(df), 252*10))
    print_df_pct(ms, ["avg_close_close", "avg_intraday", "avg_overnight"])

    print_header("VOLATILITY REGIME STUDY")
    vol_summary, vol_latest = volatility_regime_stats(df)
    if vol_summary is not None:
        print_df_pct(vol_summary, ["avg_ret", "avg_intraday", "avg_overnight", "std_ret", "pct_up"])
        print(f"\nLatest realized vol20: {vol_latest['rv20']*100:.2f}%")
        print(f"Current vol regime:    {vol_latest['vol_bucket']}")
    else:
        print("Not enough data for volatility regime study.")

    print_header("AUTOCORRELATION")
    ac = autocorr_stats(df, window=252)
    print(ac.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print_header("MEAN REVERSION / TREND BY Z-SCORE")
    zr = zscore_reversion_stats(df, ma_windows=(20, 50), forward_days=(1, 5, 10), lookback=min(len(df), 252*7))
    zr_show = zr[zr["obs"] >= 20].copy()
    print_df_pct(zr_show, ["avg_fwd_return", "median_fwd_return", "pct_up"])

    print_header("HMM REGIME DETECTION")
    hmm_out, hmm_err = hmm_regime_analysis(df)
    if hmm_out is None:
        print(f"HMM unavailable: {hmm_err}")
    else:
        print(f"Current HMM regime: {hmm_out['current_regime']} (state {hmm_out['current_state']})")
        print("\nState summary:")
        print_df_pct(hmm_out["state_summary"], ["mean_ret", "mean_intraday", "mean_overnight", "mean_vol20"])
        print("\nRecent 63-day regime mix:")
        print(hmm_out["recent_regime_mix_63d"].map(lambda x: f"{x*100:.2f}%").to_string())
        print("\nLast 10 regime labels:")
        print(hmm_out["last_10_regimes"].to_string())

    print_header("CHANGE-POINT / REGIME SHIFT STUDY")
    cp_out, cp_err = changepoint_analysis(df)
    if cp_out is None:
        print(f"Change-point unavailable: {cp_err}")
    else:
        print(f"Lookback days: {cp_out['lookback_days']}")
        print(f"Detected break count: {cp_out['break_count']}")
        print("Recent break dates:")
        if cp_out["break_dates"]:
            for d in cp_out["break_dates"]:
                print(f" - {d}")
        else:
            print(" - none")

    print_header("MONTE CARLO 20-DAY PRICE SCENARIOS (HISTORICAL BOOTSTRAP)")
    mc = monte_carlo_simulation(df)
    if mc is None:
        print("Not enough data for Monte Carlo.")
    else:
        print(
            f"Using {mc['n_paths']} simulated paths over {mc['horizon']} days, "
            f"based on historical daily returns."
        )
        print(
            f"Last price: {mc['last_price']:.2f}  | "
            f"Mean 20d return: {fmt_pct(mc['mean_return'])}  "
            f"(median: {fmt_pct(mc['median_return'])})"
        )
        print(
            f"20d return percentiles: "
            f"5%={fmt_pct(mc['pct5'])}, 25%={fmt_pct(mc['pct25'])}, "
            f"75%={fmt_pct(mc['pct75'])}, 95%={fmt_pct(mc['pct95'])}"
        )
        print(
            f"Implied 20d price range (approx): "
            f"{mc['pct5_price']:.2f} (5% worst) – {mc['pct95_price']:.2f} (5% best)"
        )

    # High-level colored summary
    print_colored_summary(df, stats_list, hmm_out, vol_latest, wd_6m, zr)

    # FINAL BRIGHT COLORED WEEKDAY SNAPSHOT
    print("\n" + "=" * 90)
    print(GREEN + BOLD + f"TODAY'S WEEKDAY SNAPSHOT: {weekday_name.upper()}" + RESET)
    print(GREEN + long_msg + RESET)
    print(YELLOW + short_msg + RESET)
    print(
        CYAN
        + "Quick read: green line = full history behavior for this weekday, "
          "yellow line = last 12 months behavior."
        + RESET
    )


if __name__ == "__main__":
    main()
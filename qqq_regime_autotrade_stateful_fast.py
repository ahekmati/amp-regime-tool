#!/usr/bin/env python3
import json
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except Exception:
    raise RuntimeError("Missing dependency: yfinance. Install with: pip install yfinance")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
except Exception:
    raise RuntimeError("Missing scikit-learn. Install with: pip install scikit-learn")

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
C2_API4_BASE = "https://api4-general.collective2.com"
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "20"))
LOOKBACK_YEARS = int(os.getenv("LOOKBACK_YEARS", "3"))
MC_SIMS = int(os.getenv("MC_SIMS", "1200"))

AUTO_TRADE_ENABLED = os.getenv("AUTO_TRADE_ENABLED", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
# DRY_RUN True = simulate only, default on for safety
DRY_RUN = os.getenv("DRY_RUN", "1").strip() == "1"

TICKER = os.getenv("TICKER", "QQQ")
MNQ_SYMBOL = os.getenv("MNQ_SYMBOL", "@MNQM6")
MNQ_QTY = int(os.getenv("MNQ_QTY", "1"))

STATE_FILE = Path(os.getenv("STATE_FILE", "./state/qqq_regime_state.json"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", "./cache"))
CACHE_TTL_MINUTES = int(os.getenv("CACHE_TTL_MINUTES", "180"))

ETF_FALLBACKS = {
    "QQQ": ["QQQ", "QQQM"],
    "SPY": ["SPY", "IVV", "VOO"],
}

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

MODEL_WEIGHTS = {
    "Trend 50/200": 1.2,
    "Price vs SMA200": 1.1,
    "Price vs SMA50": 0.9,
    "63/126 Momentum": 1.2,
    "MACD Regime": 0.8,
    "RSI Swing": 0.5,
    "21-day Return": 0.6,
    "Vol-adjusted Momentum": 0.9,
    "Logistic ML 21d": 1.2,
    "Random Forest Risk": 1.1,
    "Monte Carlo 63d": 0.5,
    "Extension Filter": 0.8,
}


# ---------------------------------------------------------------------
# UTILS / STATE
# ---------------------------------------------------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_state() -> Dict[str, Any] | None:
    if not STATE_FILE.exists():
        return None
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_state(data: Dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def transition_name(prev_regime: str, curr_regime: str) -> str:
    return f"{prev_regime}_to_{curr_regime}"


# ---------------------------------------------------------------------
# C2 API
# ---------------------------------------------------------------------
def api4_post(path: str, apikey: str, payload: dict) -> dict:
    url = f"{C2_API4_BASE}{path}"
    headers = {"Authorization": f"Bearer {apikey}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def build_market_order(strategy_id: int, full_symbol: str, side: str, qty: int) -> dict:
    """
    Simple market order (no bracket):
      side="long"  -> buy/open long
      side="short" -> sell/open short
    """
    c2_side = "1" if side == "long" else "2"
    return {
        "Order": {
            "StrategyId": strategy_id,
            "OrderType": "1",  # market
            "Side": c2_side,
            "OrderQuantity": int(qty),
            "TIF": "0",
            "C2Symbol": {
                "FullSymbol": full_symbol,
                "SymbolType": "future",
            },
        }
    }


# ---------------------------------------------------------------------
# MARKET DATA / FEATURES
# ---------------------------------------------------------------------
def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in df.columns for col in required):
        return pd.DataFrame()
    return df[required].copy().dropna()


def _cache_path(ticker: str, years: int) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{ticker.lower()}_{years}y.csv"


def _load_cached_market_data(ticker: str, years: int) -> pd.DataFrame:
    path = _cache_path(ticker, years)
    if not path.exists():
        return pd.DataFrame()
    age_seconds = time.time() - path.stat().st_mtime
    if age_seconds > CACHE_TTL_MINUTES * 60:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return _normalize_ohlcv(df)
    except Exception:
        return pd.DataFrame()


def _save_cached_market_data(ticker: str, years: int, df: pd.DataFrame) -> None:
    try:
        _normalize_ohlcv(df).to_csv(_cache_path(ticker, years))
    except Exception:
        pass


def _download_once(symbol: str, years: int) -> pd.DataFrame:
    df = yf.download(
        symbol,
        period=f"{years}y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    return _normalize_ohlcv(df)


def _history_once(symbol: str, years: int) -> pd.DataFrame:
    df = yf.Ticker(symbol).history(
        period=f"{years}y",
        interval="1d",
        auto_adjust=True,
    )
    return _normalize_ohlcv(df)


def download_market_data(ticker: str, years: int = LOOKBACK_YEARS) -> pd.DataFrame:
    cached = _load_cached_market_data(ticker, years)
    if not cached.empty:
        return cached

    candidates = ETF_FALLBACKS.get(ticker, [ticker])
    last_error = None
    for candidate in candidates:
        try:
            df = _download_once(candidate, years)
            if not df.empty:
                _save_cached_market_data(ticker, years, df)
                return df
        except Exception as e:
            last_error = e
        try:
            df = _history_once(candidate, years)
            if not df.empty:
                _save_cached_market_data(ticker, years, df)
                return df
        except Exception as e:
            last_error = e
    raise RuntimeError(f"No data downloaded for {ticker}. Last error: {last_error}")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ret_1d"] = d["Close"].pct_change()
    d["ret_5d"] = d["Close"].pct_change(5)
    d["ret_21d"] = d["Close"].pct_change(21)
    d["ret_63d"] = d["Close"].pct_change(63)
    d["ret_126d"] = d["Close"].pct_change(126)
    d["sma_50"] = d["Close"].rolling(50).mean()
    d["sma_200"] = d["Close"].rolling(200).mean()
    d["ema_12"] = d["Close"].ewm(span=12, adjust=False).mean()
    d["ema_26"] = d["Close"].ewm(span=26, adjust=False).mean()
    d["macd"] = d["ema_12"] - d["ema_26"]
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["dist_sma50"] = d["Close"] / d["sma_50"] - 1
    d["dist_sma200"] = d["Close"] / d["sma_200"] - 1
    d["sma50_sma200"] = d["sma_50"] / d["sma_200"] - 1

    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi_14"] = 100 - (100 / (1 + rs))

    tr = pd.concat(
        [
            d["High"] - d["Low"],
            (d["High"] - d["Close"].shift()).abs(),
            (d["Low"] - d["Close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    d["atr_14"] = tr.rolling(14).mean()
    d["vol_21"] = d["ret_1d"].rolling(21).std() * np.sqrt(252)
    d["vol_63"] = d["ret_1d"].rolling(63).std() * np.sqrt(252)
    d["range_pct"] = (d["High"] - d["Low"]) / d["Close"]
    vol_mean = d["Volume"].rolling(63).mean()
    vol_std = d["Volume"].rolling(63).std()
    d["volume_z"] = (d["Volume"] - vol_mean) / vol_std
    d["target_21d_up"] = (d["Close"].shift(-21) / d["Close"] - 1 > 0).astype(int)
    return d


# ---------------------------------------------------------------------
# SIGNALS / REGIME
# ---------------------------------------------------------------------
def technical_signals(d: pd.DataFrame) -> List[Dict[str, Any]]:
    last = d.dropna().iloc[-1]
    vm = float(last["ret_63d"] / max(last["vol_21"], 1e-9))
    return [
        {
            "analysis": "Trend 50/200",
            "signal": "bullish" if last["sma_50"] > last["sma_200"] else "bearish",
            "value": round(float(last["sma50_sma200"] * 100), 2),
            "note": "Golden/death cross regime",
        },
        {
            "analysis": "Price vs SMA200",
            "signal": "bullish" if last["Close"] > last["sma_200"] else "bearish",
            "value": round(float(last["dist_sma200"] * 100), 2),
            "note": "Percent above/below 200-day average",
        },
        {
            "analysis": "Price vs SMA50",
            "signal": "bullish" if last["Close"] > last["sma_50"] else "bearish",
            "value": round(float(last["dist_sma50"] * 100), 2),
            "note": "Percent above/below 50-day average",
        },
        {
            "analysis": "63/126 Momentum",
            "signal": "bullish"
            if last["ret_63d"] > 0 and last["ret_126d"] > 0
            else "bearish"
            if last["ret_63d"] < 0 and last["ret_126d"] < 0
            else "neutral",
            "value": round(float(last["ret_63d"] * 100), 2),
            "note": "Quarter and half-year momentum",
        },
        {
            "analysis": "MACD Regime",
            "signal": "bullish"
            if last["macd"] > last["macd_signal"] and last["macd"] > 0
            else "bearish"
            if last["macd"] < last["macd_signal"] and last["macd"] < 0
            else "neutral",
            "value": round(float(last["macd"]), 4),
            "note": "MACD vs signal and zero line",
        },
        {
            "analysis": "RSI Swing",
            "signal": "bullish"
            if 50 < last["rsi_14"] < 68
            else "bearish"
            if 32 < last["rsi_14"] < 50
            else "neutral",
            "value": round(float(last["rsi_14"]), 2),
            "note": "RSI trend zone with extension awareness",
        },
        {
            "analysis": "21-day Return",
            "signal": "bullish" if last["ret_21d"] > 0 else "bearish",
            "value": round(float(last["ret_21d"] * 100), 2),
            "note": "One-month return sign",
        },
        {
            "analysis": "Vol-adjusted Momentum",
            "signal": "bullish"
            if vm > 0.4
            else "bearish"
            if vm < -0.4
            else "neutral",
            "value": round(vm, 3),
            "note": "Momentum divided by annualized vol",
        },
    ]


def logistic_21d_signal(d: pd.DataFrame) -> Tuple[str, str, float]:
    cols = [
        "ret_5d",
        "ret_21d",
        "ret_63d",
        "ret_126d",
        "dist_sma200",
        "sma50_sma200",
        "rsi_14",
        "macd",
        "macd_signal",
        "vol_21",
    ]
    x = d[cols].replace([np.inf, -np.inf], np.nan).dropna()
    y = d.loc[x.index, "target_21d_up"]
    if len(x) < 280:
        return "neutral", "insufficient data", 0.0
    split = int(len(x) * 0.85)
    model = LogisticRegression(max_iter=800, solver="lbfgs")
    model.fit(x.iloc[:split], y.iloc[:split])
    prob = float(model.predict_proba(x.iloc[[-1]])[0, 1])
    signal = "bullish" if prob >= 0.55 else "bearish" if prob <= 0.45 else "neutral"
    conf = abs(prob - 0.5) * 2
    return signal, f"p_up_21d={prob:.3f}", conf


def random_forest_risk_signal(d: pd.DataFrame) -> Tuple[str, str, float]:
    future_dd = d["Close"].shift(-21).rolling(21).min() / d["Close"] - 1
    risk_label = (future_dd <= -0.05).astype(int)
    cols = [
        "ret_5d",
        "ret_21d",
        "ret_63d",
        "dist_sma50",
        "dist_sma200",
        "rsi_14",
        "macd",
        "vol_21",
        "vol_63",
        "atr_14",
        "range_pct",
        "volume_z",
    ]
    x = d[cols].replace([np.inf, -np.inf], np.nan).dropna()
    y = risk_label.loc[x.index]
    valid = y.notna()
    x, y = x.loc[valid], y.loc[valid]
    if len(x) < 280:
        return "neutral", "insufficient data", 0.0
    split = int(len(x) * 0.85)
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(x.iloc[:split], y.iloc[:split])
    p_risk = float(model.predict_proba(x.iloc[[-1]])[0, 1])
    signal = "bearish" if p_risk >= 0.55 else "bullish" if p_risk <= 0.35 else "neutral"
    conf = abs(p_risk - 0.5) * 2
    return signal, f"crash_risk_21d={p_risk:.3f}", conf


def monte_carlo_signal(d: pd.DataFrame, horizon: int = 63, sims: int = MC_SIMS) -> Tuple[str, str, float]:
    rets = d["ret_1d"].dropna().tail(252 * 2)
    if len(rets) < 160:
        return "neutral", "insufficient data", 0.0
    mu = rets.mean()
    sigma = rets.std()
    rand = np.random.normal(mu, sigma, size=(sims, horizon))
    paths = np.exp(np.log1p(rand).sum(axis=1)) - 1
    prob_up = float((paths > 0).mean())
    median_ret = float(np.median(paths))
    signal = "bullish" if prob_up >= 0.56 and median_ret > 0 else "bearish" if prob_up <= 0.44 and median_ret < 0 else "neutral"
    conf = abs(prob_up - 0.5) * 2
    return signal, f"p_up_63d={prob_up:.3f}, med={median_ret:.3%}", conf


def extension_filter(d: pd.DataFrame) -> Tuple[str, str]:
    last = d.dropna().iloc[-1]
    overextended_up = last["rsi_14"] >= 72 and last["dist_sma50"] >= 0.045
    overextended_dn = last["rsi_14"] <= 30 and last["dist_sma50"] <= -0.045
    if overextended_up:
        return "neutral", f"bullish trend but extended (RSI={last['rsi_14']:.1f}, dist50={last['dist_sma50']:.2%})"
    if overextended_dn:
        return "neutral", f"bearish trend but stretched down (RSI={last['rsi_14']:.1f}, dist50={last['dist_sma50']:.2%})"
    return ("bullish" if last["Close"] > last["sma_50"] else "bearish"), "extension filter clear"


def weighted_score(df: pd.DataFrame) -> Tuple[float, float, str]:
    score = 0.0
    max_score = 0.0
    for _, row in df.iterrows():
        w = MODEL_WEIGHTS.get(row["analysis"], 1.0)
        max_score += abs(w)
        if row["signal"] == "bullish":
            score += w
        elif row["signal"] == "bearish":
            score -= w
    conf = abs(score) / max(max_score, 1e-9)
    final = "bullish" if score >= 2.5 else "bearish" if score <= -2.5 else "neutral"
    return score, conf, final


def compute_today_composite(d: pd.DataFrame) -> Dict[str, Any]:
    analyses = technical_signals(d)
    log_sig, log_note, log_conf = logistic_21d_signal(d)
    rf_sig, rf_note, rf_conf = random_forest_risk_signal(d)
    mc_sig, mc_note, mc_conf = monte_carlo_signal(d)
    ext_sig, ext_note = extension_filter(d)

    analyses.extend(
        [
            {
                "analysis": "Logistic ML 21d",
                "signal": log_sig,
                "value": round(log_conf, 3),
                "note": log_note,
            },
            {
                "analysis": "Random Forest Risk",
                "signal": rf_sig,
                "value": round(rf_conf, 3),
                "note": rf_note,
            },
            {
                "analysis": "Monte Carlo 63d",
                "signal": mc_sig,
                "value": round(mc_conf, 3),
                "note": mc_note,
            },
            {
                "analysis": "Extension Filter",
                "signal": ext_sig,
                "value": None,
                "note": ext_note,
            },
        ]
    )

    out = pd.DataFrame(analyses)
    score, conf, final = weighted_score(out)
    last_idx = d.index[-1]
    return {
        "date": last_idx,
        "close": float(d["Close"].iloc[-1]),
        "score": float(score),
        "confidence": float(conf),
        "regime": final,
        "analyses": analyses,
    }


# ---------------------------------------------------------------------
# MAIN (REGIME → MARKET ORDERS ONLY)
# ---------------------------------------------------------------------
def main() -> None:
    print(f"=== QQQ regime autotrade FAST at {utc_now_iso()} ===")
    apikey = os.getenv("C2_API_KEY", "").strip()
    systemid_raw = os.getenv("C2_SYSTEM_ID", "").strip()

    if not apikey or not systemid_raw:
        print(RED + "Missing C2_API_KEY or C2_SYSTEM_ID." + RESET)
        return

    try:
        strategy_id = int(systemid_raw)
    except Exception:
        print(RED + f"Invalid C2_SYSTEM_ID: {systemid_raw}" + RESET)
        return

    # Download & features
    try:
        df = download_market_data(TICKER)
    except Exception as e:
        print(RED + f"Market data download failed: {e}" + RESET)
        return

    d = engineer_features(df).dropna().copy()
    if len(d) < 260:
        print(RED + f"Not enough data after feature engineering: {len(d)} rows" + RESET)
        return

    today_info = compute_today_composite(d)
    today_regime = str(today_info["regime"])
    trans_date = str(today_info["date"].date())

    print(
        CYAN
        + f"Current: {today_regime.upper()} | close={today_info['close']:.2f} "
        f"score={today_info['score']:.2f} conf={today_info['confidence']:.1%}"
        + RESET
    )

    state = load_state()
    if state is None:
        save_state(
            {
                "armed_on_utc": utc_now_iso(),
                "current_regime": today_regime,
                "last_seen_transition": None,
                "last_seen_transition_date": None,
                "last_processed_transition": None,
                "last_processed_transition_date": None,
                "last_run_utc": utc_now_iso(),
                "last_close": today_info["close"],
                "last_score": today_info["score"],
            }
        )
        print(YELLOW + f"State initialized at {STATE_FILE}. No trade on first run." + RESET)
        return

    prev_regime = str(state.get("current_regime", "neutral"))
    trans = transition_name(prev_regime, today_regime)
    print(
        CYAN
        + f"Previous state regime: {prev_regime.upper()} -> Current regime: {today_regime.upper()}"
        + RESET
    )

    # Only trade when regime actually changes
    if prev_regime == today_regime:
        state.update(
            {
                "current_regime": today_regime,
                "last_seen_transition": trans,
                "last_seen_transition_date": trans_date,
                "last_run_utc": utc_now_iso(),
                "last_close": today_info["close"],
                "last_score": today_info["score"],
            }
        )
        save_state(state)
        print(YELLOW + "No regime change since prior run. No trade." + RESET)
        return

    # We have a regime change: decide direction
    # bullish -> long, bearish -> short; neutral transitions are ignored
    if today_regime == "bullish" and prev_regime in {"bearish", "neutral"}:
        trade_side = "long"
    elif today_regime == "bearish" and prev_regime in {"bullish", "neutral"}:
        trade_side = "short"
    else:
        trade_side = None

    if trade_side is None:
        state.update(
            {
                "current_regime": today_regime,
                "last_seen_transition": trans,
                "last_seen_transition_date": trans_date,
                "last_run_utc": utc_now_iso(),
                "last_close": today_info["close"],
                "last_score": today_info["score"],
            }
        )
        save_state(state)
        print(YELLOW + f"Transition {trans} not mapped to a trade direction. No trade." + RESET)
        return

    # Prevent double-processing same transition/date
    already_processed = (
        state.get("last_processed_transition") == trans
        and state.get("last_processed_transition_date") == trans_date
    )
    if already_processed:
        state.update(
            {
                "current_regime": today_regime,
                "last_seen_transition": trans,
                "last_seen_transition_date": trans_date,
                "last_run_utc": utc_now_iso(),
                "last_close": today_info["close"],
                "last_score": today_info["score"],
            }
        )
        save_state(state)
        print(YELLOW + f"This transition {trans} was already processed earlier today. No trade." + RESET)
        return

    print(
        GREEN
        + BOLD
        + f"New regime change detected: {prev_regime.upper()} -> {today_regime.upper()} | ACTION: {trade_side.upper()} MNQ"
        + RESET
    )

    payload = build_market_order(
        strategy_id=strategy_id,
        full_symbol=MNQ_SYMBOL,
        side=trade_side,
        qty=MNQ_QTY,
    )
    print("Market order payload:", json.dumps(payload, ensure_ascii=False))

    if not AUTO_TRADE_ENABLED:
        print(YELLOW + "AUTO_TRADE_ENABLED is false. No order sent." + RESET)
        return
    if DRY_RUN:
        print(YELLOW + "DRY_RUN=1 -> order NOT sent. (Simulation only.)" + RESET)
        return

    try:
        result = api4_post("/Strategies/NewStrategyOrder", apikey, payload)
        print("C2 result:", json.dumps(result, ensure_ascii=False))
        state.update(
            {
                "current_regime": today_regime,
                "last_seen_transition": trans,
                "last_seen_transition_date": trans_date,
                "last_processed_transition": trans,
                "last_processed_transition_date": trans_date,
                "last_run_utc": utc_now_iso(),
                "last_close": today_info["close"],
                "last_score": today_info["score"],
            }
        )
        save_state(state)
        print(GREEN + BOLD + "Order sent and state updated." + RESET)
    except Exception as e:
        print(RED + f"C2 order submit failed: {e}" + RESET)


if __name__ == "__main__":
    main()

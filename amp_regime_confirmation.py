#!/usr/bin/env python3
import csv
import json
import os
import re
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone, date
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings('ignore')

try:
    import yfinance as yf
except Exception:
    raise RuntimeError('Missing dependency: yfinance. Install with: pip install yfinance')

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
except Exception:
    raise RuntimeError('Missing scikit-learn. Install with: pip install scikit-learn')

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except Exception:
    HMM_AVAILABLE = False

try:
    import statsmodels.api as sm
    MARKOV_AVAILABLE = True
except Exception:
    MARKOV_AVAILABLE = False

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
AMP_URL = "https://ampfutures.isystems.com/Systems/TopStrategies"
C2_API4_BASE = "https://api4-general.collective2.com"
REQUEST_TIMEOUT = 30

TOP_N = int(os.getenv("TOP_N", "10"))
LOOKBACK_YEARS = int(os.getenv("LOOKBACK_YEARS", "8"))
MC_SIMS = int(os.getenv("MC_SIMS", "6000"))
PERSISTENCE_DAYS = int(os.getenv("PERSISTENCE_DAYS", "3"))

AUTO_TRADE_ENABLED = os.getenv("AUTO_TRADE_ENABLED", "false").lower() == "true"
DRY_RUN = os.getenv("DRY_RUN", "1") == "1"
AGREEMENT_THRESHOLD = float(os.getenv("AGREEMENT_THRESHOLD", "0.75"))

HISTORY_DIR = Path(os.getenv("HISTORY_DIR", "./history"))
HISTORY_FILE = HISTORY_DIR / "regime_runs.csv"

MES_SYMBOL = "@MESM6"
MNQ_SYMBOL = "@MNQM6"
SUPPORTED_PRODUCTS = {"ES": MES_SYMBOL, "NQ": MNQ_SYMBOL}
ETF_MAP = {"ES": "SPY", "NQ": "QQQ"}

RISK_ON_SET = ["SPY", "QQQ", "IWM", "HYG"]
RISK_OFF_SET = ["TLT", "GLD"]
STRESS_SET = ["VIXY"]
ALL_TICKERS = list(dict.fromkeys(RISK_ON_SET + RISK_OFF_SET + STRESS_SET))

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"

MODEL_WEIGHTS = {
    "Trend 50/200": 1.1,
    "Price vs SMA200": 1.0,
    "Price vs SMA50": 0.8,
    "63/126 Momentum": 1.2,
    "MACD Regime": 0.9,
    "RSI Swing": 0.5,
    "21-day Return": 0.6,
    "Vol-adjusted Momentum": 0.9,
    "Gaussian HMM": 1.5,
    "Markov Switching": 1.4,
    "Gaussian Mixture": 0.9,
    "KMeans Regime": 0.7,
    "Logistic ML 21d": 1.2,
    "Random Forest Risk": 1.3,
    "Monte Carlo 63d": 0.8,
    "Breadth Risk-On": 1.2,
    "Cross-Asset Macro": 1.4,
    "Extension Filter": 0.7,
}

# ---------------------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------------------
@dataclass
class ScrapedRow:
    rank: int
    system: str
    product: str
    pnl: float
    current_position: str
    nearest_order: str
    developer: str = ""

@dataclass
class ParsedPosition:
    side: str
    qty: int

@dataclass
class OpenPosition:
    symbol: str
    side: str
    qty: int
    entry_price: Optional[float]
    raw: dict

# ---------------------------------------------------------------------
# UTILS & PRINT HELPERS
# ---------------------------------------------------------------------
def color_for_signal(signal: str) -> str:
    s = signal.lower()
    if s == "bullish":
        return GREEN
    if s == "bearish":
        return RED
    return YELLOW

def print_block(title: str, lines: List[str], color: str = GREEN):
    print(color + BOLD + f"\n{title}" + RESET)
    for line in lines:
        print(color + line + RESET)

def add_analysis(results: List[Dict[str, Any]], name: str, signal: str, value: Any, note: str):
    results.append({"analysis": name, "signal": signal, "value": value, "note": note})

def days_since(date_str: str) -> Optional[int]:
    if not date_str:
        return None
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (date.today() - d).days
    except Exception:
        return None

def compute_agreement(summary: dict) -> Tuple[float, str]:
    raw_score = summary["raw_score"]
    max_score = sum(abs(w) for w in MODEL_WEIGHTS.values())
    if max_score <= 0:
        return 0.0, "neutral"
    pct = raw_score / max_score
    direction = "bullish" if pct >= 0 else "bearish"
    return abs(pct), direction

# ---------------------------------------------------------------------
# AMP SCRAPER
# ---------------------------------------------------------------------
def fetch_amp_html() -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    r = requests.get(AMP_URL, headers=headers, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.text

def money_to_float(s: str) -> Optional[float]:
    s = s.replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except Exception:
        return None

def parse_current_session(html: str) -> List[ScrapedRow]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="tableCurrentSession")
    if not table:
        raise RuntimeError("Could not find tableCurrentSession in AMP HTML")
    body = table.find("tbody")
    if not body:
        raise RuntimeError("tableCurrentSession has no tbody")

    rows: List[ScrapedRow] = []
    for tr in body.find_all("tr"):
        rank_td = None
        system_td = tr.find("td", id=re.compile(r"^rankID_"))
        product_td = tr.find("td", id=re.compile(r"^rankProduct_"))
        developer_td = tr.find("td", id=re.compile(r"^rankDeveloperName_"))
        pnl_td = tr.find("td", id=re.compile(r"^rankNetResult_"))
        pos_td = tr.find("td", id=re.compile(r"^rankCurrentPosition_"))
        nearest_td = tr.find("td", id=re.compile(r"^rankClosestOrder_"))
        all_tds = tr.find_all("td")

        if len(all_tds) >= 2:
            rank_td = all_tds[1]

        if not all([rank_td, system_td, product_td, pnl_td, pos_td, nearest_td]):
            continue

        m = re.search(r"#(\d+)", rank_td.get_text(" ", strip=True))
        if not m:
            continue

        rank = int(m.group(1))
        if rank > TOP_N:
            continue

        pnl = money_to_float(pnl_td.get_text(" ", strip=True))
        if pnl is None:
            continue

        rows.append(
            ScrapedRow(
                rank=rank,
                system=system_td.get_text(" ", strip=True),
                product=product_td.get_text(" ", strip=True).upper(),
                pnl=pnl,
                current_position=pos_td.get_text(" ", strip=True),
                nearest_order=nearest_td.get_text(" ", strip=True),
                developer=developer_td.get_text(" ", strip=True) if developer_td else "",
            )
        )
    return rows

def pick_best_supported(rows: List[ScrapedRow]) -> Optional[ScrapedRow]:
    supported = [r for r in rows if r.product in SUPPORTED_PRODUCTS]
    if not supported:
        return None
    supported.sort(key=lambda x: (x.rank, -x.pnl))
    return supported[0]

def parse_direction_and_size(text: str) -> Optional[ParsedPosition]:
    text = text.strip()
    # Treat flat / empty as no position
    if text in {"", "--", "-", "Flat", "FLAT", "flat"}:
        return None

    m = re.match(r"^(Long|Short)\s+(\d+)\s*@", text, flags=re.I)
    if not m:
        return None

    return ParsedPosition(side=m.group(1).lower(), qty=int(m.group(2)))

# ---------------------------------------------------------------------
# C2 API
# ---------------------------------------------------------------------
def api4_get(path: str, apikey: str, params: Dict[str, Any]) -> dict:
    url = f"{C2_API4_BASE}{path}"
    headers = {"Authorization": f"Bearer {apikey}", "Content-Type": "application/json"}
    r = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()

def api4_post(path: str, apikey: str, payload: dict) -> dict:
    url = f"{C2_API4_BASE}{path}"
    headers = {"Authorization": f"Bearer {apikey}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()

def get_open_positions(apikey: str, strategy_id: int) -> dict:
    return api4_get("/Strategies/GetStrategyOpenPositions", apikey, {"StrategyIds": str(strategy_id)})

def extract_supported_open_positions(open_positions: dict) -> List[OpenPosition]:
    results = open_positions.get("Results", [])
    extracted: List[OpenPosition] = []
    supported_symbols = set(SUPPORTED_PRODUCTS.values())

    for p in results:
        sym = p.get("C2Symbol", {}).get("FullSymbol")
        qty = p.get("Quantity")
        if sym not in supported_symbols:
            continue
        if not qty or qty == 0:
            continue

        side = "long" if qty > 0 else "short"
        entry_price = p.get("AvgPx") or p.get("AvgEntryPrice") or p.get("EntryPrice")
        extracted.append(OpenPosition(symbol=sym, side=side, qty=abs(int(qty)), entry_price=entry_price, raw=p))

    return extracted

def build_market_order(strategy_id: int, symbol: str, side: str, qty: int) -> dict:
    side_code = "1" if side == "long" else "2"
    return {
        "Order": {
            "StrategyId": strategy_id,
            "OrderType": "1",
            "Side": side_code,
            "OrderQuantity": qty,
            "TIF": "0",
            "C2Symbol": {
                "FullSymbol": symbol,
                "SymbolType": "future"
            }
        }
    }

def build_close_order(strategy_id: int, open_pos: OpenPosition) -> dict:
    close_side = "short" if open_pos.side == "long" else "long"
    return build_market_order(strategy_id, open_pos.symbol, close_side, open_pos.qty)

# ---------------------------------------------------------------------
# MARKET DATA & FEATURE ENGINEERING
# ---------------------------------------------------------------------
def download_market_data(ticker: str, years: int = LOOKBACK_YEARS) -> pd.DataFrame:
    df = yf.download(ticker, period=f"{years}y", interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data downloaded for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ret_1d"] = d["Close"].pct_change()
    d["ret_5d"] = d["Close"].pct_change(5)
    d["ret_21d"] = d["Close"].pct_change(21)
    d["ret_63d"] = d["Close"].pct_change(63)
    d["ret_126d"] = d["Close"].pct_change(126)

    d["sma_20"] = d["Close"].rolling(20).mean()
    d["sma_50"] = d["Close"].rolling(50).mean()
    d["sma_100"] = d["Close"].rolling(100).mean()
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
    d["volume_z"] = (d["Volume"] - d["Volume"].rolling(63).mean()) / d["Volume"].rolling(63).std()

    d["target_21d_up"] = (d["Close"].shift(-21) / d["Close"] - 1 > 0).astype(int)
    return d

# ---------------------------------------------------------------------
# REGIME MODELS / SIGNALS
# ---------------------------------------------------------------------
def classify_cluster_regime(cluster_df: pd.DataFrame, label_col: str) -> Dict[int, str]:
    stats = cluster_df.groupby(label_col).agg(avg_ret63=("ret_63d", "mean"), avg_vol=("vol_21", "mean"))
    regime_map: Dict[int, str] = {}
    if stats.empty:
        return regime_map
    for idx, row in stats.iterrows():
        if row["avg_ret63"] > 0 and row["avg_vol"] <= stats["avg_vol"].median():
            regime_map[idx] = "bullish"
        elif row["avg_ret63"] < 0 and row["avg_vol"] >= stats["avg_vol"].median():
            regime_map[idx] = "bearish"
        else:
            regime_map[idx] = "neutral"
    return regime_map

def hmm_regime_signal(d: pd.DataFrame) -> Tuple[str, str, float]:
    if not HMM_AVAILABLE:
        return "neutral", "hmmlearn not installed", 0.0
    feat = d[["ret_1d", "vol_21"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(feat) < 300:
        return "neutral", "insufficient data", 0.0
    scaler = StandardScaler()
    X = scaler.fit_transform(feat.values)
    model = GaussianHMM(n_components=3, covariance_type="full", n_iter=300, random_state=42)
    model.fit(X)
    states = model.predict(X)
    tmp = feat.copy()
    tmp["state"] = states
    tmp["ret_63d"] = d.loc[tmp.index, "ret_63d"]
    grp = tmp.groupby("state").agg(avg_ret=("ret_63d", "mean"), avg_vol=("vol_21", "mean"))
    mapping = {}
    for st, row in grp.iterrows():
        if row["avg_ret"] > 0 and row["avg_vol"] <= grp["avg_vol"].median():
            mapping[st] = "bullish"
        elif row["avg_ret"] < 0 and row["avg_vol"] >= grp["avg_vol"].median():
            mapping[st] = "bearish"
        else:
            mapping[st] = "neutral"
    current_state = int(states[-1])
    probs = model.predict_proba(X)
    conf = float(np.max(probs[-1]))
    return mapping.get(current_state, "neutral"), f"state={current_state}, conf={conf:.3f}", conf

def markov_switch_signal(d: pd.DataFrame) -> Tuple[str, str, float]:
    if not MARKOV_AVAILABLE:
        return "neutral", "statsmodels not installed", 0.0
    series = d["ret_1d"].dropna()
    if len(series) < 300:
        return "neutral", "insufficient data", 0.0
    try:
        model = sm.tsa.MarkovRegression(series, k_regimes=2, trend='c', switching_variance=True)
        res = model.fit(disp=False)
        probs = res.filtered_marginal_probabilities
        means = []
        for regime in probs.columns:
            w = probs[regime]
            means.append((regime, float((w * series.loc[w.index]).sum() / w.sum())))
        bull_regime = max(means, key=lambda x: x[1])[0]
        bull_prob = float(probs[bull_regime].iloc[-1])
        signal = "bullish" if bull_prob >= 0.55 else "bearish" if bull_prob <= 0.45 else "neutral"
        conf = abs(bull_prob - 0.5) * 2
        return signal, f"bull_prob={bull_prob:.3f}", conf
    except Exception as e:
        return "neutral", f"fit_failed={str(e)[:80]}", 0.0

def gmm_regime_signal(d: pd.DataFrame) -> Tuple[str, str, float]:
    cols = ["ret_21d", "ret_63d", "vol_21", "dist_sma200", "rsi_14"]
    feat = d[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(feat) < 250:
        return "neutral", "insufficient data", 0.0
    scaler = StandardScaler()
    X = scaler.fit_transform(feat.values)
    model = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
    model.fit(X)
    labels = model.predict(X)
    probs = model.predict_proba(X)
    tmp = feat.copy()
    tmp["label"] = labels
    regime_map = classify_cluster_regime(tmp.assign(ret_63d=feat["ret_63d"], vol_21=feat["vol_21"]), "label")
    cur = int(labels[-1])
    return regime_map.get(cur, "neutral"), f"cluster={cur}, conf={np.max(probs[-1]):.3f}", float(np.max(probs[-1]))

def kmeans_regime_signal(d: pd.DataFrame) -> Tuple[str, str, float]:
    cols = ["ret_21d", "ret_63d", "vol_21", "dist_sma50", "dist_sma200", "rsi_14"]
    feat = d[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(feat) < 250:
        return "neutral", "insufficient data", 0.0
    scaler = StandardScaler()
    X = scaler.fit_transform(feat.values)
    km = KMeans(n_clusters=3, n_init=20, random_state=42)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_
    cur_label = int(labels[-1])
    cur_x = X[-1]
    dist = np.linalg.norm(cur_x - centers[cur_label])
    conf = float(1 / (1 + dist))
    tmp = feat.copy()
    tmp["label"] = labels
    regime_map = classify_cluster_regime(tmp.assign(ret_63d=feat["ret_63d"], vol_21=feat["vol_21"]), "label")
    return regime_map.get(cur_label, "neutral"), f"cluster={cur_label}, conf={conf:.3f}", conf

def logistic_21d_signal(d: pd.DataFrame) -> Tuple[str, str, float]:
    cols = ["ret_5d","ret_21d","ret_63d","ret_126d","dist_sma200","sma50_sma200","rsi_14","macd","macd_signal","vol_21"]
    x = d[cols].replace([np.inf,-np.inf], np.nan).dropna()
    y = d.loc[x.index, "target_21d_up"]
    if len(x) < 400:
        return "neutral", "insufficient data", 0.0
    split = int(len(x) * 0.8)
    model = LogisticRegression(max_iter=2000)
    model.fit(x.iloc[:split], y.iloc[:split])
    prob = float(model.predict_proba(x.iloc[[-1]])[0,1])
    signal = "bullish" if prob >= 0.55 else "bearish" if prob <= 0.45 else "neutral"
    conf = abs(prob - 0.5) * 2
    return signal, f"p_up_21d={prob:.3f}", conf

def random_forest_risk_signal(d: pd.DataFrame) -> Tuple[str, str, float, Dict[str, float]]:
    future_dd = (d["Close"].rolling(21).min().shift(-21) / d["Close"] - 1)
    risk_label = (future_dd <= -0.05).astype(int)
    cols = ["ret_5d","ret_21d","ret_63d","dist_sma50","dist_sma200","rsi_14","macd","vol_21","vol_63","atr_14","range_pct","volume_z"]
    x = d[cols].replace([np.inf,-np.inf], np.nan).dropna()
    y = risk_label.loc[x.index]
    valid = y.notna()
    x, y = x.loc[valid], y.loc[valid]
    if len(x) < 400:
        return "neutral", "insufficient data", 0.0, {}
    split = int(len(x) * 0.8)
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced_subsample'
    )
    model.fit(x.iloc[:split], y.iloc[:split])
    p_risk = float(model.predict_proba(x.iloc[[-1]])[0,1])
    signal = "bearish" if p_risk >= 0.55 else "bullish" if p_risk <= 0.35 else "neutral"
    importances = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False).head(4)
    top_imp = {k: round(float(v), 4) for k, v in importances.items()}
    conf = abs(p_risk - 0.5) * 2
    return signal, f"crash_risk_21d={p_risk:.3f}", conf, top_imp

def monte_carlo_signal(d: pd.DataFrame, horizon: int = 63, sims: int = MC_SIMS) -> Tuple[str, str, float]:
    rets = d["ret_1d"].dropna().tail(252 * 3)
    if len(rets) < 200:
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

# ---------------------------------------------------------------------
# COMPOSITE TECHNICALS
# ---------------------------------------------------------------------
def technical_signals(d: pd.DataFrame) -> List[Dict[str, Any]]:
    last = d.dropna().iloc[-1]
    results: List[Dict[str, Any]] = []

    add_analysis(results, "Trend 50/200",
        "bullish" if last["sma_50"] > last["sma_200"] else "bearish",
        round(float(last["sma50_sma200"] * 100), 2), "Golden/death cross regime")
    add_analysis(results, "Price vs SMA200",
        "bullish" if last["Close"] > last["sma_200"] else "bearish",
        round(float(last["dist_sma200"] * 100), 2), "Percent above/below 200-day average")
    add_analysis(results, "Price vs SMA50",
        "bullish" if last["Close"] > last["sma_50"] else "bearish",
        round(float(last["dist_sma50"] * 100), 2), "Percent above/below 50-day average")
    add_analysis(results, "63/126 Momentum",
        "bullish" if last["ret_63d"] > 0 and last["ret_126d"] > 0 else "bearish" if last["ret_63d"] < 0 and last["ret_126d"] < 0 else "neutral",
        round(float(last["ret_63d"] * 100), 2), "Quarter and half-year momentum")
    add_analysis(results, "MACD Regime",
        "bullish" if last["macd"] > last["macd_signal"] and last["macd"] > 0 else "bearish" if last["macd"] < last["macd_signal"] and last["macd"] < 0 else "neutral",
        round(float(last["macd"]), 4), "MACD vs signal and zero line")
    add_analysis(results, "RSI Swing",
        "bullish" if 50 < last["rsi_14"] < 68 else "bearish" if 32 < last["rsi_14"] < 50 else "neutral",
        round(float(last["rsi_14"]), 2), "RSI trend zone with extension awareness")
    add_analysis(results, "21-day Return",
        "bullish" if last["ret_21d"] > 0 else "bearish",
        round(float(last["ret_21d"] * 100), 2), "One-month return sign")
    vm = float(last["ret_63d"] / max(last["vol_21"], 1e-9))
    add_analysis(results, "Vol-adjusted Momentum",
        "bullish" if vm > 0.4 else "bearish" if vm < -0.4 else "neutral",
        round(vm, 3), "Momentum divided by annualized vol")
    return results

def extension_filter(d: pd.DataFrame) -> Tuple[str, str]:
    last = d.dropna().iloc[-1]
    overextended_up = last["rsi_14"] >= 72 and last["dist_sma50"] >= 0.045
    overextended_dn = last["rsi_14"] <= 30 and last["dist_sma50"] <= -0.045
    if overextended_up:
        return "neutral", f"bullish trend but extended (RSI={last['rsi_14']:.1f}, dist50={last['dist_sma50']:.2%})"
    if overextended_dn:
        return "neutral", f"bearish trend but stretched down (RSI={last['rsi_14']:.1f}, dist50={last['dist_sma50']:.2%})"
    return ("bullish" if last["Close"] > last["sma_50"] else "bearish"), "extension filter clear"

def cross_asset_signals(market: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    latest = {}
    for t, df in market.items():
        d = engineer_features(df)
        latest[t] = d.dropna().iloc[-1]

    risk_on_bulls = sum(1 for t in RISK_ON_SET if latest[t]["Close"] > latest[t]["sma_200"] and latest[t]["ret_63d"] > 0)
    risk_off_bulls = sum(1 for t in RISK_OFF_SET if latest[t]["Close"] > latest[t]["sma_200"] and latest[t]["ret_63d"] > 0)
    stress_up = latest["VIXY"]["ret_21d"] > 0 and latest["VIXY"]["Close"] > latest["VIXY"]["sma_50"]

    breadth_signal = "bullish" if risk_on_bulls >= 3 else "bearish" if risk_on_bulls <= 1 else "neutral"
    add_analysis(results, "Breadth Risk-On", breadth_signal, f"{risk_on_bulls}/4",
                 "SPY, QQQ, IWM, HYG above SMA200 with positive 63d momentum")

    if stress_up and risk_off_bulls >= 1:
        macro_signal = "bearish"
    elif risk_on_bulls >= 3 and not stress_up:
        macro_signal = "bullish"
    else:
        macro_signal = "neutral"

    add_analysis(results, "Cross-Asset Macro", macro_signal, f"risk_off={risk_off_bulls}, stress={'UP' if stress_up else 'DOWN'}",
                 "TLT/GLD and VIXY stress overlay")
    return results

# ---------------------------------------------------------------------
# COMPOSITE SCORE & REGIME CHANGE
# ---------------------------------------------------------------------
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

def compute_historical_composite_regime(d: pd.DataFrame) -> Tuple[Optional[pd.Timestamp], Optional[float]]:
    d = d.dropna().copy()
    if d.empty:
        return None, None

    bull = pd.Series(0.0, index=d.index)
    bear = pd.Series(0.0, index=d.index)

    w_trend = MODEL_WEIGHTS.get("Trend 50/200", 1.1)
    bull += w_trend * (d["sma_50"] > d["sma_200"]).astype(float)
    bear += w_trend * (d["sma_50"] < d["sma_200"]).astype(float)

    w_p200 = MODEL_WEIGHTS.get("Price vs SMA200", 1.0)
    bull += w_p200 * (d["Close"] > d["sma_200"]).astype(float)
    bear += w_p200 * (d["Close"] < d["sma_200"]).astype(float)

    w_p50 = MODEL_WEIGHTS.get("Price vs SMA50", 0.8)
    bull += w_p50 * (d["Close"] > d["sma_50"]).astype(float)
    bear += w_p50 * (d["Close"] < d["sma_50"]).astype(float)

    w_mom = MODEL_WEIGHTS.get("63/126 Momentum", 1.2)
    bull += w_mom * ((d["ret_63d"] > 0) & (d["ret_126d"] > 0)).astype(float)
    bear += w_mom * ((d["ret_63d"] < 0) & (d["ret_126d"] < 0)).astype(float)

    w_macd = MODEL_WEIGHTS.get("MACD Regime", 0.9)
    bull += w_macd * ((d["macd"] > d["macd_signal"]) & (d["macd"] > 0)).astype(float)
    bear += w_macd * ((d["macd"] < d["macd_signal"]) & (d["macd"] < 0)).astype(float)

    w_rsi = MODEL_WEIGHTS.get("RSI Swing", 0.5)
    bull += w_rsi * ((d["rsi_14"] > 50) & (d["rsi_14"] < 68)).astype(float)
    bear += w_rsi * ((d["rsi_14"] > 32) & (d["rsi_14"] < 50)).astype(float)

    w_21 = MODEL_WEIGHTS.get("21-day Return", 0.6)
    bull += w_21 * (d["ret_21d"] > 0).astype(float)
    bear += w_21 * (d["ret_21d"] < 0).astype(float)

    vm = d["ret_63d"] / d["vol_21"].replace(0, np.nan)
    w_vm = MODEL_WEIGHTS.get("Vol-adjusted Momentum", 0.9)
    bull += w_vm * (vm > 0.4).astype(float)
    bear += w_vm * (vm < -0.4).astype(float)

    score = bull - bear
    regime = pd.Series(0, index=d.index, dtype=int)
    regime[score >= 2.5] = 1
    regime[score <= -2.5] = -1

    current = int(regime.iloc[-1])
    if len(regime) < 2:
        return d.index[-1], float(d["Close"].iloc[-1])

    for i in range(len(regime) - 2, -1, -1):
        if int(regime.iloc[i]) != current:
            start_idx = d.index[i + 1]
            return start_idx, float(d.loc[start_idx, "Close"])

    first_idx = d.index[0]
    return first_idx, float(d.loc[first_idx, "Close"])

# ---------------------------------------------------------------------
# PERSISTENCE FILTER / HISTORY LOGGING
# ---------------------------------------------------------------------
def load_recent_history(ticker: str) -> pd.DataFrame:
    if not HISTORY_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(HISTORY_FILE)
        return df[df["ticker"] == ticker].tail(PERSISTENCE_DAYS - 1)
    except Exception:
        return pd.DataFrame()

def persistence_filter(ticker: str, raw_signal: str) -> Tuple[str, str]:
    if PERSISTENCE_DAYS <= 1:
        return raw_signal, "persistence disabled"
    hist = load_recent_history(ticker)
    if hist.empty or len(hist) < PERSISTENCE_DAYS - 1:
        return raw_signal, f"insufficient history for persistence ({len(hist)}/{PERSISTENCE_DAYS - 1})"
    prior = list(hist["raw_final_signal"].astype(str))
    if all(x == raw_signal for x in prior) and raw_signal != "neutral":
        return raw_signal, f"confirmed for {PERSISTENCE_DAYS} consecutive runs"
    if raw_signal == "neutral":
        return "neutral", "current signal neutral"
    return "neutral", f"awaiting persistence confirmation; prior={prior}"

def atr_risk_text(product: str, open_positions: List[OpenPosition], amp_stop_text: str, ticker_df: pd.DataFrame) -> List[str]:
    d = engineer_features(ticker_df).dropna()
    last = d.iloc[-1]
    atr = float(last["atr_14"])
    close = float(last["Close"])
    atr_pct = atr / close
    lines = [f"ETF ATR14 proxy: {atr:.2f} ({atr_pct:.2%} of price)"]
    if amp_stop_text and "@" in amp_stop_text:
        lines.append(f"AMP nearest risk order: {amp_stop_text}")
    if open_positions:
        for p in open_positions:
            lines.append(f"Live C2 position: {p.symbol} {p.side.upper()} x {p.qty} @ {p.entry_price}")
    return lines

def save_run(rows: List[Dict[str, Any]]):
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = HISTORY_FILE.exists()
    with open(HISTORY_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

# ---------------------------------------------------------------------
# FULL ANALYSIS FOR ONE TICKER
# ---------------------------------------------------------------------
def analysis_suite(ticker: str, df: pd.DataFrame, cross_asset_rows: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, float]]:
    d = engineer_features(df)
    d_clean = d.dropna().copy()
    if d_clean.empty:
        raise RuntimeError(f"No usable feature rows for {ticker}")

    results = technical_signals(d)

    sig, note, _ = hmm_regime_signal(d)
    add_analysis(results, "Gaussian HMM", sig, note, "3-state hidden Markov regime detection")

    sig, note, _ = markov_switch_signal(d)
    add_analysis(results, "Markov Switching", sig, note, "2-regime Markov regression using filtered probabilities")

    sig, note, _ = gmm_regime_signal(d)
    add_analysis(results, "Gaussian Mixture", sig, note, "3-cluster unsupervised regime classification")

    sig, note, _ = kmeans_regime_signal(d)
    add_analysis(results, "KMeans Regime", sig, note, "3-cluster unsupervised regime classification")

    sig, note, _ = logistic_21d_signal(d)
    add_analysis(results, "Logistic ML 21d", sig, note, "Probability next 21-day return is positive")

    sig, note, _, rf_top = random_forest_risk_signal(d)
    add_analysis(results, "Random Forest Risk", sig, note, f"Probability of >=5% drawdown over next 21 days; top features={rf_top}")

    sig, note, _ = monte_carlo_signal(d)
    add_analysis(results, "Monte Carlo 63d", sig, note, "Simulated 63-day return distribution")

    ext_sig, ext_note = extension_filter(d)
    add_analysis(results, "Extension Filter", ext_sig, ext_note, "Avoid chasing overextended swing conditions")

    results.extend(cross_asset_rows)

    out = pd.DataFrame(results)
    bullish = int((out["signal"] == "bullish").sum())
    bearish = int((out["signal"] == "bearish").sum())
    neutral = int((out["signal"] == "neutral").sum())

    raw_score, confidence, raw_final = weighted_score(out)
    persisted_final, persistence_note = persistence_filter(ticker, raw_final)
    comp_date, comp_price = compute_historical_composite_regime(d_clean)

    last = d_clean.iloc[-1]
    summary = {
        "ticker": ticker,
        "close": float(last["Close"]),
        "sma_50": float(last["sma_50"]),
        "sma_200": float(last["sma_200"]),
        "rsi_14": float(last["rsi_14"]),
        "atr_14": float(last["atr_14"]),
        "vol_21_annualized": float(last["vol_21"]),
        "bullish_count": bullish,
        "bearish_count": bearish,
        "neutral_count": neutral,
        "analyses_total": int(len(out)),
        "raw_score": float(raw_score),
        "confidence": float(confidence),
        "raw_final_signal": raw_final,
        "final_signal": persisted_final,
        "persistence_note": persistence_note,
        "composite_change_date": comp_date.strftime('%Y-%m-%d') if comp_date is not None else '',
        "composite_change_price": float(comp_price) if comp_price is not None else float('nan'),
    }
    return out, summary, rf_top

# ---------------------------------------------------------------------
# COMBINED MARKET CONFIRMATION
# ---------------------------------------------------------------------
def combine_market_confirmation(spy_summary: dict, qqq_summary: dict, amp_side: Optional[str]) -> Dict[str, Any]:
    score = spy_summary["raw_score"] + qqq_summary["raw_score"]
    combined = "bullish" if score >= 4 else "bearish" if score <= -4 else "neutral"

    persisted = (
        "bullish"
        if spy_summary["final_signal"] == "bullish" and qqq_summary["final_signal"] == "bullish"
        else "bearish"
        if spy_summary["final_signal"] == "bearish" and qqq_summary["final_signal"] == "bearish"
        else "neutral"
    )

    congruent = False
    if amp_side is not None:
        congruent = (amp_side == "long" and combined == "bullish") or (amp_side == "short" and combined == "bearish")

    return {
        "market_signal": combined,
        "persistent_signal": persisted,
        "congruent_with_amp": congruent,
        "score": score,
    }

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    apikey = os.getenv("C2_API_KEY", "").strip()
    systemid_raw = os.getenv("C2_SYSTEM_ID", "").strip()
    now = datetime.now(timezone.utc).isoformat()

    print(f"=== AMP + FUTURES CAPITAL REGIME TOOLBOX at {now} UTC ===")
    print(
        f"Dependencies: HMM={HMM_AVAILABLE}, Markov={MARKOV_AVAILABLE} "
        f"| Lookback={LOOKBACK_YEARS}y daily | Persistence={PERSISTENCE_DAYS} "
        f"| AutoTrade={AUTO_TRADE_ENABLED} | DryRun={DRY_RUN} | AgreementThreshold={AGREEMENT_THRESHOLD*100:.0f}%"
    )

    html = fetch_amp_html()
    rows = parse_current_session(html)
    if not rows:
        raise RuntimeError("No rows parsed from AMP current session table")
    best_row = pick_best_supported(rows)
    if not best_row:
        raise RuntimeError("No ES or NQ strategy found in top rows")

    desired_pos = parse_direction_and_size(best_row.current_position)
    if not desired_pos:
        print(
            YELLOW + BOLD +
            f"\nNO NEW SIGNALS: AMP is FLAT for top system ({best_row.system}, product {best_row.product})." +
            RESET
        )

    current_positions = []
    if apikey and systemid_raw:
        try:
            open_pos_raw = get_open_positions(apikey, int(systemid_raw))
            current_positions = extract_supported_open_positions(open_pos_raw)
        except Exception as e:
            print(f"Could not fetch C2 open positions: {e}")

    market_data = {ticker: download_market_data(ticker) for ticker in ALL_TICKERS}
    cross_asset_rows = cross_asset_signals(market_data)

    tables = {}
    summaries = {}
    rf_features = {}

    for ticker in ["SPY", "QQQ"]:
        tbl, summary, rf_top = analysis_suite(ticker, market_data[ticker], cross_asset_rows)
        tables[ticker] = tbl
        summaries[ticker] = summary
        rf_features[ticker] = rf_top

    confirmation = combine_market_confirmation(summaries["SPY"], summaries["QQQ"], desired_pos.side if desired_pos else None)
    traded_etf = ETF_MAP.get(best_row.product, "QQQ")

    print_block(
        "SCRAPED AMP SIGNAL",
        [
            f"Top system: {best_row.system}",
            f"Developer: {best_row.developer or 'N/A'}",
            f"Product: {best_row.product}",
            f"Scraped current position: {best_row.current_position}",
            f"Nearest order: {best_row.nearest_order}",
            f"AMP interpreted side: {desired_pos.side.upper()} x {desired_pos.qty}" if desired_pos else "AMP interpreted side: FLAT / NO ACTIVE SIGNAL",
            f"ETF proxy used for traded product: {traded_etf}",
        ],
        CYAN,
    )

    if current_positions:
        pos_lines = [f"{p.symbol} | {p.side.upper()} | Qty={p.qty} | Entry={p.entry_price}" for p in current_positions]
    else:
        pos_lines = ["No supported open MES/MNQ positions detected or C2 credentials unavailable."]
    print_block("OPEN POSITIONS", pos_lines, GREEN)

    print_block(
        "RISK NOTES",
        atr_risk_text(best_row.product, current_positions, best_row.nearest_order, market_data[traded_etf]),
        MAGENTA,
    )

    for ticker in ["SPY", "QQQ"]:
        s = summaries[ticker]
        base_signal = s["final_signal"] if s["final_signal"] != "neutral" else s["raw_final_signal"]
        color = color_for_signal(base_signal)

        comp_since = "N/A"
        days_age = None
        if s["composite_change_date"]:
            comp_since = f"{s['composite_change_date']} at {s['composite_change_price']:.2f}"
            days_age = days_since(s["composite_change_date"])

        age_text = f"{days_age} days since last change" if days_age is not None else "Age: unknown"

        print_block(
            f"{ticker} SUMMARY",
            [
                f"Close: {s['close']:.2f}",
                f"SMA50: {s['sma_50']:.2f} | SMA200: {s['sma_200']:.2f}",
                f"RSI14: {s['rsi_14']:.2f} | ATR14: {s['atr_14']:.2f} | 21d Annualized Vol: {s['vol_21_annualized']:.2%}",
                f"Bullish: {s['bullish_count']} | Bearish: {s['bearish_count']} | Neutral: {s['neutral_count']}",
                f"Weighted score: {s['raw_score']:.2f} | Confidence: {s['confidence']:.1%}",
                f"Raw regime: {s['raw_final_signal'].upper()} | Final regime after persistence: {s['final_signal'].upper()}",
                f"Composite regime in force since: {comp_since}",
                f"Composite regime age: {age_text}",
                f"Persistence note: {s['persistence_note']}",
            ],
            color,
        )

        for _, row in tables[ticker].iterrows():
            c = color_for_signal(row["signal"])
            print(f"{c}{row['analysis']:<20} | {row['signal'].upper():7s} | {row['value']} | {row['note']}{RESET}")

    final_color = color_for_signal(confirmation["market_signal"])
    amp_align = "YES" if confirmation["congruent_with_amp"] else "NO"

    print_block(
        "FINAL RECOMMENDATION",
        [
            f"SPY final: {summaries['SPY']['final_signal'].upper()} | raw score {summaries['SPY']['raw_score']:.2f} | confidence {summaries['SPY']['confidence']:.1%}",
            f"QQQ final: {summaries['QQQ']['final_signal'].upper()} | raw score {summaries['QQQ']['raw_score']:.2f} | confidence {summaries['QQQ']['confidence']:.1%}",
            f"Combined weighted ETF confirmation: {confirmation['market_signal'].upper()} | aggregate score {confirmation['score']:.2f}",
            f"Persistent regime overlay: {confirmation['persistent_signal'].upper()}",
            f"Congruent with AMP scraped side ({desired_pos.side.upper() if desired_pos else 'FLAT/NA'}): {amp_align}",
            f"Top system: {best_row.system} | Nearest order: {best_row.nearest_order}",
        ],
        final_color,
    )

    if desired_pos and desired_pos.side == "long" and confirmation["market_signal"] == "bullish":
        print(GREEN + BOLD + "ACTION BIAS: BUY / LONG SWING SETUP CONFIRMED" + RESET)
    elif desired_pos and desired_pos.side == "short" and confirmation["market_signal"] == "bearish":
        print(RED + BOLD + "ACTION BIAS: SELL / SHORT SWING SETUP CONFIRMED" + RESET)
    else:
        print(YELLOW + BOLD + "ACTION BIAS: MIXED / NO STRONG CONFIRMATION or AMP FLAT" + RESET)

    # -----------------------------------------------------------------
    # AUTO-TRADE TO C2
    # -----------------------------------------------------------------
    if not desired_pos:
        print(YELLOW + "AUTO-TRADE: AMP is flat for the top system. No trade sent." + RESET)
    elif AUTO_TRADE_ENABLED:
        if not apikey or not systemid_raw:
            print(RED + "AUTO-TRADE: missing C2_API_KEY or C2_SYSTEM_ID. No orders sent." + RESET)
        else:
            systemid = int(systemid_raw)
            etf_summary = summaries[traded_etf]
            agreement_pct, agreement_direction = compute_agreement(etf_summary)

            print_block(
                "AUTO-TRADE CHECK",
                [
                    f"ETF used for gate: {traded_etf}",
                    f"Agreement: {agreement_pct:.1%}",
                    f"Agreement direction: {agreement_direction.upper()}",
                    f"AMP side: {desired_pos.side.upper()}",
                    f"Threshold required: {AGREEMENT_THRESHOLD:.0%}",
                ],
                BLUE,
            )

            aligned = (
                (desired_pos.side == "long" and agreement_direction == "bullish") or
                (desired_pos.side == "short" and agreement_direction == "bearish")
            )

            if not aligned:
                print(YELLOW + "AUTO-TRADE: regime direction does not align with AMP side. No order sent." + RESET)
            elif agreement_pct < AGREEMENT_THRESHOLD:
                print(YELLOW + f"AUTO-TRADE: agreement {agreement_pct:.1%} is below threshold. No order sent." + RESET)
            elif not confirmation["congruent_with_amp"]:
                print(YELLOW + "AUTO-TRADE: combined confirmation is not congruent with AMP side. No order sent." + RESET)
            else:
                desired_symbol = SUPPORTED_PRODUCTS[best_row.product]

                if len(current_positions) > 1:
                    print(RED + "AUTO-TRADE: more than one supported open MES/MNQ position detected. Aborting." + RESET)
                else:
                    current = current_positions[0] if current_positions else None

                    if current is None:
                        entry_payload = build_market_order(systemid, desired_symbol, desired_pos.side, desired_pos.qty)
                        print(BLUE + "Entry payload: " + json.dumps(entry_payload, ensure_ascii=False) + RESET)
                        if DRY_RUN:
                            print(YELLOW + "DRY_RUN=1 -> not sending entry order." + RESET)
                        else:
                            entry_result = api4_post("/Strategies/NewStrategyOrder", apikey, entry_payload)
                            print(GREEN + "Entry result: " + json.dumps(entry_result, ensure_ascii=False) + RESET)
                    else:
                        same_symbol = current.symbol == desired_symbol
                        same_side = current.side == desired_pos.side
                        same_qty = current.qty == desired_pos.qty

                        if same_symbol and same_side and same_qty:
                            print(GREEN + "AUTO-TRADE: current C2 position already matches confirmed AMP signal. No order sent." + RESET)
                        else:
                            close_payload = build_close_order(systemid, current)
                            entry_payload = build_market_order(systemid, desired_symbol, desired_pos.side, desired_pos.qty)

                            print(BLUE + "Close payload: " + json.dumps(close_payload, ensure_ascii=False) + RESET)
                            print(BLUE + "New entry payload: " + json.dumps(entry_payload, ensure_ascii=False) + RESET)

                            if DRY_RUN:
                                print(YELLOW + "DRY_RUN=1 -> not sending close or entry orders." + RESET)
                            else:
                                close_result = api4_post("/Strategies/NewStrategyOrder", apikey, close_payload)
                                print(GREEN + "Close result: " + json.dumps(close_result, ensure_ascii=False) + RESET)
                                entry_result = api4_post("/Strategies/NewStrategyOrder", apikey, entry_payload)
                                print(GREEN + "Entry result: " + json.dumps(entry_result, ensure_ascii=False) + RESET)
    else:
        print(YELLOW + "AUTO-TRADE DISABLED. Set AUTO_TRADE_ENABLED=true to enable." + RESET)

    run_rows = []
    for ticker in ["SPY", "QQQ"]:
        s = summaries[ticker]
        run_rows.append(
            {
                "timestamp_utc": now,
                "ticker": ticker,
                "amp_system": best_row.system,
                "amp_product": best_row.product,
                "amp_side": desired_pos.side if desired_pos else "flat",
                "amp_nearest_order": best_row.nearest_order,
                "raw_final_signal": s["raw_final_signal"],
                "final_signal": s["final_signal"],
                "raw_score": round(s["raw_score"], 4),
                "confidence": round(s["confidence"], 4),
                "bullish_count": s["bullish_count"],
                "bearish_count": s["bearish_count"],
                "neutral_count": s["neutral_count"],
                "persistence_note": s["persistence_note"],
                "composite_change_date": s["composite_change_date"],
                "composite_change_price": s["composite_change_price"],
                "combined_market_signal": confirmation["market_signal"],
                "combined_persistent_signal": confirmation["persistent_signal"],
                "amp_congruent": confirmation["congruent_with_amp"],
            }
        )

    save_run(run_rows)
    print(BLUE + f"Logged run history to: {HISTORY_FILE}" + RESET)

if __name__ == "__main__":
    main()

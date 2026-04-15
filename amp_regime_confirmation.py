#!/usr/bin/env python3
import csv
import json
import os
import re
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone, date
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except Exception:
    raise RuntimeError("Missing dependency: yfinance. Install with: pip install yfinance")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
except Exception:
    raise RuntimeError("Missing scikit-learn. Install with: pip install scikit-learn")

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
AGREEMENT_THRESHOLD = float(os.getenv("AGREEMENT_THRESHOLD", "0.75"))

AUTO_TRADE_ENABLED = os.getenv("AUTO_TRADE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
DRY_RUN = os.getenv("DRY_RUN", "0").strip() == "1"

HISTORY_DIR = Path(os.getenv("HISTORY_DIR", "./history"))
HISTORY_FILE = HISTORY_DIR / "regime_runs.csv"

MES_SYMBOL = os.getenv("MES_SYMBOL", "@MESM6")
MNQ_SYMBOL = os.getenv("MNQ_SYMBOL", "@MNQM6")
SUPPORTED_PRODUCTS = {"ES": MES_SYMBOL, "NQ": MNQ_SYMBOL}
ETF_MAP = {"ES": "SPY", "NQ": "QQQ"}

RISK_ON_SET = ["SPY", "QQQ", "IWM", "HYG"]
RISK_OFF_SET = ["TLT", "GLD"]
STRESS_SET = ["VIXY"]
ALL_TICKERS = list(dict.fromkeys(RISK_ON_SET + RISK_OFF_SET + STRESS_SET))

ETF_FALLBACKS = {
    "SPY": ["SPY", "IVV", "VOO"],
    "QQQ": ["QQQ", "QQQM"],
    "IWM": ["IWM"],
    "HYG": ["HYG", "JNK"],
    "TLT": ["TLT", "IEF"],
    "GLD": ["GLD", "IAU"],
    "VIXY": ["VIXY", "UVXY"],
}

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

summaries_global: Dict[str, Dict[str, Any]] = {}


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
# UTILS
# ---------------------------------------------------------------------
def color_for_signal(signal: str) -> str:
    s = str(signal).lower()
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
        d = datetime.strptime(str(date_str), "%Y-%m-%d").date()
        return (date.today() - d).days
    except Exception:
        return None


def to_float_safe(v: Any) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def safe_pct(a: float, b: float) -> float:
    if b is None or b == 0 or pd.isna(b):
        return 0.0
    return float(a) / float(b)


def signal_from_score(score: float, bull: float = 0.15, bear: float = -0.15) -> str:
    if score >= bull:
        return "bullish"
    if score <= bear:
        return "bearish"
    return "neutral"


def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0).replace(0, np.nan)
    return (s - mu) / sd


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
    text = str(text).strip()
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
        extracted.append(
            OpenPosition(
                symbol=sym,
                side=side,
                qty=abs(int(qty)),
                entry_price=to_float_safe(entry_price),
                raw=p,
            )
        )

    return extracted


def build_parent_order_market_only(
    strategy_id: int,
    full_symbol: str,
    side: str,
    qty: int,
) -> dict:
    c2_side = "1" if side == "long" else "2"
    return {
        "Order": {
            "StrategyId": strategy_id,
            "OrderType": "1",
            "Side": c2_side,
            "OrderQuantity": int(qty),
            "TIF": "0",
            "C2Symbol": {
                "FullSymbol": full_symbol,
                "SymbolType": "future",
            },
        }
    }


def build_close_order(strategy_id: int, open_pos: OpenPosition) -> dict:
    close_side_code = "2" if open_pos.side == "long" else "1"
    return {
        "Order": {
            "StrategyId": strategy_id,
            "OrderType": "1",
            "Side": close_side_code,
            "OpenClose": "C",
            "OrderQuantity": int(open_pos.qty),
            "TIF": "0",
            "C2Symbol": {
                "FullSymbol": open_pos.symbol,
                "SymbolType": "future",
            },
        }
    }


# ---------------------------------------------------------------------
# MARKET DATA
# ---------------------------------------------------------------------
def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        try:
            if "Open" in df.columns.get_level_values(0):
                df.columns = [c[0] for c in df.columns]
            else:
                df.columns = [c[-1] for c in df.columns]
        except Exception:
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    rename_map = {}
    for c in df.columns:
        lc = str(c).lower()
        if lc == "open":
            rename_map[c] = "Open"
        elif lc == "high":
            rename_map[c] = "High"
        elif lc == "low":
            rename_map[c] = "Low"
        elif lc == "close":
            rename_map[c] = "Close"
        elif lc == "volume":
            rename_map[c] = "Volume"
    df = df.rename(columns=rename_map)

    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in df.columns for col in required):
        return pd.DataFrame()

    out = df[required].copy().dropna()
    out = out[~out.index.duplicated(keep="last")]
    return out


def _try_download(symbol: str, years: int) -> pd.DataFrame:
    df = yf.download(
        symbol,
        period=f"{years}y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    return _normalize_ohlcv(df)


def _try_history(symbol: str, years: int) -> pd.DataFrame:
    df = yf.Ticker(symbol).history(
        period=f"{years}y",
        interval="1d",
        auto_adjust=True,
    )
    return _normalize_ohlcv(df)


def download_market_data(ticker: str, years: int = LOOKBACK_YEARS) -> pd.DataFrame:
    candidates = ETF_FALLBACKS.get(ticker, [ticker])
    last_error = None

    for candidate in candidates:
        for _ in range(3):
            try:
                df = _try_download(candidate, years)
                if not df.empty:
                    if candidate != ticker:
                        print(YELLOW + f"Market data fallback used for {ticker}: {candidate}" + RESET)
                    return df
            except Exception as e:
                last_error = e
            time.sleep(1)

        for _ in range(2):
            try:
                df = _try_history(candidate, years)
                if not df.empty:
                    if candidate != ticker:
                        print(YELLOW + f"Market data fallback used for {ticker}: {candidate}" + RESET)
                    return df
            except Exception as e:
                last_error = e
            time.sleep(1)

    raise RuntimeError(f"No data downloaded for {ticker}. Last error: {last_error}")


# ---------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    close = x["Close"]
    high = x["High"]
    low = x["Low"]
    volume = x["Volume"]

    ret1 = close.pct_change()
    ret5 = close.pct_change(5)
    ret21 = close.pct_change(21)
    ret63 = close.pct_change(63)
    ret126 = close.pct_change(126)

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi14 = 100 - 100 / (1 + rs)

    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(14).mean()

    vol21 = ret1.rolling(21).std() * np.sqrt(252)
    vol63 = ret1.rolling(63).std() * np.sqrt(252)
    vol_z = rolling_zscore(vol21, 126)
    price_z_20 = rolling_zscore(close, 20)

    out = pd.DataFrame(index=x.index)
    out["Close"] = close
    out["Open"] = x["Open"]
    out["High"] = high
    out["Low"] = low
    out["Volume"] = volume
    out["ret1"] = ret1
    out["ret5"] = ret5
    out["ret21"] = ret21
    out["mom63"] = ret63
    out["mom126"] = ret126
    out["sma20"] = sma20
    out["sma50"] = sma50
    out["sma200"] = sma200
    out["ema12"] = ema12
    out["ema26"] = ema26
    out["macd"] = macd
    out["macd_sig"] = macd_sig
    out["rsi14"] = rsi14
    out["atr14"] = atr14
    out["atr14_pct"] = atr14 / close
    out["vol21annualized"] = vol21
    out["vol63annualized"] = vol63
    out["vol_z"] = vol_z
    out["price_z20"] = price_z_20
    out["dist_sma50"] = (close - sma50) / sma50
    out["dist_sma200"] = (close - sma200) / sma200
    out["future_21d"] = close.shift(-21) / close - 1.0
    return out.dropna()


# ---------------------------------------------------------------------
# SIGNALS
# ---------------------------------------------------------------------
def technical_signals(feat: pd.DataFrame) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    row = feat.iloc[-1]

    trend_50200 = 1 if row["sma50"] > row["sma200"] else -1
    add_analysis(results, "Trend 50/200", "bullish" if trend_50200 > 0 else "bearish",
                 f"{row['sma50']:.2f}>{row['sma200']:.2f}" if trend_50200 > 0 else f"{row['sma50']:.2f}<{row['sma200']:.2f}",
                 "Primary long-term trend filter")

    pv200 = signal_from_score(row["dist_sma200"], 0.0, 0.0)
    add_analysis(results, "Price vs SMA200", pv200, f"{row['dist_sma200']:.2%}", "Distance from 200-day average")

    pv50 = signal_from_score(row["dist_sma50"], 0.0, 0.0)
    add_analysis(results, "Price vs SMA50", pv50, f"{row['dist_sma50']:.2%}", "Distance from 50-day average")

    mom = row["mom63"] + 0.5 * row["mom126"]
    add_analysis(results, "63/126 Momentum", signal_from_score(mom, 0.03, -0.03), f"{mom:.2%}",
                 "Intermediate-term momentum blend")

    macd_state = row["macd"] - row["macd_sig"]
    add_analysis(results, "MACD Regime", signal_from_score(macd_state, 0.0, 0.0), f"{macd_state:.4f}",
                 "MACD minus signal line")

    rsi_sig = "bullish" if row["rsi14"] >= 55 else "bearish" if row["rsi14"] <= 45 else "neutral"
    add_analysis(results, "RSI Swing", rsi_sig, f"{row['rsi14']:.2f}", "RSI 14 swing zone")

    add_analysis(results, "21-day Return", signal_from_score(row["ret21"], 0.01, -0.01), f"{row['ret21']:.2%}",
                 "Recent 1-month return")

    vol_adj = row["mom63"] / max(row["vol63annualized"], 1e-6)
    add_analysis(results, "Vol-adjusted Momentum", signal_from_score(vol_adj, 0.10, -0.10), f"{vol_adj:.3f}",
                 "Momentum normalized by realized vol")

    return results


def hmm_regime_signal(feat: pd.DataFrame) -> Dict[str, Any]:
    if not HMM_AVAILABLE or len(feat) < 300:
        return {"analysis": "Gaussian HMM", "signal": "neutral", "value": "N/A", "note": "HMM unavailable or insufficient data"}
    try:
        X = feat[["ret1", "vol21annualized", "dist_sma200"]].dropna().copy()
        X = StandardScaler().fit_transform(X)
        model = GaussianHMM(n_components=2, covariance_type="full", n_iter=200, random_state=42)
        model.fit(X)
        states = model.predict(X)
        means = pd.DataFrame({"state": states, "ret": feat.dropna().iloc[-len(states):]["ret1"].values}).groupby("state")["ret"].mean()
        bull_state = means.idxmax()
        sig = "bullish" if states[-1] == bull_state else "bearish"
        return {"analysis": "Gaussian HMM", "signal": sig, "value": int(states[-1]), "note": "2-state Gaussian HMM"}
    except Exception as e:
        return {"analysis": "Gaussian HMM", "signal": "neutral", "value": "ERR", "note": f"HMM error: {e}"}


def markov_switch_signal(feat: pd.DataFrame) -> Dict[str, Any]:
    if not MARKOV_AVAILABLE or len(feat) < 300:
        return {"analysis": "Markov Switching", "signal": "neutral", "value": "N/A", "note": "Markov unavailable or insufficient data"}
    try:
        r = feat["ret1"].dropna() * 100.0
        mod = sm.tsa.MarkovRegression(r, k_regimes=2, trend="c", switching_variance=True)
        res = mod.fit(disp=False)
        probs = res.smoothed_marginal_probabilities
        means = []
        for k in range(probs.shape[1]):
            wk = probs.iloc[:, k]
            means.append((wk * r.loc[wk.index]).sum() / max(wk.sum(), 1e-9))
        bull_state = int(np.argmax(means))
        last_state = int(probs.iloc[-1].idxmax())
        sig = "bullish" if last_state == bull_state else "bearish"
        return {"analysis": "Markov Switching", "signal": sig, "value": last_state, "note": "2-regime Markov regression"}
    except Exception as e:
        return {"analysis": "Markov Switching", "signal": "neutral", "value": "ERR", "note": f"Markov error: {e}"}


def gmm_regime_signal(feat: pd.DataFrame) -> Dict[str, Any]:
    try:
        Xdf = feat[["ret21", "vol21annualized", "dist_sma200"]].dropna()
        if len(Xdf) < 150:
            return {"analysis": "Gaussian Mixture", "signal": "neutral", "value": "N/A", "note": "Insufficient data"}
        X = StandardScaler().fit_transform(Xdf)
        model = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
        model.fit(X)
        states = model.predict(X)
        tmp = Xdf.copy()
        tmp["state"] = states
        means = tmp.groupby("state")["ret21"].mean()
        bull_state = means.idxmax()
        sig = "bullish" if states[-1] == bull_state else "bearish"
        return {"analysis": "Gaussian Mixture", "signal": sig, "value": int(states[-1]), "note": "2-cluster GMM"}
    except Exception as e:
        return {"analysis": "Gaussian Mixture", "signal": "neutral", "value": "ERR", "note": f"GMM error: {e}"}


def kmeans_regime_signal(feat: pd.DataFrame) -> Dict[str, Any]:
    try:
        Xdf = feat[["ret21", "vol21annualized", "dist_sma200"]].dropna()
        if len(Xdf) < 150:
            return {"analysis": "KMeans Regime", "signal": "neutral", "value": "N/A", "note": "Insufficient data"}
        X = StandardScaler().fit_transform(Xdf)
        km = KMeans(n_clusters=2, random_state=42, n_init=20)
        states = km.fit_predict(X)
        tmp = Xdf.copy()
        tmp["state"] = states
        means = tmp.groupby("state")["ret21"].mean()
        bull_state = means.idxmax()
        sig = "bullish" if states[-1] == bull_state else "bearish"
        return {"analysis": "KMeans Regime", "signal": sig, "value": int(states[-1]), "note": "2-cluster KMeans"}
    except Exception as e:
        return {"analysis": "KMeans Regime", "signal": "neutral", "value": "ERR", "note": f"KMeans error: {e}"}


def logistic_21d_signal(feat: pd.DataFrame) -> Dict[str, Any]:
    try:
        cols = ["ret5", "ret21", "mom63", "dist_sma50", "dist_sma200", "rsi14", "vol21annualized", "atr14_pct", "vol_z"]
        df = feat[cols + ["future_21d"]].dropna().copy()
        if len(df) < 250:
            return {"analysis": "Logistic ML 21d", "signal": "neutral", "value": "N/A", "note": "Insufficient data"}
        y = (df["future_21d"] > 0).astype(int)
        X = df[cols]
        split = max(int(len(df) * 0.8), len(df) - 120)
        Xtr, ytr = X.iloc[:split], y.iloc[:split]
        Xte = X.iloc[split:]
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)
        model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        model.fit(Xtr_s, ytr)
        p = model.predict_proba(Xte_s)[-1, 1]
        sig = "bullish" if p >= 0.55 else "bearish" if p <= 0.45 else "neutral"
        return {"analysis": "Logistic ML 21d", "signal": sig, "value": f"{p:.2%}", "note": "Probability next 21d return > 0"}
    except Exception as e:
        return {"analysis": "Logistic ML 21d", "signal": "neutral", "value": "ERR", "note": f"Logistic error: {e}"}


def random_forest_risk_signal(feat: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, float]]:
    try:
        cols = ["ret1", "ret5", "ret21", "mom63", "dist_sma50", "dist_sma200", "rsi14", "vol21annualized", "atr14_pct", "vol_z", "price_z20"]
        df = feat[cols + ["future_21d"]].dropna().copy()
        if len(df) < 300:
            return (
                {"analysis": "Random Forest Risk", "signal": "neutral", "value": "N/A", "note": "Insufficient data"},
                {}
            )
        y = (df["future_21d"] > 0).astype(int)
        X = df[cols]
        split = max(int(len(df) * 0.8), len(df) - 120)
        Xtr, ytr = X.iloc[:split], y.iloc[:split]
        Xte = X.iloc[split:]
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[-1, 1]
        sig = "bullish" if p >= 0.57 else "bearish" if p <= 0.43 else "neutral"
        importances = dict(zip(cols, model.feature_importances_))
        importances = dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:5])
        return (
            {"analysis": "Random Forest Risk", "signal": sig, "value": f"{p:.2%}", "note": "RF probability next 21d return > 0"},
            importances
        )
    except Exception as e:
        return (
            {"analysis": "Random Forest Risk", "signal": "neutral", "value": "ERR", "note": f"RF error: {e}"},
            {}
        )


def monte_carlo_signal(feat: pd.DataFrame, sims: int = MC_SIMS) -> Dict[str, Any]:
    try:
        r = feat["ret1"].dropna().tail(252)
        if len(r) < 100:
            return {"analysis": "Monte Carlo 63d", "signal": "neutral", "value": "N/A", "note": "Insufficient data"}
        mu = r.mean()
        sd = r.std(ddof=0)
        horizon = 63
        rng = np.random.default_rng(42)
        paths = rng.normal(mu, sd, size=(sims, horizon))
        cum = (1.0 + paths).prod(axis=1) - 1.0
        p_up = float((cum > 0).mean())
        sig = "bullish" if p_up >= 0.55 else "bearish" if p_up <= 0.45 else "neutral"
        return {"analysis": "Monte Carlo 63d", "signal": sig, "value": f"{p_up:.2%}", "note": "Probability of positive 63d simulated return"}
    except Exception as e:
        return {"analysis": "Monte Carlo 63d", "signal": "neutral", "value": "ERR", "note": f"MC error: {e}"}


def extension_filter(feat: pd.DataFrame) -> Dict[str, Any]:
    row = feat.iloc[-1]
    z = row["price_z20"]
    if z >= 2.0:
        sig = "bearish"
        note = "Market stretched above 20d distribution"
    elif z <= -2.0:
        sig = "bullish"
        note = "Market stretched below 20d distribution"
    else:
        sig = "neutral"
        note = "No extreme short-term extension"
    return {"analysis": "Extension Filter", "signal": sig, "value": f"{z:.2f}", "note": note}


def cross_asset_signals(market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    closes = {}
    for ticker, df in market_data.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        closes[ticker] = df["Close"].copy()

    aligned = pd.DataFrame(closes).dropna()
    if aligned.empty:
        return pd.DataFrame(columns=["analysis", "signal", "value", "note"])

    ret63 = aligned.pct_change(63).iloc[-1]
    ret21 = aligned.pct_change(21).iloc[-1]

    risk_on_mean = float(ret63[[c for c in RISK_ON_SET if c in ret63.index]].mean())
    risk_off_mean = float(ret63[[c for c in RISK_OFF_SET if c in ret63.index]].mean())
    stress_ret = float(ret21["VIXY"]) if "VIXY" in ret21.index else 0.0

    breadth_score = 0
    breadth_n = 0
    for t in ["SPY", "QQQ", "IWM", "HYG"]:
        if t in aligned.columns:
            s = aligned[t]
            sma50 = s.rolling(50).mean().iloc[-1]
            if pd.notna(sma50):
                breadth_n += 1
                breadth_score += int(s.iloc[-1] > sma50)
    breadth_ratio = safe_pct(breadth_score, max(breadth_n, 1))
    breadth_signal = "bullish" if breadth_ratio >= 0.75 else "bearish" if breadth_ratio <= 0.25 else "neutral"

    spread = risk_on_mean - risk_off_mean
    macro_score = spread - 0.5 * stress_ret
    macro_signal = "bullish" if macro_score > 0.02 else "bearish" if macro_score < -0.02 else "neutral"

    rows.append({
        "analysis": "Breadth Risk-On",
        "signal": breadth_signal,
        "value": f"{breadth_ratio:.2%}",
        "note": "Fraction of SPY/QQQ/IWM/HYG above 50dma",
    })
    rows.append({
        "analysis": "Cross-Asset Macro",
        "signal": macro_signal,
        "value": f"{macro_score:.2%}",
        "note": f"63d risk-on minus risk-off, stress adjusted (VIXY 21d={stress_ret:.2%})",
    })
    return pd.DataFrame(rows)


def weighted_score(results: List[Dict[str, Any]]) -> Tuple[float, int, int, int]:
    score = 0.0
    bull = bear = neutral = 0
    for r in results:
        name = r["analysis"]
        w = MODEL_WEIGHTS.get(name, 1.0)
        sig = str(r["signal"]).lower()
        if sig == "bullish":
            score += w
            bull += 1
        elif sig == "bearish":
            score -= w
            bear += 1
        else:
            neutral += 1
    return score, bull, bear, neutral


def load_recent_history() -> pd.DataFrame:
    if not HISTORY_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(HISTORY_FILE)
        return df
    except Exception:
        return pd.DataFrame()


def compute_historical_composite_regime(ticker: str) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    hist = load_recent_history()
    if hist.empty or "ticker" not in hist.columns:
        return None, None, None
    sub = hist[hist["ticker"] == ticker].copy()
    if sub.empty or "final_signal" not in sub.columns:
        return None, None, None

    sub = sub.sort_values("timestamp_utc")
    last_sig = None
    last_date = None
    last_px = None

    for _, row in sub.iterrows():
        sig = row.get("final_signal")
        if sig != last_sig:
            last_sig = sig
            ts = str(row.get("timestamp_utc", ""))[:10]
            last_date = ts if ts else None
            last_px = to_float_safe(row.get("composite_change_price"))
    return last_sig, last_date, last_px


def persistence_filter(ticker: str, raw_signal: str, close: float) -> Tuple[str, str, Optional[str], Optional[float]]:
    hist = load_recent_history()
    if hist.empty or "ticker" not in hist.columns:
        return raw_signal, "No history yet; using raw signal.", None, close

    sub = hist[hist["ticker"] == ticker].copy()
    if sub.empty or "raw_final_signal" not in sub.columns:
        return raw_signal, "No ticker-specific history; using raw signal.", None, close

    sub = sub.sort_values("timestamp_utc")
    tail = sub.tail(max(PERSISTENCE_DAYS - 1, 0))

    recent = tail["raw_final_signal"].tolist() + [raw_signal]
    same = len(recent) >= PERSISTENCE_DAYS and all(x == raw_signal for x in recent[-PERSISTENCE_DAYS:])

    if same:
        last_change_date = str(datetime.now(timezone.utc).date())
        return raw_signal, f"Signal persisted for {PERSISTENCE_DAYS} checks.", last_change_date, close

    prior_final = sub["final_signal"].dropna().iloc[-1] if "final_signal" in sub.columns and not sub["final_signal"].dropna().empty else "neutral"
    if prior_final == raw_signal:
        return raw_signal, "Raw signal matches prior final signal.", None, close

    return prior_final, f"Persistence gate held prior final signal until {PERSISTENCE_DAYS} aligned runs.", None, close


def atr_risk_text(product: str, current_positions: List[OpenPosition], nearest_order: str, proxy_df: pd.DataFrame) -> List[str]:
    if proxy_df is None or proxy_df.empty:
        return ["ATR risk unavailable: proxy data missing."]
    feat = engineer_features(proxy_df)
    if feat.empty:
        return ["ATR risk unavailable: insufficient proxy history."]
    row = feat.iloc[-1]
    atr = row["atr14"]
    close = row["Close"]
    atr_pct = row["atr14_pct"]
    lines = [
        f"Proxy close: {close:.2f}",
        f"ATR14: {atr:.2f} ({atr_pct:.2%} of price)",
        f"Nearest AMP order reference: {nearest_order}",
        f"Mapped futures symbol: {SUPPORTED_PRODUCTS.get(product.upper(), 'N/A')}",
    ]
    if current_positions:
        for p in current_positions:
            lines.append(f"Open position: {p.symbol} {p.side.upper()} x {p.qty} @ {p.entry_price}")
    else:
        lines.append("No currently detected MES/MNQ position for ATR framing.")
    return lines


def save_run(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    write_header = not HISTORY_FILE.exists()
    df.to_csv(HISTORY_FILE, mode="a", header=write_header, index=False)


def analysis_suite(
    ticker: str,
    market_df: pd.DataFrame,
    crossasset_rows: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, float]]:
    feat = engineer_features(market_df)
    if feat.empty:
        raise RuntimeError(f"Not enough feature history for {ticker}")

    results = technical_signals(feat)
    results.append(hmm_regime_signal(feat))
    results.append(markov_switch_signal(feat))
    results.append(gmm_regime_signal(feat))
    results.append(kmeans_regime_signal(feat))
    results.append(logistic_21d_signal(feat))
    rf_row, rf_top = random_forest_risk_signal(feat)
    results.append(rf_row)
    results.append(monte_carlo_signal(feat))
    if not crossasset_rows.empty:
        for _, r in crossasset_rows.iterrows():
            results.append(r.to_dict())
    results.append(extension_filter(feat))

    tbl = pd.DataFrame(results)
    score, bullish_count, bearish_count, neutral_count = weighted_score(results)

    total_weight = sum(MODEL_WEIGHTS.get(r["analysis"], 1.0) for r in results)
    confidence = min(abs(score) / max(total_weight, 1e-9), 1.0)
    raw_signal = signal_from_score(score, 0.75, -0.75)

    row = feat.iloc[-1]
    final_signal, persistence_note, change_date, change_price = persistence_filter(ticker, raw_signal, float(row["Close"]))

    hist_sig, hist_date, hist_px = compute_historical_composite_regime(ticker)
    if change_date is None:
        change_date = hist_date
    if change_price is None:
        change_price = hist_px if hist_px is not None else float(row["Close"])

    summary = {
        "ticker": ticker,
        "close": float(row["Close"]),
        "sma50": float(row["sma50"]),
        "sma200": float(row["sma200"]),
        "rsi14": float(row["rsi14"]),
        "atr14": float(row["atr14"]),
        "vol21annualized": float(row["vol21annualized"]),
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "neutral_count": neutral_count,
        "raw_score": float(score),
        "confidence": float(confidence),
        "raw_final_signal": raw_signal,
        "final_signal": final_signal,
        "persistence_note": persistence_note,
        "composite_change_date": change_date,
        "composite_change_price": float(change_price) if change_price is not None else None,
        "historical_composite_signal": hist_sig,
    }
    return tbl, summary, rf_top


def combine_market_confirmation(
    spy_summary: Dict[str, Any],
    qqq_summary: Dict[str, Any],
    amp_side: Optional[str],
) -> Dict[str, Any]:
    score = 0.0
    score += 1.0 * spy_summary["raw_score"]
    score += 1.15 * qqq_summary["raw_score"]

    market_signal = signal_from_score(score, 1.25, -1.25)

    persistent_votes = [spy_summary["final_signal"], qqq_summary["final_signal"]]
    if persistent_votes.count("bullish") >= 2:
        persistent_signal = "bullish"
    elif persistent_votes.count("bearish") >= 2:
        persistent_signal = "bearish"
    else:
        persistent_signal = "neutral"

    congruent = False
    if amp_side == "long" and market_signal == "bullish":
        congruent = True
    elif amp_side == "short" and market_signal == "bearish":
        congruent = True

    return {
        "market_signal": market_signal,
        "persistent_signal": persistent_signal,
        "score": float(score),
        "congruent_with_amp": congruent,
    }


# ---------------------------------------------------------------------
# AUTO-TRADE EXECUTION
# ---------------------------------------------------------------------
def maybe_execute_trade(
    apikey: str,
    strategy_id: int,
    best_row: ScrapedRow,
    desired_pos: Optional[ParsedPosition],
    confirmation: Dict[str, Any],
    current_positions: List[OpenPosition],
) -> None:
    if not desired_pos:
        print(YELLOW + "AUTO-TRADE: AMP is flat for the top system. No trade sent." + RESET)
        return

    if not AUTO_TRADE_ENABLED:
        print(YELLOW + "AUTO-TRADE: Disabled by AUTO_TRADE_ENABLED flag." + RESET)
        return

    product = best_row.product.upper()
    if product not in SUPPORTED_PRODUCTS:
        print(RED + f"AUTO-TRADE: Unsupported product {product}. No trade sent." + RESET)
        return

    if desired_pos.side == "long" and confirmation["market_signal"] != "bullish":
        print(
            YELLOW
            + "AUTO-TRADE: Long signal from AMP not confirmed by market regime. No trade sent."
            + RESET
        )
        return

    if desired_pos.side == "short" and confirmation["market_signal"] != "bearish":
        print(
            YELLOW
            + "AUTO-TRADE: Short signal from AMP not confirmed by market regime. No trade sent."
            + RESET
        )
        return

    if confirmation["persistent_signal"] != confirmation["market_signal"]:
        print(
            YELLOW
            + "AUTO-TRADE: Persistent overlay not aligned with current market signal. No trade sent."
            + RESET
        )
        return

    spy_conf = summaries_global["SPY"]["confidence"]
    qqq_conf = summaries_global["QQQ"]["confidence"]
    min_conf = min(spy_conf, qqq_conf)

    if min_conf < AGREEMENT_THRESHOLD:
        print(
            YELLOW
            + f"AUTO-TRADE: Confidence gate failed {min_conf:.1%} < {AGREEMENT_THRESHOLD:.1%}. No trade sent."
            + RESET
        )
        return

    target_symbol = SUPPORTED_PRODUCTS[product]

    existing = [p for p in current_positions if p.symbol == target_symbol]
    existing_pos = existing[0] if existing else None

    if existing_pos and existing_pos.side == desired_pos.side and existing_pos.qty == desired_pos.qty:
        print(
            YELLOW
            + f"AUTO-TRADE: Already in desired position {existing_pos.symbol} "
              f"{existing_pos.side} x {existing_pos.qty}. No new order sent."
            + RESET
        )
        return

    if existing_pos and (
        existing_pos.side != desired_pos.side or existing_pos.qty != desired_pos.qty
    ):
        print(
            YELLOW
            + f"AUTO-TRADE: Existing position differs ({existing_pos.symbol} "
              f"{existing_pos.side} x {existing_pos.qty}). Sending close order first."
            + RESET
        )
        close_payload = build_close_order(strategy_id, existing_pos)
        print("Close payload:", json.dumps(close_payload, ensure_ascii=False))
        if DRY_RUN:
            print(YELLOW + "DRY_RUN=1 - close order not sent." + RESET)
            return
        close_result = api4_post("/Strategies/NewStrategyOrder", apikey, close_payload)
        print("Close result:", json.dumps(close_result, ensure_ascii=False))
        print(
            YELLOW
            + "AUTO-TRADE: Position close sent. Re-run after C2 position updates "
              "before opening the new trade."
            + RESET
        )
        return

    parent_payload = build_parent_order_market_only(
        strategy_id=strategy_id,
        full_symbol=target_symbol,
        side=desired_pos.side,
        qty=desired_pos.qty,
    )
    print("Market entry payload:", json.dumps(parent_payload, ensure_ascii=False))

    if DRY_RUN:
        print(YELLOW + "DRY_RUN=1 - market entry order not sent." + RESET)
        return

    result = api4_post("/Strategies/NewStrategyOrder", apikey, parent_payload)
    print("Market entry result:", json.dumps(result, ensure_ascii=False))
    print(GREEN + "AUTO-TRADE: market entry order submitted (no bracket). Manage stop/target manually." + RESET)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    global summaries_global

    apikey = os.getenv("C2_API_KEY", "").strip()
    systemid_raw = os.getenv("C2_SYSTEM_ID", "").strip()
    now = datetime.now(timezone.utc).isoformat()

    print(
        f"{CYAN}AMP FUTURES CAPITAL REGIME TOOLBOX at {now} UTC{RESET}"
    )
    print(
        f"{CYAN}Dependencies: HMM={HMM_AVAILABLE}, Markov={MARKOV_AVAILABLE} | "
        f"Lookback={LOOKBACK_YEARS}y daily | Persistence={PERSISTENCE_DAYS} | "
        f"AutoTrade={AUTO_TRADE_ENABLED} DryRun={DRY_RUN} "
        f"AgreementThreshold={AGREEMENT_THRESHOLD * 100.0:.1f}%{RESET}"
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
            YELLOW
            + BOLD
            + f"NEW SIGNALS: AMP is FLAT for top system {best_row.system}, product {best_row.product}."
            + RESET
        )

    current_positions: List[OpenPosition] = []
    if apikey and systemid_raw:
        try:
            openpos_raw = get_open_positions(apikey, int(systemid_raw))
            current_positions = extract_supported_open_positions(openpos_raw)
        except Exception as e:
            print(YELLOW + f"Could not fetch C2 open positions: {e}" + RESET)

    market_data: Dict[str, pd.DataFrame] = {}
    for ticker in ALL_TICKERS:
        market_data[ticker] = download_market_data(ticker)

    crossasset_rows = cross_asset_signals(market_data)

    tables: Dict[str, pd.DataFrame] = {}
    summaries: Dict[str, Dict[str, Any]] = {}
    rf_features: Dict[str, Dict[str, float]] = {}

    for ticker in ["SPY", "QQQ"]:
        tbl, summary, rftop = analysis_suite(
            ticker,
            market_data[ticker],
            crossasset_rows,
        )
        tables[ticker] = tbl
        summaries[ticker] = summary
        rf_features[ticker] = rftop

    summaries_global = summaries
    confirmation = combine_market_confirmation(
        summaries["SPY"],
        summaries["QQQ"],
        desired_pos.side if desired_pos else None,
    )

    trade_etf = ETF_MAP.get(best_row.product, "QQQ")

    print_block(
        "SCRAPED AMP SIGNAL",
        [
            f"Top system: {best_row.system}",
            f"Developer: {best_row.developer or 'N/A'}",
            f"Product: {best_row.product}",
            f"Scraped current position: {best_row.current_position}",
            f"Nearest order: {best_row.nearest_order}",
            (
                f"AMP interpreted side: {desired_pos.side.upper()} x {desired_pos.qty}"
                if desired_pos
                else "AMP interpreted side: FLAT / NO ACTIVE SIGNAL"
            ),
            f"ETF proxy used for traded product: {trade_etf}",
        ],
        CYAN,
    )

    if current_positions:
        pos_lines = [
            f"{p.symbol} {p.side.upper()} x {p.qty} @ {p.entry_price}"
            for p in current_positions
        ]
    else:
        pos_lines = ["No supported open MES/MNQ positions detected or C2 credentials unavailable."]

    print_block("OPEN POSITIONS", pos_lines, GREEN)

    print_block(
        "RISK NOTES",
        atr_risk_text(
            best_row.product,
            current_positions,
            best_row.nearest_order,
            market_data[trade_etf],
        ),
        MAGENTA,
    )

    for ticker in ["SPY", "QQQ"]:
        s = summaries[ticker]
        base_signal = s["final_signal"] if s["final_signal"] != "neutral" else s["raw_final_signal"]
        color = color_for_signal(base_signal)
        compsince = "N/A"
        days_age = None
        if s["composite_change_date"]:
            price_txt = f"{s['composite_change_price']:.2f}" if s["composite_change_price"] is not None else "N/A"
            compsince = f"{s['composite_change_date']} at {price_txt}"
            days_age = days_since(s["composite_change_date"])
        age_text = f"{days_age} days since last change" if days_age is not None else "Age unknown"

        print_block(
            f"{ticker} SUMMARY",
            [
                f"Close: {s['close']:.2f} | SMA50 {s['sma50']:.2f} / SMA200 {s['sma200']:.2f}",
                f"RSI14 {s['rsi14']:.2f} | ATR14 {s['atr14']:.2f} | 21d Annualized Vol {s['vol21annualized']:.2%}",
                f"Bullish: {s['bullish_count']} | Bearish: {s['bearish_count']} | Neutral: {s['neutral_count']}",
                f"Weighted score: {s['raw_score']:.2f} | Confidence {s['confidence']:.1%}",
                f"Raw regime: {s['raw_final_signal'].upper()} | Final after persistence: {s['final_signal'].upper()}",
                f"Composite regime in force since: {compsince}",
                f"Composite regime age: {age_text}",
                f"Persistence note: {s['persistence_note']}",
            ],
            color,
        )

        for _, row in tables[ticker].iterrows():
            c = color_for_signal(row["signal"])
            print(
                f"{c}{str(row['analysis']):<20} | {str(row['signal']).upper():7s} | "
                f"{row['value']} | {row['note']}{RESET}"
            )

        if rf_features.get(ticker):
            print(BLUE + f"Top RF features for {ticker}: {rf_features[ticker]}" + RESET)

    final_color = color_for_signal(confirmation["market_signal"])
    amp_align = "YES" if confirmation["congruent_with_amp"] else "NO"

    print_block(
        "FINAL RECOMMENDATION",
        [
            f"SPY final: {summaries['SPY']['final_signal'].upper()} | "
            f"raw score {summaries['SPY']['raw_score']:.2f} | "
            f"confidence {summaries['SPY']['confidence']:.1%}",
            f"QQQ final: {summaries['QQQ']['final_signal'].upper()} | "
            f"raw score {summaries['QQQ']['raw_score']:.2f} | "
            f"confidence {summaries['QQQ']['confidence']:.1%}",
            f"Combined weighted ETF confirmation: {confirmation['market_signal'].upper()} | "
            f"aggregate score {confirmation['score']:.2f}",
            f"Persistent regime overlay: {confirmation['persistent_signal'].upper()}",
            f"Congruent with AMP scraped side "
            f"({desired_pos.side.upper() if desired_pos else 'FLAT/NA'}): {amp_align}",
            f"Top system: {best_row.system} | Nearest order: {best_row.nearest_order}",
        ],
        final_color,
    )

    if desired_pos and desired_pos.side == "long" and confirmation["market_signal"] == "bullish":
        print(GREEN + BOLD + "ACTION BIAS: BUY / LONG SWING SETUP CONFIRMED" + RESET)
    elif desired_pos and desired_pos.side == "short" and confirmation["market_signal"] == "bearish":
        print(RED + BOLD + "ACTION BIAS: SELL / SHORT SWING SETUP CONFIRMED" + RESET)
    else:
        print(
            YELLOW
            + BOLD
            + "ACTION BIAS: MIXED / NO STRONG CONFIRMATION or AMP FLAT"
            + RESET
        )

    if apikey and systemid_raw:
        try:
            maybe_execute_trade(
                apikey=apikey,
                strategy_id=int(systemid_raw),
                best_row=best_row,
                desired_pos=desired_pos,
                confirmation=confirmation,
                current_positions=current_positions,
            )
        except Exception as e:
            print(RED + f"AUTO-TRADE ERROR: {e}" + RESET)
    else:
        print(YELLOW + "AUTO-TRADE: C2 credentials unavailable." + RESET)

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
                "amp_qty": desired_pos.qty if desired_pos else 0,
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
                "persistent_market_signal": confirmation["persistent_signal"],
                "congruent_with_amp": confirmation["congruent_with_amp"],
                "auto_trade_enabled": AUTO_TRADE_ENABLED,
                "dry_run": DRY_RUN,
            }
        )

    save_run(run_rows)
    print(BLUE + f"Logged run history to: {HISTORY_FILE}" + RESET)


if __name__ == "__main__":
    main()

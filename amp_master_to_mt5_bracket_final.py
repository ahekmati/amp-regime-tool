#!/usr/bin/env python3
import csv
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

if load_dotenv:
    load_dotenv()

AMP_URL = os.getenv("AMP_URL", "https://ampfutures.isystems.com/Systems/TopStrategies")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
TOP_N = int(os.getenv("TOP_N", "25"))
DATA_DIR = Path(os.getenv("DATA_DIR", "./amp_data"))
SNAPSHOT_DIR = DATA_DIR / "snapshots"
TRADES_CSV = DATA_DIR / "amp_topstrategies_trades.csv"
ALERTS_CSV = DATA_DIR / "amp_topstrategies_alerts.csv"
ORDERS_CSV = DATA_DIR / "mt5_orders.csv"
LATEST_JSON = DATA_DIR / "latest_snapshot.json"
STATE_JSON = DATA_DIR / "execution_state.json"

USER_AGENT = os.getenv(
    "USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
)

DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
MAGIC = int(os.getenv("MT5_MAGIC", "9041301"))
DEVIATION = int(os.getenv("MT5_DEVIATION", "30"))
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "240"))
MAX_DAILY_ORDERS_PER_SYMBOL = int(os.getenv("MAX_DAILY_ORDERS_PER_SYMBOL", "2"))

MT5_LOGIN = os.getenv("MT5_LOGIN")
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
MT5_SERVER = os.getenv("MT5_SERVER")
MT5_PATH = os.getenv("MT5_PATH")

MASTER_NQ_SYSTEM = os.getenv("MASTER_NQ_SYSTEM", "").strip()
MASTER_ES_SYSTEM = os.getenv("MASTER_ES_SYSTEM", "").strip()

MT5_SYMBOL_NQ = os.getenv("MT5_SYMBOL_NQ", "MNQM26")
MT5_SYMBOL_ES = os.getenv("MT5_SYMBOL_ES", "MESM26")

LOT_NQ = float(os.getenv("LOT_NQ", "1.0"))
LOT_ES = float(os.getenv("LOT_ES", "1.0"))

MNQ_SL_POINTS = float(os.getenv("MNQ_SL_POINTS", "200"))
MNQ_TP_POINTS = float(os.getenv("MNQ_TP_POINTS", "2200"))
MES_SL_POINTS = float(os.getenv("MES_SL_POINTS", "80"))
MES_TP_POINTS = float(os.getenv("MES_TP_POINTS", "400"))


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_html() -> str:
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(AMP_URL, headers=headers, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.text


def money_to_float(s: str) -> Optional[float]:
    if s is None:
        return None
    s = s.replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except Exception:
        return None


def normalize_text(x: str) -> str:
    return re.sub(r"\s+", " ", (x or "").strip())


def stable_row_id(system: str, product: str, developer: str) -> str:
    base = f"{normalize_text(system)}|{normalize_text(product)}|{normalize_text(developer)}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]


def parse_current_session(html: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="tableCurrentSession")
    if not table:
        raise RuntimeError("Could not find tableCurrentSession in AMP HTML")
    body = table.find("tbody")
    if not body:
        raise RuntimeError("tableCurrentSession has no tbody")

    rows = []
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

        rank_match = re.search(r"#(\d+)", rank_td.get_text(" ", strip=True))
        if not rank_match:
            continue

        rank = int(rank_match.group(1))
        if rank > TOP_N:
            continue

        system = normalize_text(system_td.get_text(" ", strip=True))
        product = normalize_text(product_td.get_text(" ", strip=True)).upper()
        developer = normalize_text(developer_td.get_text(" ", strip=True) if developer_td else "")
        pnl = money_to_float(pnl_td.get_text(" ", strip=True))
        current_position = normalize_text(pos_td.get_text(" ", strip=True))
        nearest_order = normalize_text(nearest_td.get_text(" ", strip=True))

        rows.append(
            {
                "row_id": stable_row_id(system, product, developer),
                "rank": rank,
                "system": system,
                "product": product,
                "developer": developer,
                "pnl": pnl,
                "current_position": current_position,
                "nearest_order": nearest_order,
            }
        )
    return rows


def load_previous_snapshot() -> List[Dict]:
    if not LATEST_JSON.exists():
        return []
    try:
        return json.loads(LATEST_JSON.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_snapshot(rows: List[Dict], run_ts: str) -> Path:
    stamp = run_ts.replace(":", "-")
    out = SNAPSHOT_DIR / f"snapshot_{stamp}.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    LATEST_JSON.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def index_by_row_id(rows: List[Dict]) -> Dict[str, Dict]:
    return {str(r["row_id"]): r for r in rows}


def detect_alerts(prev_rows: List[Dict], curr_rows: List[Dict], run_ts: str) -> List[Dict]:
    prev = index_by_row_id(prev_rows)
    curr = index_by_row_id(curr_rows)
    alerts = []

    for row_id, now_row in curr.items():
        old_row = prev.get(row_id)
        if old_row is None:
            alerts.append(
                {
                    "timestamp_utc": run_ts,
                    "alert_type": "new_system_in_top_list",
                    "row_id": row_id,
                    "system": now_row["system"],
                    "product": now_row["product"],
                    "developer": now_row["developer"],
                    "old_rank": None,
                    "new_rank": now_row["rank"],
                    "old_position": None,
                    "new_position": now_row["current_position"],
                    "old_nearest_order": None,
                    "new_nearest_order": now_row["nearest_order"],
                    "details": "System appeared in tracked top list.",
                }
            )
            continue

        if normalize_text(old_row.get("current_position", "")) != normalize_text(now_row.get("current_position", "")):
            alerts.append(
                {
                    "timestamp_utc": run_ts,
                    "alert_type": "position_change",
                    "row_id": row_id,
                    "system": now_row["system"],
                    "product": now_row["product"],
                    "developer": now_row["developer"],
                    "old_rank": old_row.get("rank"),
                    "new_rank": now_row.get("rank"),
                    "old_position": old_row.get("current_position"),
                    "new_position": now_row.get("current_position"),
                    "old_nearest_order": old_row.get("nearest_order"),
                    "new_nearest_order": now_row.get("nearest_order"),
                    "details": "Current position differs from previous snapshot.",
                }
            )
    return alerts


def append_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    if not rows:
        return
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def product_root(product: str) -> Optional[str]:
    p = normalize_text(product).upper()
    for root in ("MNQ", "NQ", "MES", "ES"):
        if root in p:
            return root
    return None


def parse_position_text(pos: str) -> str:
    p = normalize_text(pos).upper()
    if any(x in p for x in ["LONG", "BUY"]):
        return "long"
    if any(x in p for x in ["SHORT", "SELL"]):
        return "short"
    if any(x in p for x in ["FLAT", "NONE", "EXIT", "CLOSE"]):
        return "flat"
    return "unknown"


def load_state() -> Dict:
    if not STATE_JSON.exists():
        return {"symbols": {}}
    try:
        return json.loads(STATE_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {"symbols": {}}


def save_state(state: Dict) -> None:
    STATE_JSON.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def get_symbol_state(state: Dict, symbol: str) -> Dict:
    state.setdefault("symbols", {})
    state["symbols"].setdefault(
        symbol,
        {
            "last_target": None,
            "last_action_time_utc": None,
            "daily_order_count_date": None,
            "daily_order_count": 0,
        },
    )
    return state["symbols"][symbol]


def reset_daily_counter_if_needed(sym_state: Dict, today_utc: str) -> None:
    if sym_state.get("daily_order_count_date") != today_utc:
        sym_state["daily_order_count_date"] = today_utc
        sym_state["daily_order_count"] = 0


def under_cooldown(sym_state: Dict, now: datetime) -> bool:
    ts = sym_state.get("last_action_time_utc")
    if not ts:
        return False
    try:
        last = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return False
    return now < last + timedelta(minutes=COOLDOWN_MINUTES)


def init_mt5() -> None:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package not installed. pip install MetaTrader5")
    if MT5_PATH:
        ok = mt5.initialize(
            path=MT5_PATH,
            login=int(MT5_LOGIN) if MT5_LOGIN else None,
            password=MT5_PASSWORD,
            server=MT5_SERVER,
        )
    else:
        ok = mt5.initialize(
            login=int(MT5_LOGIN) if MT5_LOGIN else None,
            password=MT5_PASSWORD,
            server=MT5_SERVER,
        )
    if not ok:
        raise RuntimeError(f"mt5.initialize failed: {mt5.last_error()}")


def shutdown_mt5() -> None:
    if mt5 is not None:
        mt5.shutdown()


def ensure_symbol(symbol: str) -> None:
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"MT5 symbol not found: {symbol}")
    if not info.visible and not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Could not select symbol {symbol}")


def current_positions_by_symbol(symbol: str) -> List:
    positions = mt5.positions_get(symbol=symbol)
    return list(positions) if positions else []


def symbol_bracket_points(symbol: str) -> Tuple[float, float]:
    if symbol == MT5_SYMBOL_NQ:
        return MNQ_SL_POINTS, MNQ_TP_POINTS
    if symbol == MT5_SYMBOL_ES:
        return MES_SL_POINTS, MES_TP_POINTS
    raise RuntimeError(f"No bracket config for symbol {symbol}")


def bracket_prices(symbol: str, side: str, entry_price: float) -> Tuple[float, float]:
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"symbol_info unavailable for {symbol}")

    sl_points, tp_points = symbol_bracket_points(symbol)

    if side == "buy":
        sl = entry_price - sl_points
        tp = entry_price + tp_points
    elif side == "sell":
        sl = entry_price + sl_points
        tp = entry_price - tp_points
    else:
        raise RuntimeError(f"Invalid side for brackets: {side}")

    return round(sl, info.digits), round(tp, info.digits)


def send_market_order(symbol: str, side: str, volume: float, comment: str) -> Tuple[bool, str, Optional[object]]:
    ensure_symbol(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return False, f"No tick for {symbol}", None

    if side == "buy":
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
    elif side == "sell":
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
    else:
        return False, f"Invalid side {side}", None

    sl, tp = bracket_prices(symbol, side, price)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": DEVIATION,
        "magic": MAGIC,
        "comment": comment[:31],
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }

    result = mt5.order_send(request)
    if result is None:
        return False, f"order_send returned None: {mt5.last_error()}", None
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return False, f"order_send failed retcode={result.retcode}", result
    return True, f"ok sl={sl} tp={tp}", result


def close_position(position, comment: str) -> Tuple[bool, str, Optional[object]]:
    symbol = position.symbol
    volume = position.volume
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return False, f"No tick for {symbol}", None

    if position.type == mt5.POSITION_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
    else:
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "position": position.ticket,
        "price": price,
        "deviation": DEVIATION,
        "magic": MAGIC,
        "comment": comment[:31],
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }

    result = mt5.order_send(request)
    if result is None:
        return False, f"close order_send returned None: {mt5.last_error()}", None
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return False, f"close failed retcode={result.retcode}", result
    return True, "ok", result


def sync_symbol_to_target(symbol: str, target: str, volume: float, source_tag: str) -> List[Dict]:
    actions = []
    positions = current_positions_by_symbol(symbol)
    longs = [p for p in positions if p.type == mt5.POSITION_TYPE_BUY]
    shorts = [p for p in positions if p.type == mt5.POSITION_TYPE_SELL]

    def log(action: str, success: bool, message: str, result=None):
        actions.append(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "target": target,
                "volume": volume,
                "action": action,
                "success": success,
                "message": message,
                "retcode": getattr(result, "retcode", None) if result else None,
                "order": getattr(result, "order", None) if result else None,
                "deal": getattr(result, "deal", None) if result else None,
                "source": source_tag,
            }
        )

    if DRY_RUN:
        if target in ("long", "short"):
            side = "buy" if target == "long" else "sell"
            ensure_symbol(symbol)
            tick = mt5.symbol_info_tick(symbol) if mt5 else None
            ref_price = tick.ask if tick and side == "buy" else (tick.bid if tick else 0)
            if target == "long" and longs:
                log("skip_existing_long", True, "Existing long found; would do nothing")
            elif target == "short" and shorts:
                log("skip_existing_short", True, "Existing short found; would do nothing")
            elif ref_price:
                sl, tp = bracket_prices(symbol, side, ref_price)
                log("dry_run_sync", True, f"Would sync {symbol} to {target} with sl={sl} tp={tp}")
            else:
                log("dry_run_sync", True, f"Would sync {symbol} to {target}")
        elif target == "flat":
            log("dry_run_flat", True, "Would close only because explicit flat was scraped")
        else:
            log("skip", False, f"Unknown target {target}")
        return actions

    if target == "flat":
        for p in positions:
            ok, msg, res = close_position(p, f"AMP flat {source_tag}")
            log("close", ok, msg, res)
        if not positions:
            log("skip_no_position", True, "Explicit flat scraped, but no open positions to close")
        return actions

    if target == "long":
        if longs:
            log("skip_existing_long", True, "Existing long found; doing nothing")
            return actions
        for p in shorts:
            ok, msg, res = close_position(p, f"AMP flip->long {source_tag}")
            log("close_short", ok, msg, res)
        remaining = current_positions_by_symbol(symbol)
        remaining_longs = [p for p in remaining if p.type == mt5.POSITION_TYPE_BUY]
        remaining_shorts = [p for p in remaining if p.type == mt5.POSITION_TYPE_SELL]
        if remaining_longs:
            log("skip_existing_long", True, "Long already exists after refresh; doing nothing")
            return actions
        if remaining_shorts:
            log("skip_unclosed_short", False, "Short still open after close attempt; not opening long")
            return actions
        ok, msg, res = send_market_order(symbol, "buy", volume, f"AMP long {source_tag}")
        log("buy", ok, msg, res)
        return actions

    if target == "short":
        if shorts:
            log("skip_existing_short", True, "Existing short found; doing nothing")
            return actions
        for p in longs:
            ok, msg, res = close_position(p, f"AMP flip->short {source_tag}")
            log("close_long", ok, msg, res)
        remaining = current_positions_by_symbol(symbol)
        remaining_shorts = [p for p in remaining if p.type == mt5.POSITION_TYPE_SELL]
        remaining_longs = [p for p in remaining if p.type == mt5.POSITION_TYPE_BUY]
        if remaining_shorts:
            log("skip_existing_short", True, "Short already exists after refresh; doing nothing")
            return actions
        if remaining_longs:
            log("skip_unclosed_long", False, "Long still open after close attempt; not opening short")
            return actions
        ok, msg, res = send_market_order(symbol, "sell", volume, f"AMP short {source_tag}")
        log("sell", ok, msg, res)
        return actions

    log("skip", False, f"Unknown target {target}")
    return actions


def pick_master_rows(rows: List[Dict]) -> Dict[str, Optional[Dict]]:
    selected = {"NQ": None, "ES": None}
    for r in rows:
        root = product_root(r["product"])
        if root in ("NQ", "MNQ") and MASTER_NQ_SYSTEM and normalize_text(r["system"]) == normalize_text(MASTER_NQ_SYSTEM):
            selected["NQ"] = r
        elif root in ("ES", "MES") and MASTER_ES_SYSTEM and normalize_text(r["system"]) == normalize_text(MASTER_ES_SYSTEM):
            selected["ES"] = r
    return selected


def main() -> None:
    ensure_dirs()
    run_now = datetime.now(timezone.utc)
    run_ts = run_now.isoformat()
    today_utc = run_now.date().isoformat()

    html = fetch_html()
    current_rows = parse_current_session(html)
    previous_rows = load_previous_snapshot()
    alerts = detect_alerts(previous_rows, current_rows, run_ts)

    trade_rows = [
        {
            "timestamp_utc": run_ts,
            "row_id": r["row_id"],
            "rank": r["rank"],
            "system": r["system"],
            "product": r["product"],
            "developer": r["developer"],
            "pnl": r["pnl"],
            "current_position": r["current_position"],
            "nearest_order": r["nearest_order"],
        }
        for r in current_rows
    ]

    append_csv(
        TRADES_CSV,
        trade_rows,
        ["timestamp_utc", "row_id", "rank", "system", "product", "developer", "pnl", "current_position", "nearest_order"],
    )

    append_csv(
        ALERTS_CSV,
        alerts,
        ["timestamp_utc", "alert_type", "row_id", "system", "product", "developer", "old_rank", "new_rank",
         "old_position", "new_position", "old_nearest_order", "new_nearest_order", "details"],
    )

    snapshot_path = save_snapshot(current_rows, run_ts)
    state = load_state()
    master_rows = pick_master_rows(current_rows)
    actionable = []
    order_logs = []

    for family, r in master_rows.items():
        if r is None:
            symbol = MT5_SYMBOL_NQ if family == "NQ" else MT5_SYMBOL_ES
            order_logs.append(
                {
                    "timestamp_utc": run_ts,
                    "symbol": symbol,
                    "target": None,
                    "volume": LOT_NQ if family == "NQ" else LOT_ES,
                    "action": "master_not_found",
                    "success": True,
                    "message": "Configured master system not found in scrape; no action taken",
                    "retcode": None,
                    "order": None,
                    "deal": None,
                    "source": family,
                }
            )
            continue

        target = parse_position_text(r["current_position"])
        if target == "unknown":
            symbol = MT5_SYMBOL_NQ if family == "NQ" else MT5_SYMBOL_ES
            order_logs.append(
                {
                    "timestamp_utc": run_ts,
                    "symbol": symbol,
                    "target": target,
                    "volume": LOT_NQ if family == "NQ" else LOT_ES,
                    "action": "skip_unknown_target",
                    "success": False,
                    "message": f"Could not parse current_position='{r['current_position']}'",
                    "retcode": None,
                    "order": None,
                    "deal": None,
                    "source": r["system"],
                }
            )
            continue

        if family == "NQ":
            mt5_symbol = MT5_SYMBOL_NQ
            volume = LOT_NQ
        else:
            mt5_symbol = MT5_SYMBOL_ES
            volume = LOT_ES

        sym_state = get_symbol_state(state, mt5_symbol)
        reset_daily_counter_if_needed(sym_state, today_utc)

        if sym_state["daily_order_count"] >= MAX_DAILY_ORDERS_PER_SYMBOL:
            order_logs.append(
                {
                    "timestamp_utc": run_ts,
                    "symbol": mt5_symbol,
                    "target": target,
                    "volume": volume,
                    "action": "blocked_daily_limit",
                    "success": False,
                    "message": "Max daily orders reached",
                    "retcode": None,
                    "order": None,
                    "deal": None,
                    "source": r["system"],
                }
            )
            continue

        if under_cooldown(sym_state, run_now):
            order_logs.append(
                {
                    "timestamp_utc": run_ts,
                    "symbol": mt5_symbol,
                    "target": target,
                    "volume": volume,
                    "action": "blocked_cooldown",
                    "success": False,
                    "message": f"Cooldown active ({COOLDOWN_MINUTES}m)",
                    "retcode": None,
                    "order": None,
                    "deal": None,
                    "source": r["system"],
                }
            )
            continue

        if sym_state.get("last_target") == target:
            order_logs.append(
                {
                    "timestamp_utc": run_ts,
                    "symbol": mt5_symbol,
                    "target": target,
                    "volume": volume,
                    "action": "skip_same_target",
                    "success": True,
                    "message": "Target unchanged from last applied state",
                    "retcode": None,
                    "order": None,
                    "deal": None,
                    "source": r["system"],
                }
            )
            continue

        actionable.append(
            {
                "mt5_symbol": mt5_symbol,
                "target": target,
                "volume": volume,
                "source_tag": f"{r['system']}|{r['product']}|rank{r['rank']}",
            }
        )

    if actionable and mt5 is not None:
        init_mt5()

    try:
        for item in actionable:
            logs = sync_symbol_to_target(
                symbol=item["mt5_symbol"],
                target=item["target"],
                volume=item["volume"],
                source_tag=item["source_tag"],
            )
            order_logs.extend(logs)

            meaningful_success = any(
                x["success"] for x in logs
                if x["action"] not in (
                    "skip_existing_long",
                    "skip_existing_short",
                    "skip_no_position",
                    "master_not_found",
                    "skip_same_target",
                    "dry_run_sync",
                    "dry_run_flat",
                )
            )
            if DRY_RUN or meaningful_success:
                sym_state = get_symbol_state(state, item["mt5_symbol"])
                sym_state["last_target"] = item["target"]
                sym_state["last_action_time_utc"] = run_ts
                sym_state["daily_order_count"] += 1
    finally:
        if actionable and mt5 is not None:
            shutdown_mt5()

    save_state(state)

    append_csv(
        ORDERS_CSV,
        order_logs,
        ["timestamp_utc", "symbol", "target", "volume", "action", "success", "message", "retcode", "order", "deal", "source"],
    )

    print(f"Run timestamp UTC: {run_ts}")
    print(f"Rows scraped: {len(current_rows)}")
    print(f"Alerts found: {len(alerts)}")
    print(f"Master NQ found: {master_rows['NQ'] is not None}")
    print(f"Master ES found: {master_rows['ES'] is not None}")
    print(f"Actionable: {len(actionable)}")
    print(f"Dry run: {DRY_RUN}")
    print(f"Snapshot saved: {snapshot_path}")
    print(f"State saved: {STATE_JSON}")
    print(f"Orders log: {ORDERS_CSV}")

    print("\nMASTER STATUS:")
    for family, r in master_rows.items():
        if r is None:
            print(f"- {family}: NOT FOUND -> no action")
        else:
            print(f"- {family}: {r['system']} | {r['product']} | pos={r['current_position']} | rank={r['rank']} | pnl={r['pnl']}")

    if order_logs:
        print("\nORDER LOGS:")
        for o in order_logs:
            print(f"- {o['symbol']} {o['action']} target={o['target']} success={o['success']} msg={o['message']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise

#!/usr/bin/env python3
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any

import requests
from bs4 import BeautifulSoup

AMP_URL = "https://ampfutures.isystems.com/Systems/TopStrategies"
C2_API4_BASE = "https://api4-general.collective2.com"
REQUEST_TIMEOUT = 30
TOP_N = int(os.getenv("TOP_N", "10"))
DRY_RUN = os.getenv("DRY_RUN", "0").strip() == "1"

MNQ_SYMBOL = os.getenv("MNQ_SYMBOL", "@MNQM6")
HISTORY_DIR = Path(os.getenv("HISTORY_DIR", "./history"))
TRADE_LOG_FILE = HISTORY_DIR / "mnq_copier_log.csv"

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


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


def to_float_safe(v: Any) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def print_green(line: str) -> None:
    print(GREEN + line + RESET)


def print_green_bold(line: str) -> None:
    print(GREEN + BOLD + line + RESET)


def print_yellow(line: str) -> None:
    print(YELLOW + line + RESET)


def print_red(line: str) -> None:
    print(RED + line + RESET)


def print_cyan(line: str) -> None:
    print(CYAN + line + RESET)


def normalize_text(x: str) -> str:
    return re.sub(r"\s+", " ", (x or "").strip())


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


# --- consensus helpers (NQ / ES / YM) ---------------------------------


def product_root(product: str, system: str = "") -> Optional[str]:
    p = normalize_text(product).upper()
    s = normalize_text(system).upper()
    combined = f"{p} | {s}"

    if "MICRO" in combined and "NASDAQ" in combined:
        return "MNQ"
    if "E-MINI" in combined and "NASDAQ" in combined:
        return "NQ"

    for root in ("MNQ", "NQ", "MES", "ES", "MYM", "YM"):
        if root in p or root in s:
            return root

    if "NASDAQ" in combined:
        return "NQ"
    return None


def parse_position_text(pos: str) -> str:
    p = normalize_text(pos).upper()

    if not p:
        return "unknown"

    if any(x in p for x in ["LONG", "BUY"]):
        return "long"
    if any(x in p for x in ["SHORT", "SELL"]):
        return "short"
    if any(x in p for x in ["FLAT", "NONE", "EXIT", "CLOSE", "NO POSITION", "SQUARE"]):
        return "flat"

    if re.search(r"(^|[\s(])L(?:ONG)?([\s@0-9]|$)", p):
        return "long"
    if re.search(r"(^|[\s(])S(?:HORT)?([\s@0-9]|$)", p):
        return "short"

    return "unknown"


def count_directional_consensus(rows: List[ScrapedRow]) -> Dict[str, Any]:
    long_count = 0
    short_count = 0
    qualifying_rows: List[str] = []

    for r in rows:
        root = product_root(r.product, r.system)
        pos = parse_position_text(r.current_position)

        if root in ("NQ", "MNQ", "ES", "MES", "YM", "MYM"):
            if pos == "long":
                long_count += 1
                qualifying_rows.append(
                    f"rank #{r.rank} | {root} | LONG | {r.system}"
                )
            elif pos == "short":
                short_count += 1
                qualifying_rows.append(
                    f"rank #{r.rank} | {root} | SHORT | {r.system}"
                )

    return {
        "long_count": long_count,
        "short_count": short_count,
        "max_same_direction": max(long_count, short_count),
        "qualifying_count": long_count + short_count,
        "qualifying_rows": qualifying_rows,
    }


# ----------------------------------------------------------------------


def pick_best_nq(rows: List[ScrapedRow]) -> Optional[ScrapedRow]:
    nq_rows = [r for r in rows if r.product.upper() == "NQ" and r.rank <= TOP_N]
    if not nq_rows:
        return None
    nq_rows.sort(key=lambda x: (x.rank, -x.pnl))
    return nq_rows[0]


def parse_direction_and_size(text: str) -> Optional[ParsedPosition]:
    text = str(text).strip()
    if text in {"", "--", "-", "Flat", "FLAT", "flat"}:
        return None

    m = re.match(r"^(Long|Short)\s+(\d+)\s*@", text, flags=re.I)
    if not m:
        return None

    return ParsedPosition(side=m.group(1).lower(), qty=int(m.group(2)))


def api4_get(path: str, apikey: str, params: Dict[str, Any]) -> dict:
    url = f"{C2_API4_BASE}{path}"
    headers = {
        "Authorization": f"Bearer {apikey}",
        "Content-Type": "application/json",
    }
    r = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def api4_post(path: str, apikey: str, payload: dict) -> dict:
    url = f"{C2_API4_BASE}{path}"
    headers = {
        "Authorization": f"Bearer {apikey}",
        "Content-Type": "application/json",
    }
    r = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_open_positions(apikey: str, strategy_id: int) -> dict:
    return api4_get(
        "/Strategies/GetStrategyOpenPositions",
        apikey,
        {"StrategyIds": str(strategy_id)},
    )


def extract_supported_open_positions(open_positions: dict) -> List[OpenPosition]:
    results = open_positions.get("Results", [])
    extracted: List[OpenPosition] = []

    for p in results:
        sym = p.get("C2Symbol", {}).get("FullSymbol")
        qty = p.get("Quantity")

        if sym != MNQ_SYMBOL:
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


def wait_for_fill(
    apikey: str,
    strategy_id: int,
    symbol: str,
    expected_side: str,
    expected_qty: int,
    timeout_seconds: int = 45,
    poll_seconds: int = 3,
) -> Optional[OpenPosition]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            raw = get_open_positions(apikey, strategy_id)
            positions = extract_supported_open_positions(raw)
            for p in positions:
                if p.symbol == symbol and p.side == expected_side and p.qty == expected_qty:
                    return p
        except Exception:
            pass
        time.sleep(poll_seconds)
    return None


def log_event(row: Dict[str, Any]) -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not TRADE_LOG_FILE.exists()

    fieldnames = [
        "timestamp_utc",
        "strategy_name",
        "developer",
        "rank",
        "product",
        "scraped_current_position",
        "scraped_nearest_order",
        "desired_side",
        "desired_qty",
        "mnq_symbol",
        "action",
        "status",
        "fill_price",
        "payload_json",
        "response_json",
        "note",
    ]

    with TRADE_LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    apikey = os.getenv("C2_API_KEY", "").strip()
    systemid_raw = os.getenv("C2_SYSTEM_ID", "").strip()
    now = datetime.now(timezone.utc).isoformat()

    print_cyan(BOLD + f"AMP NQ -> MNQ copier started at {now}" + RESET)
    print_cyan(f"AMP_URL={AMP_URL} | TOP_N={TOP_N} | MNQ_SYMBOL={MNQ_SYMBOL} | DRY_RUN={DRY_RUN}")

    html = fetch_amp_html()
    print_green("Fetched AMP current session page successfully.")

    rows = parse_current_session(html)
    print_green(f"Parsed {len(rows)} rows from AMP table within top {TOP_N} ranks.")

    if not rows:
        raise RuntimeError("No rows parsed from AMP current session table")

    # Consensus check across NQ / ES / YM family
    consensus = count_directional_consensus(rows)
    print_green_bold("Consensus check across NQ / ES / YM family")
    print_green(
        f"Found qualifying directional signals: {consensus['qualifying_count']} "
        f"(long={consensus['long_count']}, short={consensus['short_count']})"
    )
    if consensus["qualifying_rows"]:
        print_green("Qualifying rows:")
        for line in consensus["qualifying_rows"]:
            print_green(f"  {line}")
    else:
        print_yellow("No qualifying NQ/ES/YM rows with long/short direction were found.")

    if consensus["max_same_direction"] >= 3:
        dominant = "LONG" if consensus["long_count"] >= 3 else "SHORT"
        print_green_bold(
            f"Consensus PASSED: at least 3 symbols found in the same direction -> {dominant}"
        )
    else:
        print_yellow(
            "Consensus FAILED: fewer than 3 NQ/ES/YM-family signals are aligned "
            "in the same direction. No MNQ trade will be sent."
        )

    best_row = pick_best_nq(rows)
    if not best_row:
        raise RuntimeError("No NQ strategy found inside AMP top rows")

    desired_pos = parse_direction_and_size(best_row.current_position)

    print_green_bold("SCRAPED NQ STRATEGY")
    print_green(f"Strategy: {best_row.system}")
    print_green(f"Developer: {best_row.developer or 'N/A'}")
    print_green(f"Rank: {best_row.rank}")
    print_green(f"Product: {best_row.product}")
    print_green(f"Current position: {best_row.current_position}")
    print_green(f"Nearest order: {best_row.nearest_order}")
    print_green(f"Parsed target side/qty: {desired_pos.side if desired_pos else 'N/A'} "
                f"{desired_pos.qty if desired_pos else ''}")

    # Early exit conditions
    if consensus["max_same_direction"] < 3:
        log_event(
            {
                "timestamp_utc": now,
                "strategy_name": best_row.system,
                "developer": best_row.developer,
                "rank": best_row.rank,
                "product": best_row.product,
                "scraped_current_position": best_row.current_position,
                "scraped_nearest_order": best_row.nearest_order,
                "desired_side": desired_pos.side if desired_pos else "",
                "desired_qty": desired_pos.qty if desired_pos else "",
                "mnq_symbol": MNQ_SYMBOL,
                "action": "skip_consensus_failed",
                "status": "ignored",
                "fill_price": "",
                "payload_json": "",
                "response_json": "",
                "note": "3-symbol same-direction consensus not satisfied",
            }
        )
        print_yellow("MT5 TRADE DECISION: DO NOT SEND ORDER (consensus condition failed).")
        print_cyan(f"Log file: {TRADE_LOG_FILE}")
        return

    if not desired_pos:
        print_yellow("AMP signal is flat or unparsable. Ignoring.")
        log_event(
            {
                "timestamp_utc": now,
                "strategy_name": best_row.system,
                "developer": best_row.developer,
                "rank": best_row.rank,
                "product": best_row.product,
                "scraped_current_position": best_row.current_position,
                "scraped_nearest_order": best_row.nearest_order,
                "desired_side": "",
                "desired_qty": "",
                "mnq_symbol": MNQ_SYMBOL,
                "action": "skip_flat_signal",
                "status": "ignored",
                "fill_price": "",
                "payload_json": "",
                "response_json": "",
                "note": "Flat or unparsable signals are ignored",
            }
        )
        print_cyan(f"Log file: {TRADE_LOG_FILE}")
        return

    if not apikey or not systemid_raw:
        raise RuntimeError("Missing C2_API_KEY or C2_SYSTEM_ID environment variable")

    strategy_id = int(systemid_raw)

    openpos_raw = get_open_positions(apikey, strategy_id)
    current_positions = extract_supported_open_positions(openpos_raw)

    if current_positions:
        existing = current_positions[0]
        print_yellow(
            f"MNQ already open: {existing.side.upper()} x {existing.qty} @ {existing.entry_price}. No new order sent."
        )
        log_event(
            {
                "timestamp_utc": now,
                "strategy_name": best_row.system,
                "developer": best_row.developer,
                "rank": best_row.rank,
                "product": best_row.product,
                "scraped_current_position": best_row.current_position,
                "scraped_nearest_order": best_row.nearest_order,
                "desired_side": desired_pos.side,
                "desired_qty": desired_pos.qty,
                "mnq_symbol": MNQ_SYMBOL,
                "action": "skip_existing_open_position",
                "status": "ignored",
                "fill_price": existing.entry_price,
                "payload_json": "",
                "response_json": json.dumps(existing.raw, ensure_ascii=False),
                "note": "Existing MNQ position detected",
            }
        )
        print_cyan(f"Log file: {TRADE_LOG_FILE}")
        return

    payload = build_parent_order_market_only(
        strategy_id=strategy_id,
        full_symbol=MNQ_SYMBOL,
        side=desired_pos.side,
        qty=desired_pos.qty,
    )

    print_cyan("Order payload:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if DRY_RUN:
        print_yellow("DRY_RUN=1 - order not sent.")
        log_event(
            {
                "timestamp_utc": now,
                "strategy_name": best_row.system,
                "developer": best_row.developer,
                "rank": best_row.rank,
                "product": best_row.product,
                "scraped_current_position": best_row.current_position,
                "scraped_nearest_order": best_row.nearest_order,
                "desired_side": desired_pos.side,
                "desired_qty": desired_pos.qty,
                "mnq_symbol": MNQ_SYMBOL,
                "action": "dry_run_market_entry",
                "status": "not_sent",
                "fill_price": "",
                "payload_json": json.dumps(payload, ensure_ascii=False),
                "response_json": "",
                "note": "Dry run mode enabled",
            }
        )
        print_cyan(f"Log file: {TRADE_LOG_FILE}")
        return

    result = api4_post("/Strategies/NewStrategyOrder", apikey, payload)
    print_green_bold("MT5 TRADE DECISION: SEND ORDER")
    print_green_bold("sent order")

    fill = wait_for_fill(
        apikey=apikey,
        strategy_id=strategy_id,
        symbol=MNQ_SYMBOL,
        expected_side=desired_pos.side,
        expected_qty=desired_pos.qty,
        timeout_seconds=45,
        poll_seconds=3,
    )

    fill_price = fill.entry_price if fill else ""
    if fill and fill.entry_price is not None:
        print_green(f"Fill detected at price: {fill.entry_price}")
    else:
        print_yellow("No fill detected within wait window.")

    log_event(
        {
            "timestamp_utc": now,
            "strategy_name": best_row.system,
            "developer": best_row.developer,
            "rank": best_row.rank,
            "product": best_row.product,
            "scraped_current_position": best_row.current_position,
            "scraped_nearest_order": best_row.nearest_order,
            "desired_side": desired_pos.side,
            "desired_qty": desired_pos.qty,
            "mnq_symbol": MNQ_SYMBOL,
            "action": "market_entry",
            "status": "sent",
            "fill_price": fill_price,
            "payload_json": json.dumps(payload, ensure_ascii=False),
            "response_json": json.dumps(result, ensure_ascii=False),
            "note": "Order submitted from scraped top-ranked NQ strategy after 3-symbol consensus",
        }
    )

    print_cyan(f"Log file: {TRADE_LOG_FILE}")
    print_green_bold(
        f"FINAL SUMMARY -> send_trade=True | side={desired_pos.side} | qty={desired_pos.qty} "
        f"| consensus_max_same_direction={consensus['max_same_direction']}"
    )


if __name__ == "__main__":
    main()
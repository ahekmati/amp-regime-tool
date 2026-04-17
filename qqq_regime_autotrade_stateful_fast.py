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

C2_STRATEGY_ID = 155560809

MGC_SYMBOL = os.getenv("MGC_SYMBOL", "@MGCM6")
MCL_SYMBOL = os.getenv("MCL_SYMBOL", "@MCLM6")
MES_SYMBOL = os.getenv("MES_SYMBOL", "@MESM6")
MYM_SYMBOL = os.getenv("MYM_SYMBOL", "@MYMM6")

SUPPORTED_PRODUCTS = {
    "GC": MGC_SYMBOL,
    "CL": MCL_SYMBOL,
    "ES": MES_SYMBOL,
    "YM": MYM_SYMBOL,
}

HISTORY_DIR = Path(os.getenv("HISTORY_DIR", "./history"))
TRADE_LOG_FILE = HISTORY_DIR / "multi_micro_copier_log.csv"

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


def print_yellow(line: str) -> None:
    print(YELLOW + line + RESET)


def print_red(line: str) -> None:
    print(RED + line + RESET)


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


def summarize_products(rows: List[ScrapedRow]) -> str:
    return ", ".join([f"#{r.rank}:{r.product}" for r in rows])


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
        "mapped_symbol",
        "scraped_current_position",
        "scraped_nearest_order",
        "signal_found",
        "signal_text",
        "desired_side",
        "desired_qty",
        "action",
        "status",
        "fill_confirmed",
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
    now = datetime.now(timezone.utc).isoformat()

    print(CYAN + BOLD + f"AMP multi-product -> micro copier started at {now}" + RESET)
    print(CYAN + f"C2 strategy id: {C2_STRATEGY_ID}" + RESET)
    print(CYAN + f"Supported products: {SUPPORTED_PRODUCTS}" + RESET)

    html = fetch_amp_html()
    rows = parse_current_session(html)
    if not rows:
        raise RuntimeError("No rows parsed from AMP current session table")

    print(CYAN + "Top rows found:" + RESET)
    for r in rows:
        print(CYAN + f"#{r.rank} {r.product} | {r.system} | {r.current_position}" + RESET)

    best_row = pick_best_supported(rows)
    if not best_row:
        found = summarize_products(rows)
        print_yellow("No GC, CL, ES, or YM strategy found inside AMP top rows. Nothing to do.")
        print_yellow(f"Products seen: {found}")
        log_event(
            {
                "timestamp_utc": now,
                "strategy_name": "",
                "developer": "",
                "rank": "",
                "product": "",
                "mapped_symbol": "",
                "scraped_current_position": "",
                "scraped_nearest_order": "",
                "signal_found": "no",
                "signal_text": "",
                "desired_side": "",
                "desired_qty": "",
                "action": "skip_no_supported_product_found",
                "status": "ignored",
                "fill_confirmed": "no",
                "fill_price": "",
                "payload_json": "",
                "response_json": "",
                "note": f"Top rows seen: {found}",
            }
        )
        print(CYAN + f"Log file: {TRADE_LOG_FILE}" + RESET)
        return

    desired_pos = parse_direction_and_size(best_row.current_position)
    mapped_symbol = SUPPORTED_PRODUCTS[best_row.product]

    print_green(BOLD + "SCRAPED STRATEGY FOUND" + RESET)
    print_green(f"Strategy: {best_row.system}")
    print_green(f"Developer: {best_row.developer or 'N/A'}")
    print_green(f"Product: {best_row.product}")
    print_green(f"Mapped symbol: {mapped_symbol}")
    print_green(f"Current position: {best_row.current_position}")
    print_green(f"Nearest order: {best_row.nearest_order}")

    if desired_pos:
        signal_text = f"{desired_pos.side.upper()} x {desired_pos.qty}"
        print_green(BOLD + "SIGNAL FOUND" + RESET)
        print_green(f"Signal: {signal_text}")
    else:
        signal_text = "FLAT / NO TRADE SIGNAL"
        print_yellow("AMP signal is flat or unparsable. Ignoring.")
        log_event(
            {
                "timestamp_utc": now,
                "strategy_name": best_row.system,
                "developer": best_row.developer,
                "rank": best_row.rank,
                "product": best_row.product,
                "mapped_symbol": mapped_symbol,
                "scraped_current_position": best_row.current_position,
                "scraped_nearest_order": best_row.nearest_order,
                "signal_found": "no",
                "signal_text": signal_text,
                "desired_side": "",
                "desired_qty": "",
                "action": "skip_flat_signal",
                "status": "ignored",
                "fill_confirmed": "no",
                "fill_price": "",
                "payload_json": "",
                "response_json": "",
                "note": "Flat signals are ignored",
            }
        )
        print(CYAN + f"Log file: {TRADE_LOG_FILE}" + RESET)
        return

    if not apikey:
        raise RuntimeError("Missing C2_API_KEY environment variable")

    openpos_raw = get_open_positions(apikey, C2_STRATEGY_ID)
    current_positions = extract_supported_open_positions(openpos_raw)

    if current_positions:
        existing = current_positions[0]
        print_yellow(
            f"Existing supported position already open: {existing.symbol} {existing.side.upper()} x {existing.qty} @ {existing.entry_price}. Only one trade at a time is allowed, so no new order sent."
        )
        log_event(
            {
                "timestamp_utc": now,
                "strategy_name": best_row.system,
                "developer": best_row.developer,
                "rank": best_row.rank,
                "product": best_row.product,
                "mapped_symbol": mapped_symbol,
                "scraped_current_position": best_row.current_position,
                "scraped_nearest_order": best_row.nearest_order,
                "signal_found": "yes",
                "signal_text": signal_text,
                "desired_side": desired_pos.side,
                "desired_qty": desired_pos.qty,
                "action": "skip_existing_any_supported_position",
                "status": "ignored",
                "fill_confirmed": "no",
                "fill_price": existing.entry_price,
                "payload_json": "",
                "response_json": json.dumps(existing.raw, ensure_ascii=False),
                "note": "A supported micro position is already open; one-at-a-time rule enforced",
            }
        )
        print(CYAN + f"Log file: {TRADE_LOG_FILE}" + RESET)
        return

    payload = build_parent_order_market_only(
        strategy_id=C2_STRATEGY_ID,
        full_symbol=mapped_symbol,
        side=desired_pos.side,
        qty=desired_pos.qty,
    )

    print(CYAN + "Order payload:" + RESET)
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
                "mapped_symbol": mapped_symbol,
                "scraped_current_position": best_row.current_position,
                "scraped_nearest_order": best_row.nearest_order,
                "signal_found": "yes",
                "signal_text": signal_text,
                "desired_side": desired_pos.side,
                "desired_qty": desired_pos.qty,
                "action": "dry_run_market_entry",
                "status": "not_sent",
                "fill_confirmed": "no",
                "fill_price": "",
                "payload_json": json.dumps(payload, ensure_ascii=False),
                "response_json": "",
                "note": "Dry run mode enabled",
            }
        )
        print(CYAN + f"Log file: {TRADE_LOG_FILE}" + RESET)
        return

    result = api4_post("/Strategies/NewStrategyOrder", apikey, payload)
    print_green(BOLD + "sent order" + RESET)

    fill = wait_for_fill(
        apikey=apikey,
        strategy_id=C2_STRATEGY_ID,
        symbol=mapped_symbol,
        expected_side=desired_pos.side,
        expected_qty=desired_pos.qty,
        timeout_seconds=45,
        poll_seconds=3,
    )

    fill_confirmed = "yes" if fill else "no"
    fill_price = fill.entry_price if fill else ""

    if fill:
        if fill.entry_price is not None:
            print_green(BOLD + "C2 ORDER FILL CONFIRMED" + RESET)
            print_green(f"Filled: {fill.symbol} {fill.side.upper()} x {fill.qty}")
            print_green(f"Fill price: {fill.entry_price}")
        else:
            print_green(BOLD + "C2 ORDER FILL CONFIRMED" + RESET)
            print_green(f"Filled: {fill.symbol} {fill.side.upper()} x {fill.qty}")
            print_green("Fill price: unavailable")
    else:
        print_yellow("No matching C2 fill detected within wait window.")

    log_event(
        {
            "timestamp_utc": now,
            "strategy_name": best_row.system,
            "developer": best_row.developer,
            "rank": best_row.rank,
            "product": best_row.product,
            "mapped_symbol": mapped_symbol,
            "scraped_current_position": best_row.current_position,
            "scraped_nearest_order": best_row.nearest_order,
            "signal_found": "yes",
            "signal_text": signal_text,
            "desired_side": desired_pos.side,
            "desired_qty": desired_pos.qty,
            "action": "market_entry",
            "status": "sent",
            "fill_confirmed": fill_confirmed,
            "fill_price": fill_price,
            "payload_json": json.dumps(payload, ensure_ascii=False),
            "response_json": json.dumps(result, ensure_ascii=False),
            "note": "Order submitted from scraped top-ranked supported strategy",
        }
    )

    print(CYAN + f"Log file: {TRADE_LOG_FILE}" + RESET)


if __name__ == "__main__":
    main()
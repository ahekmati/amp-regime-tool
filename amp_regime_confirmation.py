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

    print(CYAN + BOLD + f"AMP NQ -> MNQ copier started at {now}" + RESET)

    html = fetch_amp_html()
    rows = parse_current_session(html)
    if not rows:
        raise RuntimeError("No rows parsed from AMP current session table")

    best_row = pick_best_nq(rows)
    if not best_row:
        raise RuntimeError("No NQ strategy found inside AMP top rows")

    desired_pos = parse_direction_and_size(best_row.current_position)

    print_green(BOLD + "SCRAPED NQ STRATEGY" + RESET)
    print_green(f"Strategy: {best_row.system}")
    print_green(f"Developer: {best_row.developer or 'N/A'}")
    print_green(f"Current position: {best_row.current_position}")
    print_green(f"Nearest order: {best_row.nearest_order}")

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
                "note": "Flat signals are ignored",
            }
        )
        print(CYAN + f"Log file: {TRADE_LOG_FILE}" + RESET)
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
        print(CYAN + f"Log file: {TRADE_LOG_FILE}" + RESET)
        return

    payload = build_parent_order_market_only(
        strategy_id=strategy_id,
        full_symbol=MNQ_SYMBOL,
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
        print(CYAN + f"Log file: {TRADE_LOG_FILE}" + RESET)
        return

    result = api4_post("/Strategies/NewStrategyOrder", apikey, payload)
    print_green(BOLD + "sent order" + RESET)

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
            "note": "Order submitted from scraped top-ranked NQ strategy",
        }
    )

    print(CYAN + f"Log file: {TRADE_LOG_FILE}" + RESET)


if __name__ == "__main__":
    main()
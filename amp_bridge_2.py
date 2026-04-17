#!/usr/bin/env python3
import json
import os
import re
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv(dotenv_path=".env.script2")

AMP_URL = os.getenv("AMP_URL", "https://ampfutures.isystems.com/Systems/TopStrategies")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
TOP_N = int(os.getenv("TOP_N", "25"))
USER_AGENT = os.getenv(
    "USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
)

MT5_SYMBOL_NQ = os.getenv("MT5_SYMBOL_NQ", "MNQM26")
LOT_NQ = float(os.getenv("LOT_NQ", "1.0"))
MNQ_SL_POINTS = float(os.getenv("MNQ_SL_POINTS", "200"))
MNQ_TP_POINTS = float(os.getenv("MNQ_TP_POINTS", "2200"))
SIGNAL_FILE = Path(os.getenv("SIGNAL_FILE", "./amp_bridge_signal_nq.json"))
RANK_FALLBACK_MAX = int(os.getenv("RANK_FALLBACK_MAX", "10"))

DEBUG_DIR = Path(os.getenv("DEBUG_DIR", "./amp_debug"))
SAVE_DEBUG = os.getenv("SAVE_DEBUG", "1").strip().lower() in ("1", "true", "yes", "y")

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def green(msg: str) -> None:
    print(f"{GREEN}{msg}{RESET}")

def green_bold(msg: str) -> None:
    print(f"{GREEN}{BOLD}{msg}{RESET}")

def yellow(msg: str) -> None:
    print(f"{YELLOW}{msg}{RESET}")

def red(msg: str) -> None:
    print(f"{RED}{msg}{RESET}")

def cyan(msg: str) -> None:
    print(f"{CYAN}{msg}{RESET}")

def normalize_text(x: str) -> str:
    return re.sub(r"\s+", " ", (x or "").strip())

def money_to_float(s: str) -> Optional[float]:
    if s is None:
        return None
    s = s.replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except Exception:
        return None

def stable_row_id(system: str, product: str, developer: str) -> str:
    base = f"{normalize_text(system)}|{normalize_text(product)}|{normalize_text(developer)}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]

def fetch_html() -> str:
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(AMP_URL, headers=headers, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.text

def debug_write(name: str, content: str) -> None:
    if not SAVE_DEBUG:
        return
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    (DEBUG_DIR / name).write_text(content, encoding="utf-8")

def parse_current_session(html: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="tableCurrentSession")
    if not table:
        raise RuntimeError("Could not find tableCurrentSession in AMP HTML")
    body = table.find("tbody")
    if not body:
        raise RuntimeError("tableCurrentSession has no tbody")

    rows = []
    debug_lines = []

    for idx, tr in enumerate(body.find_all("tr"), start=1):
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

        raw_text = normalize_text(tr.get_text(" ", strip=True))
        debug_lines.append(f"ROW {idx}: {raw_text}")

        if not all([rank_td, system_td, product_td, pnl_td, pos_td, nearest_td]):
            debug_lines.append(f"ROW {idx}: skipped missing required tds")
            continue

        rank_match = re.search(r"#(\d+)", rank_td.get_text(" ", strip=True))
        if not rank_match:
            debug_lines.append(f"ROW {idx}: skipped no rank match")
            continue

        rank = int(rank_match.group(1))
        if rank > TOP_N:
            debug_lines.append(f"ROW {idx}: skipped rank {rank} > TOP_N {TOP_N}")
            continue

        system = normalize_text(system_td.get_text(" ", strip=True))
        product = normalize_text(product_td.get_text(" ", strip=True)).upper()
        developer = normalize_text(developer_td.get_text(" ", strip=True) if developer_td else "")
        pnl = money_to_float(pnl_td.get_text(" ", strip=True))
        current_position = normalize_text(pos_td.get_text(" ", strip=True))
        nearest_order = normalize_text(nearest_td.get_text(" ", strip=True))

        row = {
            "row_id": stable_row_id(system, product, developer),
            "rank": rank,
            "system": system,
            "product": product,
            "developer": developer,
            "pnl": pnl,
            "current_position": current_position,
            "nearest_order": nearest_order,
        }
        rows.append(row)
        debug_lines.append(f"ROW {idx}: parsed -> {json.dumps(row, ensure_ascii=False)}")

    debug_write("debug_parse_rows.txt", "\n".join(debug_lines))
    return rows

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

def count_directional_consensus(rows: List[Dict]) -> Dict[str, int]:
    long_count = 0
    short_count = 0
    qualifying_rows = []

    for r in rows:
        root = product_root(r["product"], r["system"])
        pos = parse_position_text(r["current_position"])

        if root in ("NQ", "MNQ", "ES", "MES", "YM", "MYM"):
            if pos == "long":
                long_count += 1
                qualifying_rows.append(
                    f"rank #{r['rank']} | {root} | LONG | {r['system']}"
                )
            elif pos == "short":
                short_count += 1
                qualifying_rows.append(
                    f"rank #{r['rank']} | {root} | SHORT | {r['system']}"
                )

    return {
        "long_count": long_count,
        "short_count": short_count,
        "max_same_direction": max(long_count, short_count),
        "qualifying_count": long_count + short_count,
        "qualifying_rows": qualifying_rows,
    }

def choose_nq_leader(rows: List[Dict]) -> Optional[Dict]:
    debug_lines = [f"total rows={len(rows)}"]

    consensus = count_directional_consensus(rows)
    debug_lines.append(
        f"consensus_long_count_NQ_ES_YM={consensus['long_count']} "
        f"consensus_short_count_NQ_ES_YM={consensus['short_count']} "
        f"max_same_direction={consensus['max_same_direction']}"
    )

    if consensus["max_same_direction"] < 3:
        debug_lines.append(
            "Consensus condition FAILED: need at least 3 NQ/ES/YM strategies "
            "all in the same direction (all long or all short). "
            "No NQ leader will be selected."
        )
        debug_write("debug_choose_nq_leader.txt", "\n".join(debug_lines))
        return None

    nq_rows = []
    for r in rows:
        root = product_root(r["product"], r["system"])
        target = parse_position_text(r["current_position"])
        debug_lines.append(
            f"candidate_check rank={r['rank']} product={r['product']} system={r['system']} "
            f"root={root} pos={r['current_position']} target={target}"
        )
        if root in ("NQ", "MNQ"):
            nq_rows.append(r)

    nq_rows.sort(key=lambda x: x["rank"])
    debug_lines.append(f"nq_rows={len(nq_rows)} rank_fallback_max={RANK_FALLBACK_MAX}")

    max_rank_to_consider = max(1, RANK_FALLBACK_MAX)

    for r in nq_rows:
        target = parse_position_text(r["current_position"])
        if r["rank"] > max_rank_to_consider:
            debug_lines.append(f"skip rank={r['rank']} > {max_rank_to_consider}")
            continue
        debug_lines.append(
            f"consider rank={r['rank']} product={r['product']} pos={r['current_position']} target={target}"
        )
        if target in ("long", "short", "flat"):
            debug_lines.append(f"SELECTED row_id={r['row_id']} rank={r['rank']} target={target}")
            debug_write("debug_choose_nq_leader.txt", "\n".join(debug_lines))
            return r

    debug_lines.append("no valid NQ candidate found")
    debug_write("debug_choose_nq_leader.txt", "\n".join(debug_lines))
    return None

def get_top_nq_rows(rows: List[Dict], n: int = 3) -> List[Dict]:
    nq_rows = []
    for r in rows:
        root = product_root(r["product"], r["system"])
        if root in ("NQ", "MNQ"):
            nq_rows.append(r)
    nq_rows.sort(key=lambda x: x["rank"])
    return nq_rows[:n]

def target_to_action(target: Optional[str]) -> Optional[str]:
    if target == "long":
        return "BUY"
    if target == "short":
        return "SELL"
    if target == "flat":
        return "FLAT"
    return None

def build_payload(row: Optional[Dict], error: Optional[str] = None) -> Dict:
    ts = now_utc()
    generated_iso = ts.isoformat()
    generated_unix = int(ts.timestamp())

    payload = {
        "generated_at_utc": generated_iso,
        "source": "AMP TopStrategies",
        "mode": "NQ-only top-ranked selector",
        "error": error,
        "signal": {
            "generated_at_utc": generated_unix,
            "symbol": MT5_SYMBOL_NQ,
            "master_found": False,
            "target": None,
            "action": None,
            "volume": LOT_NQ,
            "sl_points": MNQ_SL_POINTS,
            "tp_points": MNQ_TP_POINTS,
            "system": None,
            "product": None,
            "rank": None,
            "current_position_raw": None,
            "nearest_order": None,
            "row_id": None,
            "message": "No valid NQ/MNQ candidate found in allowed ranks; do nothing",
        },
    }

    if row is not None:
        target = parse_position_text(row["current_position"])
        payload["signal"].update(
            {
                "master_found": True,
                "target": target,
                "action": target_to_action(target),
                "system": row["system"],
                "product": row["product"],
                "rank": row["rank"],
                "current_position_raw": row["current_position"],
                "nearest_order": row["nearest_order"],
                "row_id": row["row_id"],
                "message": f"Selected top-ranked NQ candidate rank {row['rank']}",
            }
        )

    return payload

def write_signal(payload: Dict) -> None:
    SIGNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SIGNAL_FILE.with_suffix(SIGNAL_FILE.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(SIGNAL_FILE)

def main() -> None:
    rows: List[Dict] = []
    top3: List[Dict] = []
    consensus_info: Dict[str, object] = {
        "long_count": 0,
        "short_count": 0,
        "max_same_direction": 0,
        "qualifying_count": 0,
        "qualifying_rows": [],
    }

    started_at = now_utc()
    green_bold(f"=== AMP NQ BRIDGE START {started_at.isoformat()} ===")
    cyan(f"AMP_URL={AMP_URL}")
    cyan(f"TOP_N={TOP_N} | RANK_FALLBACK_MAX={RANK_FALLBACK_MAX} | MT5_SYMBOL_NQ={MT5_SYMBOL_NQ}")

    try:
        green("Fetching AMP current session page...")
        html = fetch_html()
        debug_write("debug_amp_current_session.html", html)
        green(f"AMP fetch OK. HTML length={len(html)} bytes")

        rows = parse_current_session(html)
        debug_write("debug_amp_rows.json", json.dumps(rows, indent=2, ensure_ascii=False))
        green(f"Parsed {len(rows)} rows from AMP table within top {TOP_N} ranks")

        if not rows:
            yellow("No rows parsed from AMP after filtering.")
        else:
            green("Parsed rows summary:")
            for r in rows[:10]:
                root = product_root(r["product"], r["system"])
                pos = parse_position_text(r["current_position"])
                green(
                    f"  rank #{r['rank']} | product={r['product']} | root={root} | "
                    f"parsed_target={pos} | system='{r['system']}'"
                )

        consensus_info = count_directional_consensus(rows)
        qualifying_rows = consensus_info["qualifying_rows"]

        green_bold("Consensus check across NQ / ES / YM family")
        green(
            f"Found qualifying directional signals: {consensus_info['qualifying_count']} "
            f"(long={consensus_info['long_count']}, short={consensus_info['short_count']})"
        )

        if qualifying_rows:
            green("Qualifying rows:")
            for line in qualifying_rows:
                green(f"  {line}")
        else:
            yellow("No qualifying NQ/ES/YM rows with long/short direction were found.")

        if consensus_info["max_same_direction"] >= 3:
            dominant = "LONG" if consensus_info["long_count"] >= 3 else "SHORT"
            green_bold(
                f"Consensus PASSED: at least 3 symbols found in the same direction -> {dominant}"
            )
        else:
            yellow(
                "Consensus FAILED: fewer than 3 NQ/ES/YM-family signals are aligned "
                "in the same direction."
            )

        top3 = get_top_nq_rows(rows, n=3)
        if top3:
            green_bold("Top NQ / MNQ candidates found")
            for r in top3:
                pos = parse_position_text(r.get("current_position", ""))
                green(
                    f"  rank #{r['rank']}: system='{r['system']}' product='{r['product']}' "
                    f"developer='{r['developer']}' pnl={r['pnl']} "
                    f"current_position='{r['current_position']}' parsed_target='{pos}'"
                )
        else:
            yellow("No NQ/MNQ candidates found on the page.")

        leader = choose_nq_leader(rows)
        payload = build_payload(leader)

    except Exception as e:
        red(f"ERROR: {type(e).__name__}: {e}")
        payload = build_payload(None, error=f"{type(e).__name__}: {e}")

    write_signal(payload)
    debug_write("debug_last_payload.json", json.dumps(payload, indent=2, ensure_ascii=False))

    s = payload["signal"]

    green(f"Wrote signal file: {SIGNAL_FILE}")
    green(
        "Directional consensus across NQ/ES/YM family: "
        f"long_count={consensus_info['long_count']} "
        f"short_count={consensus_info['short_count']} "
        f"max_same_direction={consensus_info['max_same_direction']}"
    )

    if s["master_found"]:
        green_bold("Top-ranked NQ leader found")
        green(f"Leader system: {s['system']}")
        green(f"Leader product: {s['product']}")
        green(f"Leader rank: {s['rank']}")
        green(f"Leader current position raw: {s['current_position_raw']}")
        green(f"Leader parsed target: {s['target']}")
        green(f"Leader action: {s['action']}")
        green(f"Leader nearest order: {s['nearest_order']}")
        green(f"Signal generated_at_utc (unix): {s['generated_at_utc']}")
    else:
        yellow("No valid top-ranked NQ leader was selected.")

    print(
        f"selected_found={s['master_found']} "
        f"rank={s['rank']} target={s['target']} action={s['action']} "
        f"product={s['product']} system={s['system']} signal_generated_at_utc={s.get('generated_at_utc')} "
        f"error={payload.get('error')}"
    )

    if s["master_found"] and s["action"]:
        green_bold("MT5 TRADE DECISION: SEND ORDER")
        green(
            f"Sending order to MT5: symbol={s['symbol']} action={s['action']} "
            f"volume={s['volume']} sl_points={s['sl_points']} tp_points={s['tp_points']}"
        )
        green(
            f"Source strategy: system='{s['system']}' rank={s['rank']} "
            f"current_position='{s['current_position_raw']}'"
        )
    else:
        yellow("MT5 TRADE DECISION: DO NOT SEND ORDER")
        yellow(
            "Reason: no valid NQ/MNQ candidate found in allowed ranks, "
            "consensus condition failed, or parsed action was empty."
        )

    green_bold(
        f"FINAL SUMMARY -> send_trade={bool(s['master_found'] and s['action'])} "
        f"| target={s['target']} | action={s['action']} | rank={s['rank']}"
    )

if __name__ == "__main__":
    main()
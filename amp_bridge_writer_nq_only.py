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


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    for root in ("MNQ", "NQ", "MES", "ES"):
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


def choose_nq_leader(rows: List[Dict]) -> Optional[Dict]:
    debug_lines = [f"total rows={len(rows)}"]

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
    """Return top-n NQ/MNQ rows by rank, regardless of whether they are tradable."""
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
    payload = {
        "generated_at_utc": now_utc_iso(),
        "source": "AMP TopStrategies",
        "mode": "NQ-only top-ranked selector",
        "error": error,
        "signal": {
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
    try:
        html = fetch_html()
        debug_write("debug_amp_current_session.html", html)

        rows = parse_current_session(html)
        debug_write("debug_amp_rows.json", json.dumps(rows, indent=2, ensure_ascii=False))

        # always compute top 3 NQ/MNQ for visibility
        top3 = get_top_nq_rows(rows, n=3)

        leader = choose_nq_leader(rows)
        payload = build_payload(leader)

    except Exception as e:
        payload = build_payload(None, error=f"{type(e).__name__}: {e}")

    write_signal(payload)
    debug_write("debug_last_payload.json", json.dumps(payload, indent=2, ensure_ascii=False))

    # Console output
    print(f"Wrote signal file: {SIGNAL_FILE}")
    s = payload["signal"]

    # Print top 3 systems seen on the page
    if top3:
        print("Top 3 NQ/MNQ systems (by rank) from site:")
        for r in top3:
            pos = parse_position_text(r.get("current_position", ""))
            print(
                f"  rank #{r['rank']}: system='{r['system']}' product='{r['product']}' "
                f"developer='{r['developer']}' pnl={r['pnl']} "
                f"current_position='{r['current_position']}' parsed_target='{pos}'"
            )
    else:
        print("No NQ/MNQ systems found on the page (top 3 list is empty).")

    print(
        f"selected_found={s['master_found']} "
        f"rank={s['rank']} target={s['target']} action={s['action']} "
        f"product={s['product']} system={s['system']} error={payload.get('error')}"
    )

    # Explicit trade / no-trade message
    if s["master_found"] and s["action"]:
        print(
            f"sending order to mt5: symbol={s['symbol']} action={s['action']} "
            f"volume={s['volume']} sl_points={s['sl_points']} tp_points={s['tp_points']} "
            f"system='{s['system']}' rank={s['rank']}"
        )
    else:
        print("no order sent to mt5 (no valid NQ/MNQ candidate in allowed ranks).")


if __name__ == "__main__":
    main()
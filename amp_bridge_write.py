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
RANK_FALLBACK_MAX = int(os.getenv("RANK_FALLBACK_MAX", "2"))


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


def choose_nq_leader(rows: List[Dict]) -> Optional[Dict]:
    nq_rows = [r for r in rows if product_root(r["product"]) in ("NQ", "MNQ")]
    nq_rows.sort(key=lambda x: x["rank"])

    max_rank_to_consider = max(1, RANK_FALLBACK_MAX)
    for r in nq_rows:
        if r["rank"] > max_rank_to_consider:
            continue
        target = parse_position_text(r["current_position"])
        if target in ("long", "short", "flat"):
            return r
    return None


def build_payload(row: Optional[Dict]) -> Dict:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "AMP TopStrategies",
        "mode": "NQ-only top-ranked selector",
        "signal": {
            "symbol": MT5_SYMBOL_NQ,
            "master_found": False,
            "target": None,
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


def main() -> None:
    html = fetch_html()
    rows = parse_current_session(html)
    leader = choose_nq_leader(rows)
    payload = build_payload(leader)

    SIGNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SIGNAL_FILE.with_suffix(SIGNAL_FILE.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(SIGNAL_FILE)

    print(f"Wrote signal file: {SIGNAL_FILE}")
    s = payload["signal"]
    print(f"selected_found={s['master_found']} rank={s['rank']} target={s['target']} product={s['product']} system={s['system']}")


if __name__ == "__main__":
    main()

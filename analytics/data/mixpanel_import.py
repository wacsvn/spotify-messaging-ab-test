"""
Import simulated_experiment_log.csv into Mixpanel via the /import endpoint.

Why /import (not /track):
  - /track only accepts events timestamped within the last 5 days and stamps
    events with ingestion time if `time` is missing. The old script sent no
    `time`, so all ~398k events collapsed onto one moment.
  - /import accepts any timestamp, requires `time` + `distinct_id` + `$insert_id`,
    and with strict=1 returns per-event validation errors instead of a
    meaningless "1".

Setup (env vars, never hardcode):
  MIXPANEL_TOKEN    project token (Project Settings -> Project Token)
  MIXPANEL_REGION   api (US, default) | api-eu | api-in   <- must match project data residency

Usage:
  python mixpanel_import.py --dry-run             # build + validate, send nothing
  python mixpanel_import.py --limit 2000          # send ONE batch, verify in Mixpanel
  python mixpanel_import.py                       # full import
"""

import argparse
import gzip
import json
import os
import random
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

BATCH_SIZE = 2000
INSERT_ID_RE = re.compile(r"^[A-Za-z0-9-]{1,36}$")

PROPERTY_COLUMNS = [
    "search_minutes",
    "search_sessions",
    "autoplay_minutes",
    "smart_queue_minutes",
    "total_minutes",
    "crashes",
    "w4_active_day",
    "experiment_day",
]


def build_events(df: pd.DataFrame, start: date, event_name: str) -> list[dict]:
    start_epoch = int(
        datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp()
    )
    events = []
    for r in df.itertuples(index=False):
        # Deterministic time-of-day (06:00-20:00 UTC) per user so reruns are idempotent.
        offset = 6 * 3600 + (int(r.user_id, 16) % (14 * 3600))
        ts = start_epoch + (int(r.experiment_day) - 1) * 86400 + offset
        props = {
            "time": ts,
            "distinct_id": str(r.user_id),
            "$insert_id": f"{r.user_id}-{int(r.experiment_day)}",
            "experiment_variant": str(r.variant),
            "is_complier": int(r.is_complier),
        }
        for col in PROPERTY_COLUMNS:
            val = getattr(r, col)
            props[col] = float(val) if col.endswith("_minutes") else int(val)
        events.append({"event": event_name, "properties": props})
    return events


def validate(events: list[dict]) -> None:
    ids = set()
    now = time.time()
    for e in events:
        p = e["properties"]
        assert INSERT_ID_RE.match(p["$insert_id"]), p["$insert_id"]
        assert p["$insert_id"] not in ids, f"duplicate insert id {p['$insert_id']}"
        ids.add(p["$insert_id"])
        assert 0 < p["time"] <= now + 3600, p["time"]
    print(f"Validated {len(events):,} events: unique $insert_ids, timestamps in range.")


def send_batch(batch, url, token, attempt_limit=5):
    body = gzip.compress(json.dumps(batch).encode("utf-8"))
    headers = {"Content-Type": "application/json", "Content-Encoding": "gzip"}
    for attempt in range(attempt_limit):
        resp = requests.post(
            url, data=body, headers=headers, auth=(token, ""), timeout=60
        )
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 502, 503):
            wait = min(60, 2 * 2**attempt) + random.uniform(1, 5)
            print(f"  {resp.status_code}, backing off {wait:.1f}s")
            time.sleep(wait)
            continue
        # 400 / 401 / 413: do not retry, surface details.
        print(f"FAILED {resp.status_code}: {resp.text[:1500]}")
        sys.exit(1)
    print("Gave up after retries.")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    default_csv = Path(__file__).parent  / "simulated_experiment_log.csv"
    ap.add_argument("--csv", default=str(default_csv))
    ap.add_argument("--event-name", default="app_session_summary")
    ap.add_argument(
        "--start-date",
        default=(date.today() - timedelta(days=29)).isoformat(),
        help="Calendar date for experiment_day 1 (YYYY-MM-DD, UTC).",
    )
    ap.add_argument("--limit", type=int, default=None, help="Only send first N events.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df):,} rows, {df.user_id.nunique():,} users from {args.csv}")

    events = build_events(df, date.fromisoformat(args.start_date), args.event_name)
    validate(events)
    if args.limit:
        events = events[: args.limit]

    first = datetime.fromtimestamp(
        min(e["properties"]["time"] for e in events), timezone.utc
    )
    last = datetime.fromtimestamp(
        max(e["properties"]["time"] for e in events), timezone.utc
    )
    print(f"Time span: {first:%Y-%m-%d} -> {last:%Y-%m-%d} UTC")
    print("Sample event:", json.dumps(events[0], indent=2))

    if args.dry_run:
        print("Dry run: nothing sent.")
        return

    token = os.environ.get("MIXPANEL_TOKEN")
    if not token:
        sys.exit("Set MIXPANEL_TOKEN env var.")
    region = os.environ.get("MIXPANEL_REGION", "api")
    url = f"https://{region}.mixpanel.com/import?strict=1"
    print(f"Target: {url}")

    sent = 0
    for i in range(0, len(events), BATCH_SIZE):
        result = send_batch(events[i : i + BATCH_SIZE], url, token)
        sent += result.get("num_records_imported", 0)
        print(
            f"  batch {i // BATCH_SIZE + 1}: imported {result.get('num_records_imported')} (total {sent:,})"
        )
        time.sleep(0.1)

    print(f"Done. Mixpanel confirmed {sent:,} of {len(events):,} events imported.")


if __name__ == "__main__":
    main()

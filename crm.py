"""SQLite-backed mini CRM + appointment calendar.

Production: Postgres, plus a CRMAdapter (HubSpot / Salesforce / Zoho) and Google/Outlook Calendar.
`push_to_external_crm` shows where that webhook/API call goes.
"""
import json
import sqlite3
import threading
import time
from datetime import date, datetime, timedelta

DB = "crm.db"
WORK_HOURS = range(10, 18)          # 10:00 - 17:00 start times
_lock = threading.Lock()


def _conn():
    c = sqlite3.connect(DB, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init():
    with _lock, _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS leads(
            id INTEGER PRIMARY KEY AUTOINCREMENT, phone TEXT UNIQUE, name TEXT, city TEXT,
            monthly_bill INTEGER, timeline TEXT, language TEXT, score INTEGER, tier TEXT,
            updated_at REAL);
        CREATE TABLE IF NOT EXISTS bookings(
            id INTEGER PRIMARY KEY AUTOINCREMENT, lead_id INTEGER, slot TEXT UNIQUE, status TEXT, created_at REAL);
        CREATE TABLE IF NOT EXISTS calls(
            id TEXT PRIMARY KEY, lead_id INTEGER, language TEXT, sentiment_avg REAL, tier TEXT,
            score INTEGER, outcome TEXT, summary TEXT, transcript TEXT, intents TEXT, duration REAL, created_at REAL);
        """)


def upsert_lead(slots: dict, lang: str, score: int, tier: str) -> int | None:
    phone = slots.get("phone")
    if not phone:
        return None
    with _lock, _conn() as c:
        c.execute("""INSERT INTO leads(phone,name,city,monthly_bill,timeline,language,score,tier,updated_at)
                     VALUES(?,?,?,?,?,?,?,?,?)
                     ON CONFLICT(phone) DO UPDATE SET name=COALESCE(excluded.name,name),
                       city=COALESCE(excluded.city,city), monthly_bill=COALESCE(excluded.monthly_bill,monthly_bill),
                       timeline=COALESCE(excluded.timeline,timeline), language=excluded.language,
                       score=excluded.score, tier=excluded.tier, updated_at=excluded.updated_at""",
                  (phone, slots.get("name"), slots.get("city"), slots.get("monthly_bill"),
                   slots.get("timeline"), lang, score, tier, time.time()))
        return c.execute("SELECT id FROM leads WHERE phone=?", (phone,)).fetchone()["id"]


def free_slots(day: date, limit: int = 3, after_hour: int = 0) -> list[datetime]:
    if day.weekday() == 6:                      # Sunday closed
        return []
    with _lock, _conn() as c:
        taken = {r["slot"] for r in c.execute("SELECT slot FROM bookings WHERE status='confirmed' AND slot LIKE ?",
                                              (day.isoformat() + "%",))}
    now = datetime.now()
    out = []
    for h in WORK_HOURS:
        s = datetime(day.year, day.month, day.day, h)
        if s.isoformat(timespec="minutes") in taken or s <= now or h < after_hour:
            continue
        out.append(s)
    return out[:limit]


def next_free(day: date, limit: int = 3) -> tuple[date, list[datetime]]:
    for i in range(0, 10):
        d = day + timedelta(days=i)
        s = free_slots(d, limit)
        if s:
            return d, s
    return day, []


def book(lead_id: int, slot: datetime) -> bool:
    key = slot.isoformat(timespec="minutes")
    with _lock, _conn() as c:
        try:
            c.execute("INSERT INTO bookings(lead_id,slot,status,created_at) VALUES(?,?,?,?)",
                      (lead_id, key, "confirmed", time.time()))
            return True
        except sqlite3.IntegrityError:          # UNIQUE(slot) => double-booking impossible
            return False


def save_call(call: dict):
    with _lock, _conn() as c:
        c.execute("INSERT OR REPLACE INTO calls VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                  (call["id"], call.get("lead_id"), call["language"], call["sentiment_avg"], call["tier"],
                   call["score"], call["outcome"], call["summary"], json.dumps(call["transcript"]),
                   json.dumps(call["intents"]), call["duration"], time.time()))


def push_to_external_crm(lead: dict):
    """Placeholder for HubSpot/Salesforce/Zoho sync (queue job in production)."""
    return {"synced": False, "note": "stub - wire a CRMAdapter here", "lead": lead}


def rows(sql: str, args=()):
    with _lock, _conn() as c:
        return [dict(r) for r in c.execute(sql, args)]

"""Dialogue manager.

Design: POLICY decides (deterministic, testable), GENERATOR verbalises (LLM if available, else templates).
  user text -> NLU (lang, sentiment, entities, intent) -> RAG -> policy -> directive -> reply text
The LLM never decides whether a booking happens or what the lead score is; it only phrases the reply.
"""
import os
import time
import uuid
from datetime import date, datetime

import analysis as A
import crm
from rag import Retriever, first_sentences

COMPANY = os.getenv("COMPANY_NAME", "SunNest Solar")
AGENT_NAME = os.getenv("AGENT_NAME", "Asha")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
KB = Retriever(os.path.join(os.path.dirname(__file__), "knowledge"))

# Per-language canned lines (fallback generator). LLM path translates freely.
T = {
    "en": {
        "greet": f"Hello! I'm {AGENT_NAME} from {COMPANY}. I help families and businesses cut their electricity bills with rooftop solar. May I ask what your monthly electricity bill is roughly?",
        "ask_bill": "Roughly how much is your monthly electricity bill?",
        "ask_city": "Which city is the property in?",
        "ask_name": "May I have your name, please?",
        "ask_phone": "What's the best mobile number to confirm your visit?",
        "offer_visit": "Based on that, a free site survey would give you an exact quote. Shall I book one for you?",
        "ask_time": "Which day and time suits you? For example, tomorrow at 11 am.",
        "offer_slots": "I have {slots} free on {day}. Which one works?",
        "booked": "Done! Your free site survey is booked for {when}. I'll send the details on WhatsApp. Anything else I can help with?",
        "slot_taken": "That slot is taken. I can do {slots} on {day}. Which works?",
        "no_slots": "I couldn't find free slots soon; a colleague will call you to schedule.",
        "price_ack": "I understand cost matters. ",
        "not_interested": "No problem at all, thank you for your time. Have a great day!",
        "delay_ack": "Of course, take your time. ",
        "bye": "Thank you for your time. Have a wonderful day!",
        "unknown": "I'm not fully sure about that, but I can have a specialist confirm it during your free survey. ",
        "closing_nudge": "Would you like me to book that free site survey?",
    },
    "hi": {
        "greet": f"नमस्ते! मैं {COMPANY} से {AGENT_NAME} बोल रही हूँ। हम सोलर पैनल से आपका बिजली का बिल कम करने में मदद करते हैं। आपका महीने का बिजली बिल लगभग कितना आता है?",
        "ask_bill": "आपका महीने का बिजली बिल लगभग कितना आता है?",
        "ask_city": "आपकी प्रॉपर्टी किस शहर में है?",
        "ask_name": "क्या मैं आपका नाम जान सकती हूँ?",
        "ask_phone": "विज़िट कन्फर्म करने के लिए आपका मोबाइल नंबर क्या है?",
        "offer_visit": "इसके आधार पर, एक मुफ़्त साइट सर्वे से आपको सही कोटेशन मिल जाएगा। क्या मैं बुक कर दूँ?",
        "ask_time": "आपको कौन सा दिन और समय ठीक रहेगा? जैसे कल सुबह 11 बजे।",
        "offer_slots": "{day} को {slots} खाली हैं। कौन सा ठीक रहेगा?",
        "booked": "हो गया! आपका मुफ़्त साइट सर्वे {when} के लिए बुक है। डिटेल्स WhatsApp पर भेज दूँगी। और कुछ मदद चाहिए?",
        "slot_taken": "वह स्लॉट भरा है। {day} को {slots} उपलब्ध हैं। कौन सा ठीक रहेगा?",
        "no_slots": "अभी खाली स्लॉट नहीं मिले; हमारा सहकर्मी आपको कॉल करेगा।",
        "price_ack": "मैं समझती हूँ कि कीमत मायने रखती है। ",
        "not_interested": "कोई बात नहीं, आपके समय के लिए धन्यवाद। आपका दिन शुभ हो!",
        "delay_ack": "बिल्कुल, आप आराम से सोचिए। ",
        "bye": "आपके समय के लिए धन्यवाद। आपका दिन शुभ हो!",
        "unknown": "इस बारे में मैं पूरी तरह निश्चित नहीं हूँ, हमारा विशेषज्ञ सर्वे के दौरान बता देगा। ",
        "closing_nudge": "क्या मैं मुफ़्त साइट सर्वे बुक कर दूँ?",
    },
    "te": {
        "greet": f"నమస్కారం! నేను {COMPANY} నుండి {AGENT_NAME} మాట్లాడుతున్నాను. సోలార్ ద్వారా మీ కరెంటు బిల్లు తగ్గించడంలో మేము సహాయం చేస్తాము. మీ నెలవారీ కరెంటు బిల్లు సుమారుగా ఎంత వస్తుంది?",
        "ask_bill": "మీ నెలవారీ కరెంటు బిల్లు సుమారుగా ఎంత వస్తుంది?",
        "ask_city": "మీ ప్రాపర్టీ ఏ నగరంలో ఉంది?",
        "ask_name": "మీ పేరు చెప్పగలరా?",
        "ask_phone": "విజిట్ ధృవీకరించడానికి మీ మొబైల్ నంబర్ ఏమిటి?",
        "offer_visit": "దీని ఆధారంగా, ఉచిత సైట్ సర్వే ద్వారా మీకు ఖచ్చితమైన కోట్ లభిస్తుంది. నేను బుక్ చేయమంటారా?",
        "ask_time": "మీకు ఏ రోజు, ఏ సమయం అనుకూలం? ఉదాహరణకు రేపు ఉదయం 11 గంటలకు.",
        "offer_slots": "{day} నాడు {slots} ఖాళీగా ఉన్నాయి. ఏది మీకు సరిపోతుంది?",
        "booked": "అయిపోయింది! మీ ఉచిత సైట్ సర్వే {when} కి బుక్ అయింది. వివరాలు WhatsApp లో పంపుతాను. ఇంకేమైనా సహాయం కావాలా?",
        "slot_taken": "ఆ స్లాట్ నిండిపోయింది. {day} నాడు {slots} అందుబాటులో ఉన్నాయి. ఏది సరిపోతుంది?",
        "no_slots": "ప్రస్తుతం ఖాళీ స్లాట్లు దొరకలేదు; మా సహోద్యోగి మీకు కాల్ చేస్తారు.",
        "price_ack": "ఖర్చు ముఖ్యమని నేను అర్థం చేసుకున్నాను. ",
        "not_interested": "పర్వాలేదు, మీ సమయానికి ధన్యవాదాలు. మీ రోజు శుభంగా గడవాలి!",
        "delay_ack": "తప్పకుండా, మీరు నెమ్మదిగా ఆలోచించండి. ",
        "bye": "మీ సమయానికి ధన్యవాదాలు. శుభదినం!",
        "unknown": "దీని గురించి నాకు పూర్తిగా తెలియదు, సర్వే సమయంలో మా నిపుణులు చెబుతారు. ",
        "closing_nudge": "ఉచిత సైట్ సర్వే బుక్ చేయమంటారా?",
    },
}

SESSIONS: dict[str, dict] = {}   # prod: Redis with TTL (stateless API pods)


def _fmt_slots(slots, lang):
    return ", ".join(s.strftime("%I:%M %p").lstrip("0") for s in slots)


def _fmt_day(d: date):
    return d.strftime("%A, %d %b")


# ---------------------------------------------------------------- start / end
def start_call(lang: str = "en") -> dict:
    lang = lang if lang in T else "en"
    cid = uuid.uuid4().hex[:10]
    s = {"id": cid, "lang": lang, "history": [], "slots": {}, "sentiments": [], "intents": [],
         "stage": "greeted", "booking": None, "lead_id": None, "t0": time.time(), "pending_day": None,
         "sources": []}
    greet = T[lang]["greet"]
    s["history"].append({"role": "agent", "text": greet})
    SESSIONS[cid] = s
    return {"call_id": cid, "reply": greet, "lang": lang}


def _next_missing(s):
    sl = s["slots"]
    for key, tmpl in (("monthly_bill", "ask_bill"), ("city", "ask_city")):
        if not sl.get(key):
            return tmpl
    return None


def _try_book(s, lang) -> str | None:
    """Return reply text if the booking step produced one, else None."""
    sl, tx = s["slots"], T[lang]
    if not sl.get("name"):
        s["expecting"] = "name"
        return tx["ask_name"]
    if not sl.get("phone"):
        s["expecting"] = "phone"
        return tx["ask_phone"]
    s["expecting"] = None
    day = date.fromisoformat(sl["pref_date"]) if sl.get("pref_date") else None
    if not day:
        s["stage"] = "need_time"
        return tx["ask_time"]
    sc = A.score_lead(sl, s["sentiments"], s["intents"], False)
    s["lead_id"] = crm.upsert_lead(sl, lang, sc["score"], sc["tier"])
    if sl.get("pref_hour") is not None:
        want = datetime(day.year, day.month, day.day, sl["pref_hour"])
        free = crm.free_slots(day, limit=20)
        if want in free:
            if crm.book(s["lead_id"], want):
                s["booking"] = want.isoformat(timespec="minutes")
                s["stage"] = "booked"
                return tx["booked"].format(when=want.strftime("%A, %d %b at %I:%M %p"))
        d2, alts = crm.next_free(day)
        if not alts:
            return tx["no_slots"]
        sl.pop("pref_hour", None)
        return tx["slot_taken"].format(slots=_fmt_slots(alts, lang), day=_fmt_day(d2))
    d2, alts = crm.next_free(day)
    s["stage"] = "choosing_slot"
    if not alts:
        return tx["no_slots"]
    sl["pref_date"] = d2.isoformat()
    return tx["offer_slots"].format(slots=_fmt_slots(alts, lang), day=_fmt_day(d2))


# --------------------------------------------------------------- one turn
def handle_turn(call_id: str, text: str) -> dict:
    s = SESSIONS[call_id]
    t0 = time.perf_counter()
    lang = A.detect_language(text, s["lang"])
    lang = lang if lang in T else "en"
    s["lang"] = lang
    tx = T[lang]

    ent = A.extract_entities(text)
    s["slots"].update({k: v for k, v in ent.items()})
    sent = A.sentiment(text)
    intent = A.classify_intent(text)
    if s.get("expecting") == "name" and "name" not in ent and intent in ("statement", "greeting"):
        bn = A.bare_name(text)                          # context-aware slot filling
        if bn:
            ent["name"] = bn
            s["slots"]["name"] = bn
    s["sentiments"].append(sent["score"])
    s["intents"].append(intent)
    s["history"].append({"role": "user", "text": text, "sentiment": sent, "intent": intent})

    booking_stage = s["stage"] in ("need_contact", "need_time", "choosing_slot")
    if intent == "question":
        hits = KB.search(text, k=2)
    elif intent in ("statement", "objection_delay") and not booking_stage:
        hits = KB.search(text, k=1, min_score=0.2)       # stricter: avoid irrelevant answers to chit-chat
    else:
        hits = []
    if intent == "objection_price":
        hits = KB.search("financing EMI subsidy payback savings", k=1)
    s["sources"] += [h["title"] for h in hits]

    parts: list[str] = []
    end_call = False
    st = s["stage"]

    if intent == "not_interested":
        parts.append(tx["not_interested"]); end_call = True
    elif intent == "goodbye":
        parts.append(tx["bye"]); end_call = True
    else:
        in_booking = st in ("offered", "need_contact", "need_time", "choosing_slot") or intent == "book" or \
                     (st == "booked" and ent.get("pref_hour") is not None)
        if intent == "objection_price":
            parts.append(tx["price_ack"])
        elif intent == "objection_delay":
            parts.append(tx["delay_ack"])
        if hits and intent != "affirm":
            parts.append(first_sentences(hits[0]["text"], 2) + " ")
        elif intent == "question":
            parts.append(tx["unknown"])

        wants_booking = intent == "book" or (st == "offered" and intent in ("affirm",)) or \
                        (st in ("need_contact", "need_time", "choosing_slot") and intent not in ("question", "deny")) or \
                        (st == "offered" and ent.get("pref_date"))
        if st == "booked" and not (intent == "book"):
            wants_booking = False
        if wants_booking:
            r = _try_book(s, lang)
            if r:
                if s["stage"] not in ("booked", "choosing_slot", "need_time"):
                    s["stage"] = "need_contact"
                parts.append(r)
        elif st in ("offered", "need_contact", "need_time", "choosing_slot") and intent == "deny":
            parts.append(tx["delay_ack"]); s["stage"] = "greeted"; s["expecting"] = None
        elif st != "booked":
            miss = _next_missing(s)
            if miss and not (intent == "objection_price"):
                parts.append(tx[miss])
            elif s["stage"] != "offered":
                parts.append(tx["offer_visit"]); s["stage"] = "offered"
            else:
                parts.append(tx["closing_nudge"])
        else:
            parts.append(tx["bye"]); end_call = intent in ("affirm",) and False

    directive = " ".join(p.strip() for p in parts if p.strip())
    reply = llm_rephrase(s, text, directive, hits) or directive
    s["history"].append({"role": "agent", "text": reply})

    sc = A.score_lead(s["slots"], s["sentiments"], s["intents"], s["booking"] is not None)
    if s["slots"].get("phone"):
        s["lead_id"] = crm.upsert_lead(s["slots"], lang, sc["score"], sc["tier"])
    return {"reply": reply, "lang": lang, "sentiment": sent, "intent": intent, "slots": s["slots"],
            "lead": sc, "booking": s["booking"], "sources": [{"title": h["title"], "score": h["score"]} for h in hits],
            "end_call": end_call, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


# -------------------------------------------------------------- optional LLM
def llm_rephrase(s, user_text, directive, hits) -> str | None:
    """If ANTHROPIC_API_KEY is set, let an LLM phrase the reply naturally in the caller's language.
    Facts/actions in `directive` are authoritative. Any failure -> None -> template reply is used."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import requests
        ctx = "\n---\n".join(h["text"] for h in hits) or "(none)"
        lang_name = {"en": "English", "hi": "Hindi", "te": "Telugu"}[s["lang"]]
        system = (f"You are {AGENT_NAME}, a warm, concise phone sales agent for {COMPANY}. Reply in {lang_name}, "
                  "max 2 short sentences, spoken style, no markdown. Use ONLY the knowledge context for facts. "
                  "The DIRECTIVE contains the actions/facts you must convey (e.g. a confirmed booking); never contradict it.\n"
                  f"KNOWLEDGE:\n{ctx}")
        msgs = [{"role": "user" if h["role"] == "user" else "assistant", "content": h["text"]} for h in s["history"][-8:-1]]
        msgs = [m for m in msgs]
        msgs.append({"role": "user", "content": f"Caller said: {user_text}\nDIRECTIVE: {directive}"})
        while msgs and msgs[0]["role"] != "user":
            msgs.pop(0)
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=8,
                          headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                          json={"model": LLM_MODEL, "max_tokens": 200, "system": system, "messages": msgs})
        r.raise_for_status()
        return r.json()["content"][0]["text"].strip()
    except Exception:
        return None


# ------------------------------------------------------- summary & finish
def end_call(call_id: str) -> dict:
    s = SESSIONS.get(call_id)
    if not s:
        return {"error": "unknown call"}
    sl = s["sentiments"]
    avg = round(sum(sl) / len(sl), 2) if sl else 0.0
    sc = A.score_lead(s["slots"], sl, s["intents"], s["booking"] is not None)
    outcome = "booked" if s["booking"] else "not_interested" if "not_interested" in s["intents"] else "follow_up"
    objections = sorted({i for i in s["intents"] if i.startswith("objection")})
    topics = sorted(set(s["sources"]))
    f = s["slots"]
    summary = (
        f"Lead {f.get('name', 'Unknown')} ({f.get('phone', 'no phone')}) from {f.get('city', 'unknown city')}, "
        f"monthly bill {('Rs ' + str(f['monthly_bill'])) if f.get('monthly_bill') else 'not shared'}, "
        f"language {s['lang']}. Outcome: {outcome}"
        f"{' for ' + s['booking'].replace('T', ' ') if s['booking'] else ''}. "
        f"Sentiment avg {avg:+.2f}. Lead tier {sc['tier'].upper()} ({sc['score']}/100). "
        f"Topics asked: {', '.join(topics) or 'none'}. Objections: {', '.join(objections) or 'none'}. "
        f"Next action: {'send WhatsApp confirmation + reminder 2h before visit' if s['booking'] else 'sales rep follow-up call in 24h' if outcome == 'follow_up' else 'add to DND list'}."
    )
    call = {"id": call_id, "lead_id": s["lead_id"], "language": s["lang"], "sentiment_avg": avg, "tier": sc["tier"],
            "score": sc["score"], "outcome": outcome, "summary": summary, "transcript": s["history"],
            "intents": s["intents"], "duration": round(time.time() - s["t0"], 1)}
    crm.save_call(call)
    SESSIONS.pop(call_id, None)
    return {"summary": summary, "outcome": outcome, "lead": sc, "sentiment_avg": avg, "booking": s["booking"]}


def insights() -> dict:
    calls = crm.rows("SELECT * FROM calls")
    n = len(calls)
    if not n:
        return {"calls": 0}
    booked = sum(c["outcome"] == "booked" for c in calls)
    from collections import Counter
    objs = Counter()
    for c in calls:
        import json as _j
        for i in _j.loads(c["intents"]):
            if i.startswith("objection") or i == "not_interested":
                objs[i] += 1
    return {"calls": n, "booking_rate": round(booked / n, 2),
            "avg_sentiment": round(sum(c["sentiment_avg"] for c in calls) / n, 2),
            "tiers": dict(Counter(c["tier"] for c in calls)),
            "languages": dict(Counter(c["language"] for c in calls)),
            "top_objections": dict(objs.most_common(3)),
            "avg_duration_s": round(sum(c["duration"] for c in calls) / n, 1)}

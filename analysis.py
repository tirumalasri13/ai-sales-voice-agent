"""Deterministic NLU helpers: language, sentiment, entity extraction, lead scoring.

Kept rule-based on purpose: fast (<1 ms), free, explainable, and easy to unit-test.
In production these can be swapped for a classifier / LLM structured-output call.
Languages: English (en), Hindi (hi), Telugu (te) - native script and romanized.
"""
import re
from datetime import date, timedelta

CITY_LIST = ("ludhiana|jalandhar|amritsar|phagwara|patiala|chandigarh|mohali|bathinda|delhi|mumbai|"
             "bangalore|bengaluru|pune|chennai|jaipur|hyderabad|secunderabad|vijayawada|visakhapatnam|vizag|"
             "guntur|tirupati|warangal|nellore|kurnool|rajahmundry|karimnagar|nizamabad|khammam")


# ---------------------------------------------------------------- language
def detect_language(text: str, default: str = "en") -> str:
    if re.search(r"[ఀ-౿]", text):      # Telugu script
        return "te"
    if re.search(r"[ऀ-ॿ]", text):      # Devanagari
        return "hi"
    low = text.lower()
    hi_roman = {"mujhe", "mera", "meri", "kitna", "kitni", "hai", "nahi", "haan", "kya",
                "chahiye", "aap", "kal", "bataiye", "theek", "paisa", "kharcha"}
    te_roman = {"naku", "nenu", "mee", "enta", "kavali", "avunu", "ledu", "vaddu", "repu", "cheppandi",
                "bagundi", "undi", "ayindi", "peru", "kharchu", "emiti", "ivvandi", "cheyyandi", "gantalaku"}
    words = set(re.findall(r"[a-z]+", low))
    if len(words & te_roman) >= 2:
        return "te"
    if len(words & hi_roman) >= 2:
        return "hi"
    return default if default in ("hi", "te") and len(words) < 3 else "en"


# --------------------------------------------------------------- sentiment
POS = {"great", "good", "love", "interested", "perfect", "excellent", "sure", "yes", "nice",
       "awesome", "helpful", "thanks", "thank", "happy", "sounds", "definitely", "please",
       "accha", "achha", "badhiya", "theek", "haan", "shukriya", "dhanyavad", "haanji",
       "bagundi", "avunu", "sare", "manchidi", "dhanyavadalu", "santosham",
       "अच्छा", "बढ़िया", "हाँ", "धन्यवाद", "ठीक",
       "మంచిది", "బాగుంది", "అవును", "సరే", "ధన్యవాదాలు", "సంతోషం"}
NEG = {"expensive", "costly", "bad", "no", "not", "never", "worried", "problem", "scam", "waste",
       "angry", "annoyed", "slow", "confusing", "doubt", "unsure", "hate", "stop", "busy",
       "mehnga", "mahanga", "nahi", "nahin", "dikkat", "pareshani", "bekaar",
       "ekkuva", "kharidu", "ledu", "vaddu", "kashtam", "ibbandi",
       "महंगा", "नहीं", "परेशानी",
       "ఎక్కువ", "ఖరీదు", "వద్దు", "లేదు", "కష్టం", "ఇబ్బంది"}
NEGATORS = {"not", "no", "never", "don't", "dont", "isn't", "nahi", "nahin", "ledu", "vaddu", "లేదు", "వద్దు"}


def sentiment(text: str) -> dict:
    toks = re.findall(r"[\wऀ-ॿఀ-౿']+", text.lower())
    score = 0.0
    for i, t in enumerate(toks):
        if t in POS:
            flipped = i > 0 and toks[i - 1] in NEGATORS
            score += -1 if flipped else 1
        elif t in NEG and t not in NEGATORS:
            score -= 1
        elif t in NEGATORS:
            score -= 0.3
    norm = max(-1.0, min(1.0, score / max(3, len(toks) ** 0.5 * 1.5)))
    label = "positive" if norm > 0.15 else "negative" if norm < -0.15 else "neutral"
    return {"score": round(norm, 2), "label": label}


# -------------------------------------------------------------- extraction
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def extract_entities(text: str, today: date | None = None) -> dict:
    today = today or date.today()
    low = text.lower()
    out: dict = {}

    m = re.search(r"(?<!\d)(?:\+?91[\s-]?)?([6-9]\d{9})(?!\d)", re.sub(r"[\s-]", "", text))
    if m:
        out["phone"] = m.group(1)

    m = re.search(r"(?:my name is|i am|i'm|this is|mera naam|naa peru|na peru|naa name|मेरा नाम|నా పేరు)\s+([A-Za-zऀ-ॿఀ-౿]+)", text, re.I)
    if not m:                                           # "Sri is my name", "call me Sri", "it's Sri"
        m = re.search(r"([A-Za-z]+)\s+is\s+my\s+name|(?:call me|it's|its)\s+([A-Za-z]+)", text, re.I)
        if m:
            m = re.match(r"(.*)", m.group(1) or m.group(2))
    if m and m.group(1).lower() not in {"interested", "looking", "not", "calling", "from", "also", "just"}:
        out["name"] = m.group(1).capitalize()

    m = re.search(r"(?:₹|rs\.?|rupees|inr)\s*([\d,]{3,7})|([\d,]{3,7})\s*(?:rupees|rs|₹|per month|a month|monthly)", low)
    if m:
        val = (m.group(1) or m.group(2)).replace(",", "")
        if val.isdigit() and 200 <= int(val) <= 500000:
            out["monthly_bill"] = int(val)
    if "monthly_bill" not in out:
        m = re.search(r"bill (?:is|of|around|about)?\s*([\d,]{3,6})", low)
        if m and m.group(1).replace(",", "").isdigit():
            out["monthly_bill"] = int(m.group(1).replace(",", ""))
    if "monthly_bill" not in out:                       # Telugu: "bill 4500 vastundi", "4500 రూపాయలు"
        m = re.search(r"([\d,]{3,6})\s*(?:రూపాయలు|రూపాయి|రూ\.?|rupayalu|vastundi|vasthundi)", low)
        if m and m.group(1).replace(",", "").isdigit() and 200 <= int(m.group(1).replace(",", "")) <= 500000:
            out["monthly_bill"] = int(m.group(1).replace(",", ""))

    m = re.search(rf"(?:in|from|at|live in|based in)\s+({CITY_LIST})", low) or re.search(rf"\b({CITY_LIST})\b", low)
    if m:
        out["city"] = m.group(1).capitalize()
    elif re.search(r"హైదరాబాద్", text):
        out["city"] = "Hyderabad"
    elif re.search(r"విజయవాడ", text):
        out["city"] = "Vijayawada"
    elif re.search(r"విశాఖపట్నం|వైజాగ్", text):
        out["city"] = "Visakhapatnam"
    elif re.search(r"తిరుపతి", text):
        out["city"] = "Tirupati"
    elif re.search(r"వరంగల్", text):
        out["city"] = "Warangal"
    elif re.search(r"గుంటూరు", text):
        out["city"] = "Guntur"

    if re.search(r"\b(asap|immediately|this week|this month|right away|urgent|jaldi|ventane|vegam)\b|వెంటనే|త్వరగా", low):
        out["timeline"] = "immediate"
    elif re.search(r"\b(next month|few months|soon|in a month|2 months|3 months)\b|వచ్చే నెల", low):
        out["timeline"] = "near"
    elif re.search(r"\b(just looking|exploring|someday|next year|not sure when|later)\b|తరువాత", low):
        out["timeline"] = "later"

    if re.search(r"\b(my house|my home|own (?:a )?house|my shop|my factory|i own|mera ghar|owner|naa illu|na illu)\b|నా ఇల్లు|సొంత ఇల్లు", low):
        out["owner"] = True

    # date / time preference
    d = None
    if re.search(r"\btoday\b|\baaj\b|\bivvala\b|\bivala\b|ఈరోజు|ఇవాళ", low):
        d = today
    elif re.search(r"\btomorrow\b|\bkal\b|\brepu\b|कल|రేపు", low):
        d = today + timedelta(days=1)
    elif re.search(r"day after tomorrow|parso|ellundi|ఎల్లుండి", low):
        d = today + timedelta(days=2)
    else:
        for i, wd in enumerate(WEEKDAYS):
            if re.search(rf"\b{wd}\b", low):
                delta = (i - today.weekday()) % 7 or 7
                d = today + timedelta(days=delta)
                break
    if d:
        out["pref_date"] = d.isoformat()

    h = None
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)", low)
    if m:
        h = int(m.group(1)) % 12 + (12 if m.group(3).startswith("p") else 0)
    else:
        m = re.search(r"\bat\s+(\d{1,2})\b", low)
        if m:
            h = int(m.group(1))
            if h < 9:
                h += 12                      # "at 4" -> 16:00 for business hours
    if h is None:
        m = re.search(r"(\d{1,2})\s*(?:baje|bje|बजे|gantalaku|gantalaki|గంటలకు|గంటకు)", low)
        if m:
            h = int(m.group(1))
            if h < 9:
                h += 12
    if h is None:
        if re.search(r"morning|ఉదయం|పొద్దున", low): h = 10
        elif re.search(r"afternoon|మధ్యాహ్నం", low): h = 14
        elif re.search(r"evening|సాయంత్రం", low): h = 16
    if h is not None and 0 <= h <= 23:
        out["pref_hour"] = h
    return out


NOT_NAMES = {"yes", "no", "ok", "okay", "sure", "hello", "hi", "hey", "thanks", "thank", "nope", "yeah", "haan",
             "nahi", "avunu", "ledu", "vaddu", "sare", "what", "why", "how", "book", "later", "bye", "please"}


def bare_name(text: str) -> str | None:
    """When the agent has just asked for a name, accept a short plain reply like 'Sri' or 'Sri Ram' as the name."""
    t = text.strip().strip(".!,")
    if not re.fullmatch(r"[A-Za-z\u0900-\u097F\u0C00-\u0C7F]+(?: [A-Za-z\u0900-\u097F\u0C00-\u0C7F]+){0,2}", t):
        return None
    if any(w.lower() in NOT_NAMES for w in t.split()):
        return None
    return " ".join(w.capitalize() for w in t.split())


# ------------------------------------------------------------------ intent
def classify_intent(text: str) -> str:
    low = text.lower()
    if re.search(r"\b(bye|goodbye|that's all|thats all|hang up|alvida|dhanyavad bye)\b|अलविदा|ఉంటాను|వెళ్తాను", low):
        return "goodbye"
    if re.search(r"\b(book|schedule|appointment|site visit|visit|demo|meeting|slot|survey|call me back)\b|బుక్|సర్వే|అపాయింట్‌మెంట్", low):
        return "book"
    if re.search(r"\b(expensive|costly|too much|mehnga|mahanga|can't afford|cannot afford|budget|ekkuva|kharidu)\b|महंगा|ఎక్కువ|ఖరీదు|భరించలేను", low):
        return "objection_price"
    if re.search(r"\b(not interested|no thanks|don't call|do not call|stop calling|remove my number)\b|ఆసక్తి లేదు|కాల్ చేయకండి|వద్దు", low):
        return "not_interested"
    if re.search(r"\b(think about it|need time|talk to my|discuss with|spouse|wife|husband|family)\b|ఆలోచించాలి|ఇంట్లో అడగాలి", low):
        return "objection_delay"
    if "?" in text or re.search(r"\b(what|how|which|when|where|why|does|do you|is there|can i|tell me|kitna|kya|kaise|enta|emiti|ela|cheppandi)\b|क्या|कितना|ఎంత|ఏమిటి|ఎలా|ఉందా|చెప్పండి", low):
        return "question"
    if re.search(r"^\s*(yes|yeah|yep|sure|ok|okay|haan|haanji|avunu|sare|please do|go ahead|sounds good)\b|हाँ|అవును|సరే", low):
        return "affirm"
    if re.search(r"^\s*(no|nope|nah|nahi|ledu|vaddu)\b|नहीं|కాదు|లేదు", low):
        return "deny"
    if re.search(r"\b(hello|hi|hey|namaste|namaskaram)\b|नमस्ते|నమస్కారం|హలో", low):
        return "greeting"
    return "statement"


# ------------------------------------------------------------ lead scoring
def score_lead(slots: dict, sentiments: list[float], intents: list[str], booked: bool) -> dict:
    """Transparent additive score 0-100; returns tier + the reasons (explainable for the interview)."""
    pts, why = 0, []

    def add(p, reason):
        nonlocal pts
        pts += p
        why.append(f"+{p} {reason}" if p >= 0 else f"{p} {reason}")

    if slots.get("monthly_bill"):
        add(15, "shared monthly bill (budget signal)")
        if slots["monthly_bill"] >= 3000:
            add(10, "high consumption (>= Rs 3000)")
    if slots.get("owner"):
        add(10, "owns the property (decision authority)")
    tl = slots.get("timeline")
    if tl == "immediate": add(20, "wants to buy soon")
    elif tl == "near": add(10, "near-term timeline")
    elif tl == "later": add(-5, "just exploring")
    if slots.get("city"): add(5, "service-area location given")
    if slots.get("phone"): add(10, "shared phone number")
    if "question" in intents: add(5, "asked product questions (engaged)")
    if "objection_price" in intents: add(-5, "price objection")
    if "objection_delay" in intents: add(-5, "wants to delay")
    if "not_interested" in intents: add(-40, "explicitly not interested")
    if sentiments:
        avg = sum(sentiments) / len(sentiments)
        add(round(avg * 15), f"avg sentiment {avg:+.2f}")
    if booked: add(30, "appointment booked")
    pts = max(0, min(100, pts))
    tier = "hot" if pts >= 65 else "warm" if pts >= 35 else "cold"
    return {"score": pts, "tier": tier, "reasons": why}

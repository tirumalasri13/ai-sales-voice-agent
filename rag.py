"""Tiny RAG layer: markdown -> chunks -> TF-IDF index -> cosine retrieval.

Interface (`Retriever.search`) is the same one a vector DB would expose, so swapping in
multilingual embeddings (bge-m3 / e5) + pgvector is a drop-in change.
Prototype tricks: light stemming, city-name stripping (so "Ludhiana" doesn't hijack the query),
and a small Hindi/Telugu -> English query-expansion map so romanized/native questions still hit the English KB.
"""
import re
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CITIES = r"ludhiana|jalandhar|amritsar|phagwara|patiala|chandigarh|mohali|bathinda|delhi|mumbai|bangalore|bengaluru|pune|chennai|jaipur|hyderabad|secunderabad|vijayawada|visakhapatnam|vizag|guntur|tirupati|warangal|nellore|kurnool"

# multilingual cue -> English expansion terms
EXPAND = [
    (r"kharcha|kitna|kitni|kharchu|enta|dhara|daam|keemat|rate|paisa|कीमत|खर्च|कितना|ఎంత|ధర|ఖర్చు", "price cost system pricing"),
    (r"subsidy|sabsidi|सब्सिडी|సబ్సిడీ|రాయితీ", "government subsidy"),
    (r"emi|loan|kist|vayida|किस्त|लोन|ఈఎంఐ|వాయిదా|లోన్", "financing emi"),
    (r"warranty|guarantee|गारंटी|वारंटी|వారంటీ|గ్యారంటీ", "warranty"),
    (r"kitne din|kab tak|install|lagwana|enni rojulu|कितने दिन|ఎన్ని రోజులు|ఇన్‌స్టాల్", "installation timeline"),
    (r"bijli|बिजली|current bill|karentu|bill|बिल|కరెంటు|బిల్లు|కరెంట్", "savings bill"),
    (r"battery|बैटरी|బ్యాటరీ|inverter", "battery backup"),
    (r"kisan|pump|खेत|రైతు|పంపు|పంప్", "solar water pumps farmers"),
]


def _stem(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _analyzer(text: str):
    toks = [_stem(t) for t in re.findall(r"[a-z]+", text.lower()) if t not in ENGLISH_STOP_WORDS and len(t) > 1]
    return toks + [a + "_" + b for a, b in zip(toks, toks[1:])]


class Retriever:
    def __init__(self, kb_dir: str):
        self.chunks: list[dict] = []
        for f in sorted(Path(kb_dir).glob("*.md")):
            text = f.read_text(encoding="utf-8")
            for section in re.split(r"\n(?=#{1,3} )", text):
                section = section.strip()
                if len(section) < 30:
                    continue
                title = section.splitlines()[0].lstrip("# ").strip()
                self.chunks.append({"source": f.name, "title": title, "text": section})
        self.vec = TfidfVectorizer(analyzer=_analyzer, sublinear_tf=True)
        # title counted twice -> headings weigh more
        self.matrix = self.vec.fit_transform([c["title"] + " " + c["title"] + " " + c["text"] for c in self.chunks])

    def _prepare(self, query: str) -> str:
        q = query.lower()
        stripped = re.sub(rf"\b({CITIES})\b", " ", q)
        if _analyzer(stripped):            # keep original only if nothing else is left
            q = stripped
        extra = " ".join(exp for pat, exp in EXPAND if re.search(pat, query.lower()))
        return q + " " + extra

    def search(self, query: str, k: int = 2, min_score: float = 0.08) -> list[dict]:
        if not self.chunks:
            return []
        sims = cosine_similarity(self.vec.transform([self._prepare(query)]), self.matrix)[0]
        idx = sims.argsort()[::-1][:k]
        return [{**self.chunks[i], "score": round(float(sims[i]), 3)} for i in idx if sims[i] >= min_score]


def first_sentences(text: str, n: int = 2) -> str:
    body = "\n".join(l for l in text.splitlines() if not l.startswith("#"))
    body = re.sub(r"^[-*]\s+", "", body, flags=re.M).replace("\n", " ")
    parts = re.split(r"(?<=[.!?])\s+", body.strip())
    return " ".join(parts[:n])

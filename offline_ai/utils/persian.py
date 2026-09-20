"""Persian text helpers for retrieval, extraction, and grounded answers."""

from __future__ import annotations

import re
import unicodedata

PERSIAN_RANGE = r"\u0600-\u06FF\u200c\u200d"

JALALI_MONTHS = (
    "فروردین",
    "اردیبهشت",
    "خرداد",
    "تیر",
    "مرداد",
    "شهریور",
    "مهر",
    "آبان",
    "آذر",
    "دی",
    "بهمن",
    "اسفند",
)

COUNTRIES = {
    "ایران": "COUNTRY",
    "اسپانیا": "COUNTRY",
    "آمریکا": "COUNTRY",
    "ایالات متحده": "COUNTRY",
    "چین": "COUNTRY",
    "روسیه": "COUNTRY",
    "آلمان": "COUNTRY",
    "فرانسه": "COUNTRY",
    "انگلیس": "COUNTRY",
    "بریتانیا": "COUNTRY",
    "ترکیه": "COUNTRY",
    "عراق": "COUNTRY",
    "افغانستان": "COUNTRY",
    "پاکستان": "COUNTRY",
    "اسرائیل": "COUNTRY",
    "عربستان": "COUNTRY",
    "امارات": "COUNTRY",
    "قطر": "COUNTRY",
    "سوریه": "COUNTRY",
    "لبنان": "COUNTRY",
    "یمن": "COUNTRY",
    "مصر": "COUNTRY",
}

CITIES = {
    "تهران": "LOCATION",
    "اصفهان": "LOCATION",
    "مشهد": "LOCATION",
    "تبریز": "LOCATION",
    "شیراز": "LOCATION",
    "اهواز": "LOCATION",
    "کرج": "LOCATION",
    "قم": "LOCATION",
    "کرمان": "LOCATION",
    "یزد": "LOCATION",
    "مادرید": "LOCATION",
    "بارسلون": "LOCATION",
    "بارسلونا": "LOCATION",
}

QUESTION_WORDS = {
    "کیست",
    "کیه",
    "چیست",
    "چیه",
    "کجاست",
    "کجا",
    "کی",
    "چه",
    "چی",
    "چرا",
    "چگونه",
    "چطور",
    "آیا",
    "کدام",
    "گزارش",
    "بگو",
    "بده",
    "کن",
    "کنید",
    "who",
    "what",
    "where",
    "when",
    "why",
    "how",
}

STOPWORDS = QUESTION_WORDS | {
    "و",
    "در",
    "به",
    "از",
    "با",
    "را",
    "که",
    "این",
    "آن",
    "تا",
    "برای",
    "یک",
    "شده",
    "شد",
    "است",
    "هست",
    "بود",
    "بوده",
    "می",
    "هم",
    "یا",
    "اگر",
    "روی",
    "روی",
    "روز",
    "مهم",
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "by",
    "is",
    "was",
    "are",
    "were",
}

JALALI_DATE_RE = re.compile(
    rf"(?P<d1>\d{{1,2}})(?:\s+و\s+(?P<d2>\d{{1,2}}))?\s+"
    rf"(?P<month>{'|'.join(JALALI_MONTHS)})"
    rf"(?:\s+(?P<year>\d{{3,4}}))?"
)

# Role / relation helpers used by multi-hop QA
ROLE_WORDS = {
    "قاتل",
    "کشنده",
    "متهم",
    "قربانی",
    "مقتول",
}

RELATION_STOPWORDS = STOPWORDS | ROLE_WORDS | {
    "زمانی",
    "تاریخ",
    "پیدا",
    "نشده",
    "نکردیم",
    "دیگر",
    "بعد",
}
TOKEN_RE = re.compile(rf"[{PERSIAN_RANGE}]+|[A-Za-z0-9]+", re.UNICODE)


def contains_persian(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def normalize_persian(text: str) -> str:
    """Unify Arabic/Persian letters, ZWNJ, and whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    trans = str.maketrans(
        {
            "ي": "ی",
            "ك": "ک",
            "ة": "ه",
            "ؤ": "و",
            "إ": "ا",
            "أ": "ا",
            "ٱ": "ا",
            "ۀ": "ه",
            "\u200c": " ",
            "\u200d": "",
            "\u0640": "",
        }
    )
    text = text.translate(trans)
    text = re.sub(r"[\u064b-\u065f\u0670]", "", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_persian(text))


def content_tokens(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in STOPWORDS and len(t) >= 2}


def search_tokens(query: str) -> list[str]:
    """Tokens useful for retrieval; question words are dropped."""
    seen: set[str] = set()
    out: list[str] = []
    for tok in tokenize(query):
        if tok in STOPWORDS or len(tok) < 2:
            continue
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    if out:
        return out
    return [t for t in tokenize(query) if len(t) >= 2]


def extract_jalali_dates(text: str) -> list[str]:
    """Return normalized Jalali date phrases (year optional; ranges like 18 و 19 دی)."""
    text = normalize_persian(text or "")
    out: list[str] = []
    for m in JALALI_DATE_RE.finditer(text):
        d1 = m.group("d1")
        d2 = m.group("d2")
        month = m.group("month")
        year = m.group("year")
        if d2:
            phrase = f"{d1} و {d2} {month}"
        else:
            phrase = f"{d1} {month}"
        if year:
            phrase = f"{phrase} {year}"
        if phrase not in out:
            out.append(phrase)
    return out


def expand_query_tokens(query: str) -> set[str]:
    """Content tokens plus light Persian stemming for retrieval bridging."""
    toks = content_tokens(query)
    extra: set[str] = set()
    for t in list(toks):
        if t.endswith("ی") and len(t) > 3:
            extra.add(t[:-1])
        if t in {"قاتل", "قتل", "کشته", "کشتن"}:
            extra.update({"قتل", "قاتل", "کشته", "رسوند", "رساند"})
        if t in {"پیدا", "نشده", "نیافتیم", "نکردیم"}:
            extra.update({"پیدا", "نیافتیم", "نکردیم", "مفقود"})
    return toks | extra


def token_overlap(query: str, document: str) -> int:
    q = expand_query_tokens(query)
    d = set(tokenize(document))
    # Also count expanded stems present in doc tokens
    return sum(1 for t in q if t in d)


def role_focus_person(query: str) -> str | None:
    """Extract person after role words: قاتل علی → علی."""
    n = normalize_persian(query)
    m = re.search(
        r"(?:قاتل|کشنده|متهم|مقتول|قربانی)\s+([\u0600-\u06ff]+)",
        n,
    )
    if not m:
        return None
    return m.group(1).strip() or None


def insufficient_message(query: str) -> str:
    if contains_persian(query):
        return "شواهد کافی در حافظه ذخیره‌شده وجود ندارد."
    return "Insufficient evidence in stored memory."

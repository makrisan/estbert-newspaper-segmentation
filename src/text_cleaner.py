import re
from html import unescape


# --- HTML CLEANING ---

def clean_html(text: str) -> str:
    """
    Eemaldab HTML tagid ja entity'd.
    Kasutada enne tekstijuppide eraldamist.
    """
    text = unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_p_tags(html_text: str) -> list[str]:
    """
    Võtab <p>...</p> blokkidest teksti välja.
    Säilitab OCR-müra, teeb ainult minimaalse puhastuse.
    """
    html_text = unescape(html_text)
    paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", html_text, flags=re.DOTALL | re.IGNORECASE)

    cleaned = []
    for p in paragraphs:
        p = clean_html(p)
        if p:
            cleaned.append(p)

    return cleaned


# --- TEKSTI NORMALISEERIMINE ---

def clean_text(text: str) -> str:
    """
    Eemaldab alguse/lõpu tühikud ja normaliseerib tühikud.
    Kasutada pärast HTML cleaning'ut.
    """
    if text is None:
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


# --- MÜRA FILTREERIMINE ---

LAYOUT_NOISE_PATTERNS = [
    r"^[A-ZÕÄÖÜŠŽ]$",
    r"^\d+$",
    r"^EESTI PÄEVALEHT$",
    r"^ESTNISKA DAGBLADET$",
    r"^\| ESTNISKA DAGBLADET$",
    r"^Sidan \d+$",
    r"^Lk\.? \d+$",
    r"^Lehekülg \d+$",
    r"^Estniska Dagbladet idag$",
    r"^Kolmapäev, \d{1,2}\. .+ \d{4}.*$",
    r"^[A-ZÕÄÖÜŠŽa-zõäöüšž]+, \d{1,2}\. [a-zõäöüšž]+ \d{4}( Nr\. \d+ \(\d+\))?$",
    r"^Sidan \d+$",
    r"^\|?\s*[A-ZÕÄÖÜŠŽ ]{5,}\s*\|?$",
]


def is_layout_noise(text: str) -> bool:
    """
    Tuvastab ilmselge ajalehe layout-müra.
    """
    for pattern in LAYOUT_NOISE_PATTERNS:
        if re.match(pattern, text.strip(), flags=re.IGNORECASE):
            return True

    if len(text) <= 30 and text.isupper():
        return True

    return False


# --- LAUSEPIIRI KAITSMINE ---

MONTH_NAMES = [
    "jaanuar", "veebruar", "märts", "aprill", "mai", "juuni",
    "juuli", "august", "september", "oktoober", "november", "detsember"
]

ABBREVIATIONS = [
    "Nr.", "nr.", "lk.", "a.", "st.", "nt.", "jm.", "jne.", "Dr.", "Prof.",
    "no.", "jne", "teist-pidi",
]


def protect_abbreviations(text: str) -> str:
    """
    Kaitseb lühendeid ja kuupäevi vale lausepiiri tuvastamise eest.
    Asendab punktid ajutiselt <DOT>-iga.
    """
    for month in MONTH_NAMES:
        text = re.sub(
            rf"(\d{{1,2}})\.\s*\n?\s*({month}\w*)",
            rf"\1<DOT> \2",
            text,
            flags=re.IGNORECASE
        )

    text = re.sub(
        r"(\d{4})\.\s*\n?\s*(aastal|aasta|a)",
        r"\1<DOT> \2",
        text,
        flags=re.IGNORECASE
    )

    for abbreviation in ABBREVIATIONS:
        protected = abbreviation.replace(".", "<DOT>")
        text = text.replace(abbreviation, protected)

    return text


def restore_abbreviations(text: str) -> str:
    """
    Taastab ajutiselt kaitstud punktid tagasi.
    """
    return text.replace("<DOT>", ".")


def split_into_sentences(paragraphs: list[str]) -> list[str]:
    """
    Jagab lõigud lauseteks.
    Kaitseb lühendeid ja kuupäevi vale poolitamise eest.
    """
    sentences = []

    for paragraph in paragraphs:
        if is_layout_noise(paragraph):
            continue

        protected = protect_abbreviations(paragraph)
        parts = re.split(r"(?<=[.!?])\s+", protected)

        for part in parts:
            part = restore_abbreviations(part)
            part = clean_text(part)

            if part and not is_layout_noise(part) and len(part) >= 20:
                sentences.append(part)

    return sentences
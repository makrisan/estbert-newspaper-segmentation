import re
from html import unescape


# --- HTML CLEANING ---

def clean_html(text: str) -> str:
    text = unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_p_tags(html_text: str) -> list[str]:
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
    if text is None:
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


# --- MÜRA FILTREERIMINE ---

# Nädalapäevade eesliide lausemustri jaoks.
# Laiendatud kõigi 7 päevaga — varem oli ainult "Kolmapäev".
_WEEKDAYS = (
    "Esmaspäev|Teisipäev|Kolmapäev|Neljapäev|Reede|Laupäev|Pühapäev"
)

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
    # Kõik nädalapäevad koos kuupäevaga (nt "Kolmapäev, 12. mai 2011")
    rf"^(?:{_WEEKDAYS}),\s+\d{{1,2}}\.\s+.+\d{{4}}.*$",
    # Trükiväljaande päis koos numbriga (nt "Reede, 5. märts 2010 Nr. 12 (345)")
    r"^[A-ZÕÄÖÜŠŽa-zõäöüšž]+,\s+\d{1,2}\.\s+[a-zõäöüšž]+\s+\d{4}(\s+Nr\.\s+\d+\s+\(\d+\))?$",
    r"^\|?\s*[A-ZÕÄÖÜŠŽ ]{5,}\s*\|?$",
]


def is_layout_noise(text: str) -> bool:
    for pattern in LAYOUT_NOISE_PATTERNS:
        if re.match(pattern, text.strip(), flags=re.IGNORECASE):
            return True
    if len(text) <= 30 and text.isupper():
        return True
    return False


# --- LAUSEPIIRI KAITSMINE ---

MONTH_NAMES = [
    "jaanuar", "veebruar", "märts", "aprill", "mai", "juuni",
    "juuli", "august", "september", "oktoober", "november", "detsember",
]

# Kuu lühendid OCR tekstis (nt vana ajaleht kasutab "veebr.", "okt." jne).
MONTH_ABBREVS = [
    "jaan", "veebr", "märts", "apr", "mai", "juuni",
    "juuli", "aug", "sept", "okt", "nov", "dets",
]

# Lühendite nimekiri.
# Järjestus: pikemad enne lühemaid, et vältida osalist asendamist.
# Nt "jne." leitakse enne "jne"-d.
ABBREVIATIONS = [
    # Akadeemilised/ametlikud tiitlid
    "Prof.", "prof.", "Dr.", "dr.", "Mag.", "mag.", "Ins.", "ins.",

    # Loendused ja viited
    "Nr.", "nr.", "lk.", "Lk.", "jj.", "jm.", "jms.",
    "jne.", "jt.", "nt.", "nn.", "nö.", "ns.", "vm.", "vms.", "vs.", "ca.",

    # Tõlkimis- ja toimetamisviited
    "tlk.", "toim.", "koost.",

    # Mõõtühikud ja rahaühikud
    "snt.", "kr.", "mk.", "mln.", "mrd.", "tuh.",
    "km.", "cm.", "mm.", "kg.",

    # Ajalised lühendid
    "saj.", "a.", "st.",

    # Suunad ja viited
    "vastup.", "teist-pidi",

    # Muud sagedased
    "no.", "sealh.", "sh.", "s.o.", "s.t.", "k.a.", "v.a.", "n.-ö.",

    # Isiku initsiaalid — üks suur täht + punkt (nt "C. Kreek", "R. Valgre")
    # Käsitletud eraldi regex'iga allpool (INITIAL_PATTERN)

    # Ajalise planeerimise lühendid (nt kuulutustes)
    "hilj.",   # hiljemalt
    "pühap.",  # pühapäev
    "laup.",   # laupäev
    "kolmap.", # kolmapäev
    "teisip.", # teisipäev
    "neljap.", # neljapäev
    "esmasp.", # esmaspäev
    "tel.",    # telefon (kuulutustes)
    "aad.",    # aadress
]

# Initsiaalimuster: üks suurtäht + punkt + tühik + suurtäht (nimi järel).
# Nt "C. Kreek", "R. Valgre", "N. Liidu".
# re.UNICODE tagab, et [A-ZÕÄÖÜŠŽ] töötab korrektselt.
INITIAL_PATTERN = re.compile(
    r"\b([A-ZÕÄÖÜŠŽ])\.\s+(?=[A-ZÕÄÖÜŠŽ])",
    flags=re.UNICODE,
)

# Järgarvsõnade muster: number + punkt + tühik/reavahetuse + väiketäht.
# Kaitseb nt "12. nädal", "23.\noktoobril" jne.
# Ainult väiketähe ees — suurtäht (nt "1. Sõda") viitab tõenäoliselt lause algusele.
ORDINAL_PATTERN = re.compile(
    r"(\d+)\.\s*\n?\s*(?=[a-zõäöüšžа-я])",
)


def protect_abbreviations(text: str) -> str:
    """
    Kaitseb lühendeid, järgarvusid, initsiaalide ja kuupäevi
    vale lausepiiri tuvastamise eest.
    Asendab punktid ajutiselt <DOT>-iga.

    Reeglite järjekord (tähtis!):
      0. Eemalda Markdown bold-märgid lühendite ümbert ajutiselt (**...**)
      1. Kuu lühendid OCR tekstis (nt "veebr.")
      2. Täiskuunimed koos numbriga (nt "12. jaanuar")
      3. Aastaarvud (nt "1920. aastal")
      4. Isiku initsiaalid (nt "C. Kreek")
      5. Järgarvsõnad (nt "12. nädal", "23.\noktoobril")
      6. Lühendite nimekiri
    """
    # 0. Eemalda Markdown bold-märgid lühendite ümbert ajutiselt,
    #    et regex leiaks lühendi õigesti üles.
    #    nt "**nn.**" → "nn." töötluse ajaks, taastame hiljem
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)

    # 1. Kuu lühendid — käsitleme enne täisnimesid, kuna nad on lühemad
    #    ja võivad segadusse ajada (nt "veebr." enne "veebruar")
    for abbrev in MONTH_ABBREVS:
        # number + punkt + tühik/\n + kuu lühend + punkt (nt "26. veebr.")
        text = re.sub(
            rf"(\d{{1,2}})\.\s*\n?\s*({abbrev})(\.|(?=\s))",
            lambda m: f"{m.group(1)}<DOT> {m.group(2)}<DOT>",
            text,
            flags=re.IGNORECASE,
        )
        # Kuu lühend ilma eelneva numbrita (nt "veebr. 1964")
        text = re.sub(
            rf"\b({abbrev})\.\s+(\d{{4}})",
            rf"\1<DOT> \2",
            text,
            flags=re.IGNORECASE,
        )

    # 2. Täiskuunimed koos eelneva numbriga (nt "12. jaanuar")
    for month in MONTH_NAMES:
        text = re.sub(
            rf"(\d{{1,2}})\.\s*\n?\s*({month}\w*)",
            rf"\1<DOT> \2",
            text,
            flags=re.IGNORECASE,
        )

    # 3. Aastaarvud: "1920. aastal", "2003. a"
    text = re.sub(
        r"(\d{4})\.\s*\n?\s*(aastal|aasta|a\b)",
        r"\1<DOT> \2",
        text,
        flags=re.IGNORECASE,
    )

    # 4. Isiku initsiaalid: "C. Kreek", "N. Liidu"
    #    Asendame ainult punkti, säilitame tühiku
    text = INITIAL_PATTERN.sub(r"\1<DOT> ", text)

    # 5. Järgarvsõnad väiketähe või \n ees (nt "12. nädal", "23.\noktoobril")
    text = ORDINAL_PATTERN.sub(r"\1<DOT> ", text)

    # 6. Lühendite nimekiri (pikemad enne lühemaid)
    for abbreviation in ABBREVIATIONS:
        protected = abbreviation.replace(".", "<DOT>")
        text = text.replace(abbreviation, protected)

    return text


def restore_abbreviations(text: str) -> str:
    """Taastab ajutiselt kaitstud punktid tagasi."""
    return text.replace("<DOT>", ".")


def split_into_sentences(paragraphs: list[str]) -> list[str]:
    """
    Jagab lõigud lauseteks.
    Kaitseb lühendeid, initsiaalide ja kuupäevi vale poolitamise eest.

    Märkus OCR hüüumärkide kohta:
      Mõnes näites esineb "meloodiliste! ja kõrgetel" — hüüumärk
      keskel lause on OCR müra (peaks olema koma vms).
      Seda ei saa usaldusväärselt automaatselt parandada, kuna
      mõnikord on hüüumärk päriselt lause lõpp.
      Praegu aktsepteerime selle piiranguna — kui soovid, saame
      lisada heuristika "! + väiketäht → tõenäoliselt müra".
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

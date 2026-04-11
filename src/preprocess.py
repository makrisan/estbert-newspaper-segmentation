import re

# eemaldab alguse/lõpu tühikud
# asendab mitu tühikut ühega
def clean_text(text):
    if text is None:
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

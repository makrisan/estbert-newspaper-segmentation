import json
from src.preprocess import clean_text

# avab .jsonl faili
# loeb read ükshaaval
# muudab rea JSON-objektiks
# kontrollib, et text ja label oleks olemas
# puhastab teksti
# tagastab kõik kirjed listina
def load_jsonl(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            # kontrollime, et vajalikud väljad olemas oleks
            if "text" not in item or "label" not in item:
                raise ValueError(f"Puuduv väli kirjes: {item}")

            item["text"] = clean_text(item["text"])
            data.append(item)

    return data
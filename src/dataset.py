import json
import os
from src.preprocess import clean_text

def load_jsonl(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Faili ei leitud: {path}")

    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"Vigane JSON rida: {line}")
                continue

            if "id" not in item or "text" not in item or "label" not in item:
                raise ValueError(f"Puuduv väli kirjes: {item}")

            if item["label"] not in [0, 1]:
                raise ValueError(f"Vale label väärtus: {item['label']}")

            item["text"] = clean_text(item["text"])
            data.append(item)

    return data


if __name__ == "__main__":
    data = load_jsonl("data/sample/sample_dataset.jsonl")
    print(f"Laetud {len(data)} kirjet\n")

    for item in data[:3]:
        print(item)
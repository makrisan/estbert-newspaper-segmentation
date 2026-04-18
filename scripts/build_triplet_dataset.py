import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = BASE_DIR / "data" / "sample" / "sample_dataset.jsonl"
OUTPUT_PATH = BASE_DIR / "data" / "sample" / "triplet_dataset.jsonl"


def load_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(f"Faili ei leitud: {path}")

    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    data = load_jsonl(INPUT_PATH)

    triplets = []

    for i, item in enumerate(data):
        prev_text = data[i - 1]["text"] if i > 0 else ""
        curr_text = item["text"]
        next_text = data[i + 1]["text"] if i < len(data) - 1 else ""

        label = item["label"]
        if i == 0:
            label = 0

        triplet_item = {
            "id": item["id"],
            "prev": prev_text,
            "curr": curr_text,
            "next": next_text,
            "label": label
        }

        triplets.append(triplet_item)

    save_jsonl(triplets, OUTPUT_PATH)

    print("Triplet-dataset loodud.")
    print("Sisendkirjeid:", len(data))
    print("Väljundkirjeid:", len(triplets))
    print("Salvestatud faili:", OUTPUT_PATH)

    ones = sum(1 for x in triplets if x["label"] == 1)
    zeros = sum(1 for x in triplets if x["label"] == 0)
    print(f"label 1: {ones}, label 0: {zeros}")


if __name__ == "__main__":
    main()

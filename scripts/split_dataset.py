# standard ratio
# TRAIN_RATIO = 0.8 → õppimine
# VAL_RATIO = 0.1 → kontroll treeningu ajal
# TEST_RATIO = 0.1 → lõplik hindamine

import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.dataset import load_jsonl

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "sample" / "triplet_dataset.jsonl"
TRAIN_PATH = BASE_DIR / "data" / "sample" / "train.jsonl"
VAL_PATH = BASE_DIR / "data" / "sample" / "val.jsonl"
TEST_PATH = BASE_DIR / "data" / "sample" / "test.jsonl"


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def label_stats(dataset, name):
    ones = sum(1 for x in dataset if x["label"] == 1)
    zeros = sum(1 for x in dataset if x["label"] == 0)
    total = len(dataset)

    print(f"{name} -> kokku: {total}")
    print(f"label 0: {zeros} ({zeros / total:.2%})")
    print(f"label 1: {ones} ({ones / total:.2%})")

    if ones > 0:
        print(f"0:1 suhe = {zeros / ones:.2f}:1")


def main():
    data = load_jsonl(str(DATA_PATH))
    labels = [x["label"] for x in data]

    train_data, temp_data = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    temp_labels = [x["label"] for x in temp_data]

    val_data, test_data = train_test_split(
        temp_data,
        test_size=0.5,
        random_state=42,
        stratify=temp_labels
    )

    save_jsonl(train_data, TRAIN_PATH)
    save_jsonl(val_data, VAL_PATH)
    save_jsonl(test_data, TEST_PATH)

    print("Kokku näiteid:", len(data))
    print("\nSPLIT STATISTIKA:")
    label_stats(train_data, "Train")
    label_stats(val_data, "Val")
    label_stats(test_data, "Test")


if __name__ == "__main__":
    main()

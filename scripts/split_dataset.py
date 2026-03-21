import json
import random
from src.dataset import load_jsonl

# standard ratio
# TRAIN_RATIO = 0.8 → õppimine
# VAL_RATIO = 0.1 → kontroll treeningu ajal
# TEST_RATIO = 0.1 → lõplik hindamine

DATA_PATH = "../data/sample/sample_dataset.jsonl"
TRAIN_PATH = "../data/sample/train.jsonl"
VAL_PATH = "../data/sample/val.jsonl"
TEST_PATH = "../data/sample/test.jsonl"

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def label_stats(dataset, name):
    ones = sum(1 for x in dataset if x["label"] == 1)
    zeros = sum(1 for x in dataset if x["label"] == 0)
    print(f"{name} → kokku: {len(dataset)}, label 1: {ones}, label 0: {zeros}")


def main():
    data = load_jsonl(DATA_PATH)

    positives = [x for x in data if x["label"] == 1]
    negatives = [x for x in data if x["label"] == 0]

    random.shuffle(positives)
    random.shuffle(negatives)

    # Positiivsed käsitsi: 5 train, 1 val, 1 test
    pos_train = positives[:5]
    pos_val = positives[5:6]
    pos_test = positives[6:7]

    # Negatiivsed proportsionaalselt
    neg_train = negatives[:34]
    neg_val = negatives[34:38]
    neg_test = negatives[38:]

    train_data = pos_train + neg_train
    val_data = pos_val + neg_val
    test_data = pos_test + neg_test

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    save_jsonl(train_data, TRAIN_PATH)
    save_jsonl(val_data, VAL_PATH)
    save_jsonl(test_data, TEST_PATH)

    print("Kokku näiteid:", len(data))
    print("Positiivseid:", len(positives))
    print("Negatiivseid:", len(negatives))
    print("\nSPLIT STATISTIKA:")
    label_stats(train_data, "Train")
    label_stats(val_data, "Val")
    label_stats(test_data, "Test")


if __name__ == "__main__":
    main()

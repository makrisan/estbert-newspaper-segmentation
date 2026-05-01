# standard ratio
# TRAIN_RATIO = 0.8 → õppimine
# VAL_RATIO = 0.1 → kontroll treeningu ajal
# TEST_RATIO = 0.1 → lõplik hindamine

import json
from pathlib import Path
import sys
import argparse

from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import load_jsonl

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "large_triplet_dataset.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "large"


def save_jsonl(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Split triplet dataset into train/val/test")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input JSONL dataset path"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for train/val/test JSONL files"
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def extract_group_id(example):
    raw_id = str(example.get("id", ""))
    if "_" in raw_id:
        return raw_id.rsplit("_", 1)[0]
    return raw_id


def split_groups(group_ids, train_ratio, val_ratio, test_ratio, seed):
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.0")

    unique_groups = sorted(set(group_ids))

    splitter_1 = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    train_group_idx, temp_group_idx = next(
        splitter_1.split(unique_groups, groups=unique_groups)
    )

    train_groups = {unique_groups[i] for i in train_group_idx}
    temp_groups = [unique_groups[i] for i in temp_group_idx]

    temp_ratio = val_ratio + test_ratio
    val_share_within_temp = val_ratio / temp_ratio if temp_ratio else 0.5

    splitter_2 = GroupShuffleSplit(
        n_splits=1,
        train_size=val_share_within_temp,
        random_state=seed,
    )
    val_group_idx, test_group_idx = next(
        splitter_2.split(temp_groups, groups=temp_groups)
    )

    val_groups = {temp_groups[i] for i in val_group_idx}
    test_groups = {temp_groups[i] for i in test_group_idx}

    return train_groups, val_groups, test_groups


def check_no_group_leakage(train_data, val_data, test_data):
    train_groups = {extract_group_id(x) for x in train_data}
    val_groups = {extract_group_id(x) for x in val_data}
    test_groups = {extract_group_id(x) for x in test_data}

    train_val_overlap = train_groups & val_groups
    train_test_overlap = train_groups & test_groups
    val_test_overlap = val_groups & test_groups

    if train_val_overlap or train_test_overlap or val_test_overlap:
        raise RuntimeError("Group leakage detected across splits")

    print("\nGroup leakage check: OK")
    print(
        f"Groups -> train: {len(train_groups)}, val: {len(val_groups)}, test: {len(test_groups)}"
    )


def main():
    args = parse_args()

    input_path = args.input_path
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    test_path = output_dir / "test.jsonl"

    data = load_jsonl(input_path)

    group_ids = [extract_group_id(x) for x in data]
    train_groups, val_groups, test_groups = split_groups(
        group_ids,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )

    train_data, val_data, test_data = [], [], []
    for item in data:
        group_id = extract_group_id(item)
        if group_id in train_groups:
            train_data.append(item)
        elif group_id in val_groups:
            val_data.append(item)
        else:
            test_data.append(item)

    save_jsonl(train_data, train_path)
    save_jsonl(val_data, val_path)
    save_jsonl(test_data, test_path)

    print("Sisendfail:", input_path)
    print("Väljundkaust:", output_dir)
    print("Kokku näiteid:", len(data))
    print("\nSPLIT STATISTIKA:")
    label_stats(train_data, "Train")
    label_stats(val_data, "Val")
    label_stats(test_data, "Test")
    check_no_group_leakage(train_data, val_data, test_data)


if __name__ == "__main__":
    main()

import json
import os

import torch
from torch.utils.data import Dataset

from src.text_cleaner import clean_text

# MAX_LENGTH määrab, kui pikad tekstid tokenizer mudelile annab.
# EstBERT maksimaalne sisendpikkus on 512 tokenit.
MAX_LENGTH = 512


def build_triplet_input(prev_text: str, curr_text: str, next_text: str, sep_token: str) -> str:
    """
    Ehitab BERT sisendi formaadis: prev [SEP] curr [SEP] next.
    Kui prev on tühi (artikli algus), on see BERT-ile oluline signaal label=1 jaoks.
    """
    return f"{prev_text or ''} {sep_token} {curr_text or ''} {sep_token} {next_text or ''}"


def load_jsonl(path) -> list[dict]:
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

            if "id" not in item or "label" not in item:
                raise ValueError(f"Puuduv väli kirjes: {item}")

            if item["label"] not in [0, 1]:
                raise ValueError(f"Vale label väärtus: {item['label']}")

            if not all(k in item for k in ["prev", "curr", "next"]):
                raise ValueError(f"Kirjel puuduvad triplet-väljad: {item}")

            item["prev"] = clean_text(item["prev"])
            item["curr"] = clean_text(item["curr"])
            item["next"] = clean_text(item["next"])

            data.append(item)

    return data


class NewsDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=MAX_LENGTH):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sep_token = tokenizer.sep_token if tokenizer.sep_token else "[SEP]"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        model_input = build_triplet_input(
            item["prev"],
            item["curr"],
            item["next"],
            self.sep_token
        )

        encoding = self.tokenizer(
            model_input,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.long)
        }
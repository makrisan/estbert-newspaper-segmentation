import json
import os

import torch
from torch.utils.data import Dataset

from src.preprocess import clean_text

# MAX_LENGTH määrab, kui pikad tekstid tokenizer mudelile annab.
# EstBERT maksimaalne sisendpikkus on 512 tokenit.
MAX_LENGTH = 512


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

            if "id" not in item or "label" not in item:
                raise ValueError(f"Puuduv väli kirjes: {item}")

            if item["label"] not in [0, 1]:
                raise ValueError(f"Vale label väärtus: {item['label']}")

            # Vana formaat
            if "text" in item:
                item["text"] = clean_text(item["text"])

            # Triplet-formaat
            elif all(k in item for k in ["prev", "curr", "next"]):
                item["prev"] = clean_text(item["prev"])
                item["curr"] = clean_text(item["curr"])
                item["next"] = clean_text(item["next"])

            else:
                raise ValueError(
                    f"Kirjel puudub kas 'text' või triplet-väljad: {item}"
                )

            data.append(item)

    return data


def build_triplet_model_input(prev_text, curr_text, next_text, sep_token="[SEP]"):
    """
    Ühtne koht triplet-inputi loomiseks.
    Sama funktsiooni saab kasutada nii treeningus kui inference'is,
    et formaat ei läheks lahku.
    """
    prev_text = prev_text or ""
    curr_text = curr_text or ""
    next_text = next_text or ""

    return f"{prev_text} {sep_token} {curr_text} {sep_token} {next_text}"

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

        # Triplet-formaat: eelmine, praegune, järgmine
        if all(k in item for k in ["prev", "curr", "next"]):
            model_input = build_triplet_model_input(
                item["prev"],
                item["curr"],
                item["next"],
                sep_token=self.sep_token
            )
        else:
            model_input = item.get("text", "")

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

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from src.model_setup import load_model

    data = load_jsonl("data/sample/triplet_dataset.jsonl")
    tokenizer, _ = load_model()

    dataset = NewsDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    batch = next(iter(dataloader))

    print(f"Dataset suurus: {len(dataset)} kirjet")
    print(f"Batch suurus:   {batch['input_ids'].shape[0]}\n")
    print(f"input_ids kuju:      {batch['input_ids'].shape}")
    print(f"attention_mask kuju: {batch['attention_mask'].shape}")
    print(f"labels kuju:         {batch['labels'].shape}")
    print(f"labels väärtused:    {batch['labels']}")

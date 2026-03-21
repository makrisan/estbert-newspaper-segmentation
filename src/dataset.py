import json
import os
import torch
from torch.utils.data import Dataset
from src.preprocess import clean_text

# MAX_LENGTH määrab kui pikad tekstid tokenizer aktsepteerib
# EstBERT max on 512, kasutame sama
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

            if "id" not in item or "text" not in item or "label" not in item:
                raise ValueError(f"Puuduv väli kirjes: {item}")

            if item["label"] not in [0, 1]:
                raise ValueError(f"Vale label väärtus: {item['label']}")

            item["text"] = clean_text(item["text"])
            data.append(item)

    return data


class ArticleDataset(Dataset):
    # võtab sisse load_jsonl() väljundi ja tokenizeri
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # tokeniseerime teksti
        encoded = self.tokenizer(
            item["text"],
            max_length=MAX_LENGTH,  # maksimaalne pikkus
            truncation=True,        # kui pikem kui 512, lõika ära
            padding="max_length",   # kui lühem kui 512, täida nullidega
            return_tensors="pt"     # tagasta PyTorch tensorina
        )

        return {
            # squeeze eemaldab ülearuse dimensiooni: [1, 512] -> [512]
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.long)
        }


class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(item["label"], dtype=torch.long)
        }


if __name__ == "__main__":
    from src.model_setup import load_model
    from torch.utils.data import DataLoader

    # laeme andmed ja mudeli
    data = load_jsonl("data/sample/sample_dataset.jsonl")
    tokenizer, _ = load_model()

    # loome dataseti ja dataloaderi
    dataset = ArticleDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # kontrollime esimest batchi
    batch = next(iter(dataloader))

    print(f"Dataset suurus: {len(dataset)} kirjet")
    print(f"Batch suurus:   {batch['input_ids'].shape[0]}\n")
    print(f"input_ids kuju:     {batch['input_ids'].shape}")
    print(f"attention_mask kuju: {batch['attention_mask'].shape}")
    print(f"labels kuju:        {batch['labels'].shape}")
    print(f"labels väärtused:   {batch['labels']}")

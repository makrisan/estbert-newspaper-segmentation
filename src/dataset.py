import json
import os

import torch
from torch.utils.data import Dataset

from src.text_cleaner import clean_text

MAX_LENGTH = 512

# Token budget igale osale (3 osa + 3 special tokenit [CLS] ja 2x [SEP])
# 512 - 3 = 509 tokenit tekstile, jagatuna kolmeks võrdseks osaks
PREV_MAX_TOKENS = 170
CURR_MAX_TOKENS = 170
NEXT_MAX_TOKENS = 169


def build_triplet_input(prev_text: str, curr_text: str, next_text: str, sep_token: str) -> str:
    """
    Ehitab BERT sisendi formaadis: prev [SEP] curr [SEP] next.
    """
    return f"{prev_text or ''} {sep_token} {curr_text or ''} {sep_token} {next_text or ''}"


def tokenize_with_budget(tokenizer, prev: str, curr: str, next_: str, max_length: int) -> dict:
    """
    Tokeniseerib triplet kontrollitud token eelarvega.

    Probleem vana lähenemisega:
        tokenizer(full_string, truncation=True, max_length=512)
        lõikab teksti lihtsalt 512 tokeni pealt maha — see tähendab
        et 'next' kaob täielikult kui 'prev' + 'curr' on pikad.
        Cross-segment BERT paber näitas et ilma parema kontekstita
        langeb F1 64-lt 20-le.

    Lahendus:
        Tokeniseerime iga osa eraldi, piirame token arvu,
        ja ühendame käsitsi koos special tokenitega.

    Args:
        tokenizer: EstBERT tokenizer
        prev:      eelnev lause
        curr:      praegune lause (kõige tähtsam)
        next_:     järgmine lause
        max_length: maksimaalne kogupikkus (512)

    Returns:
        Dict input_ids ja attention_mask tensoriga
    """
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    def encode_part(text: str, max_tokens: int) -> list[int]:
        """Tokeniseerib teksti ja piirab pikkust, ilma special tokeniteta."""
        ids = tokenizer(
            text or "",
            add_special_tokens=False,
            truncation=True,
            max_length=max_tokens,
        )["input_ids"]
        return ids

    prev_ids = encode_part(prev, PREV_MAX_TOKENS)
    curr_ids = encode_part(curr, CURR_MAX_TOKENS)
    next_ids = encode_part(next_, NEXT_MAX_TOKENS)

    # [CLS] prev [SEP] curr [SEP] next
    input_ids = [cls_id] + prev_ids + [sep_id] + curr_ids + [sep_id] + next_ids

    # Padding max_length peale
    padding_length = max_length - len(input_ids)
    attention_mask = [1] * len(input_ids) + [0] * padding_length
    input_ids = input_ids + [pad_id] * padding_length

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        encoding = tokenize_with_budget(
            self.tokenizer,
            prev=item["prev"],
            curr=item["curr"],
            next_=item["next"],
            max_length=self.max_length,
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": torch.tensor(item["label"], dtype=torch.long),
        }
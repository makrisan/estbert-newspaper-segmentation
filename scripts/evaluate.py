from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader

from src.dataset import load_jsonl, NewsDataset
from src.model_setup import load_model

BASE_DIR = Path(__file__).resolve().parents[1]

TEST_PATH = BASE_DIR / "data" / "sample" / "test.jsonl"
MODEL_PATH = BASE_DIR / "models" / "model.pt"

# --- HYPERPARAMETERS ---
# Mitu näidel mudel korraga läbi töötleb
# train size / batch size = x batxhi ühe epohhi kohta
BATCH_SIZE = 4


def main():
    print("1. Alustan evaluation'it")

    test_data = load_jsonl(str(TEST_PATH))
    print("Test size:", len(test_data))

    tokenizer, model = load_model()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modeli faili ei leitud: {MODEL_PATH}")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    print("2. Treenitud mudel laetud")

    test_dataset = NewsDataset(test_data, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print("\nEVALUATION TULEMUSED:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nConfusion matrix:")
    print(cm)

if __name__ == "__main__":
    main()

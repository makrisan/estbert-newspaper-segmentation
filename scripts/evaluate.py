# pip install scikit-learn
from src.dataset import load_jsonl, NewsDataset
from src.model_setup import load_model
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- PATHID ---
TEST_PATH = "../data/sample/test.jsonl"
MODEL_PATH = "../models/final_model"

# --- HYPERPARAMETERS ---
# Mitu näidel mudel korraga läbi töötleb
# train size / batch size = x batxhi ühe epohhi kohta
BATCH_SIZE = 4


def main():
    print("1. Alustan evaluation'it")

    # lae testandmed
    test_data = load_jsonl(TEST_PATH)
    print("Test size:", len(test_data))

    # lae tokenizer ja mudel
    tokenizer, model = load_model(MODEL_PATH)

    # lae treenitud mudel
    print("2. Treenitud mudel laetud")

    # loo dataset ja dataloader
    test_dataset = NewsDataset(test_data, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    # ennustamine
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

    # metrikad
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    # väljund
    print("\nEVALUATION TULEMUSED:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nConfusion matrix:")
    print(cm)

    # lihtne kommentaar
    print("\nLühike kommentaar:")
    print("Need tulemused näitavad, kui hästi mudel eristab artikli alguseid ja mitte-alguseid.")
    print("Kuna testandmestik on väga väike, tuleb tulemusi tõlgendada ettevaatlikult.")
    print("Samas annab see esialgse hinnangu, kas treeningpipeline töötab ja mudel õpib midagi kasulikku.")

    # maatriksi lugemine: [[1 2][3 4]] [[TN  FP][FN  TP]]
    # 1 negatiivset näidet ennustati õigesti negatiivseks
    # 2 negatiivset näidet ennustati valesti positiivseks
    # 3 positiivne näide ennustati valesti negatiivseks
    # 4 positiivset näidet ennustati õigesti positiivseks

if __name__ == "__main__":
    main()

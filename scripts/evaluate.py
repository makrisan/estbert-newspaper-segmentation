from pathlib import Path
import sys
import argparse
import json

import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_jsonl, NewsDataset
from src.model_setup import load_model

VAL_PATH = PROJECT_ROOT / "data" / "large" / "val.jsonl"
TEST_PATH = PROJECT_ROOT / "data" / "large" / "test.jsonl"
MODEL_PATH = PROJECT_ROOT / "models" / "final_model"
REPORT_PATH = PROJECT_ROOT / "data" / "output" / "evaluation_report.json"

# --- HYPERPARAMETERS ---
# Mitu näidel mudel korraga läbi töötleb
# train size / batch size = x batxhi ühe epohhi kohta
BATCH_SIZE = 4


def parse_args():
    parser = argparse.ArgumentParser(description="Hindab mudelit threshold tuning'uga")
    parser.add_argument("--val-path", type=Path, default=VAL_PATH)
    parser.add_argument("--test-path", type=Path, default=TEST_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def predict_probs(data, tokenizer, model, device, batch_size) -> tuple[list, list]:
    dataset = NewsDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    all_labels, all_probs = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = softmax(outputs.logits, dim=1)[:, 1]

            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return all_labels, all_probs


def compute_metrics_for_threshold(labels, probs, threshold: float) -> dict:
    preds = [1 if p >= threshold else 0 for p in probs]
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "confusion_matrix": cm.tolist(),
    }


def find_best_threshold(labels, probs) -> dict:
    best = None
    for i in range(5, 96):
        metrics = compute_metrics_for_threshold(labels, probs, threshold=i / 100)
        if best is None or metrics["f1"] > best["f1"]:
            best = metrics
    return best


def print_metrics(title: str, metrics: dict):
    print(f"\n{title}")
    print(f"Threshold: {metrics['threshold']:.2f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")


def main():
    args = parse_args()
    val_path = resolve_path(args.val_path)
    test_path = resolve_path(args.test_path)
    model_path = resolve_path(args.model_path)
    report_path = resolve_path(args.report_path)

    print("1. Alustan hindamist")
    val_data = load_jsonl(val_path)
    test_data = load_jsonl(test_path)
    print(f"Val size:  {len(val_data)}")
    print(f"Test size: {len(test_data)}")

    # lae tokenizer ja mudel
    tokenizer, model = load_model(model_path)

    # lae treenitud mudel
    print("2. Treenitud mudel laetud")

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    val_labels, val_probs = predict_probs(val_data, tokenizer, model, device, args.batch_size)
    test_labels, test_probs = predict_probs(test_data, tokenizer, model, device, args.batch_size)

    default_metrics = compute_metrics_for_threshold(test_labels, test_probs, threshold=0.5)
    best_val = find_best_threshold(val_labels, val_probs)
    tuned_test_metrics = compute_metrics_for_threshold(test_labels, test_probs, threshold=best_val["threshold"])

    print_metrics("TEST (default threshold=0.50)", default_metrics)
    print_metrics("VALIDATION parim threshold", best_val)
    print_metrics("TEST (validation threshold'iga)", tuned_test_metrics)

    # Märkus: confusion matrix loetakse [[TN, FP], [FN, TP]]
    print(f"\nParim threshold inference'i jaoks: {best_val['threshold']:.2f}")
    print("Kasuta seda väärtust inference.py --threshold argumendina.")

    report = {
        "val_path": str(val_path),
        "test_path": str(test_path),
        "model_path": str(model_path),
        "default_test": default_metrics,
        "best_val_threshold": best_val,
        "tuned_test": tuned_test_metrics,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nRaport salvestatud: {report_path}")


    # maatriksi lugemine: [[1 2][3 4]] [[TN  FP][FN  TP]]
    # 1 negatiivset näidet ennustati õigesti negatiivseks
    # 2 negatiivset näidet ennustati valesti positiivseks
    # 3 positiivne näide ennustati valesti negatiivseks
    # 4 positiivset näidet ennustati õigesti positiivseks

if __name__ == "__main__":
    main()

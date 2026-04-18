import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from transformers import Trainer, TrainingArguments

from src.dataset import load_jsonl, NewsDataset
from src.model_setup import load_model

BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = BASE_DIR / "data" / "sample" / "train.jsonl"
VAL_PATH = BASE_DIR / "data" / "sample" / "val.jsonl"
MODELS_DIR = BASE_DIR / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
FINAL_MODEL_DIR = MODELS_DIR / "final_model"
STATE_DICT_PATH = MODELS_DIR / "model.pt"
CLASS_DISTRIBUTION_PATH = BASE_DIR / "class_distribution.png"
CONFUSION_MATRIX_PATH = BASE_DIR / "confusion_matrix.png"

BATCH_SIZE = 4
EPOCHS = 2
LEARNING_RATE = 2e-5


def calculate_class_stats(data):
    labels = [item["label"] for item in data]
    num_zeros = labels.count(0)
    num_ones = labels.count(1)

    ratio = num_zeros / num_ones if num_ones > 0 else float("inf")
    class_1_weight = num_zeros / num_ones if num_ones > 0 else 1.0

    return num_zeros, num_ones, ratio, class_1_weight


def visualize_class_distribution(num_zeros, num_ones, save_path):
    plt.figure(figsize=(8, 5))
    plt.bar(
        ["Label 0 (Not Start)", "Label 1 (Article Start)"],
        [num_zeros, num_ones]
    )
    plt.ylabel("Count")
    plt.title("Class Distribution in Training Data")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Class distribution chart saved to {save_path}")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0
    )
    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Not Start", "Article Start"])
    ax.set_yticklabels(["Not Start", "Article Start"])

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Confusion matrix saved to {save_path}")


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(model.device)
        )
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


def main():
    print("1. Alustan treeningut")

    train_data = load_jsonl(str(TRAIN_PATH))
    val_data = load_jsonl(str(VAL_PATH))

    print("Train size:", len(train_data))
    print("Val size:", len(val_data))

    num_zeros, num_ones, ratio, class_1_weight = calculate_class_stats(train_data)

    print("\nClass distribution:")
    print(f"  Label 0 (Not article start): {num_zeros}")
    print(f"  Label 1 (Article start): {num_ones}")

    if num_ones > 0:
        print(f"  Ratio (0/1): {ratio:.2f}")
    else:
        print("  Ratio (0/1): inf")

    print(f"  Class 1 weight for loss: {class_1_weight:.2f}\n")

    visualize_class_distribution(num_zeros, num_ones, CLASS_DISTRIBUTION_PATH)

    tokenizer, model = load_model()

    train_dataset = NewsDataset(train_data, tokenizer, max_length=512)
    val_dataset = NewsDataset(val_data, tokenizer, max_length=512)

    class_weights = torch.tensor([1.0, class_1_weight], dtype=torch.float)

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        logging_dir=str(BASE_DIR / "logs"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("\n=== Starting Training with Hugging Face Trainer ===\n")
    trainer.train()

    print("\n=== Final Evaluation on Validation Set ===")
    eval_results = trainer.evaluate()

    print("\nFinal Metrics:")
    print(f"  Accuracy:  {eval_results['eval_accuracy']:.4f}")
    print(f"  F1-score:  {eval_results['eval_f1']:.4f}")
    print(f"  Precision: {eval_results['eval_precision']:.4f}")
    print(f"  Recall:    {eval_results['eval_recall']:.4f}")

    print("\n=== Generating Confusion Matrix ===")
    predictions = trainer.predict(val_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, CONFUSION_MATRIX_PATH)

    print("\nConfusion Matrix:")
    print("                    Predicted")
    print("                  Not Start | Article Start")
    print(f"True Not Start:      {cm[0][0]:3d}   |   {cm[0][1]:3d}")
    print(f"True Article Start:  {cm[1][0]:3d}   |   {cm[1][1]:3d}")

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Hugging Face formaat
    model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

    # state_dict formaat inference/evaluate jaoks
    torch.save(model.state_dict(), STATE_DICT_PATH)

    print(f"\n✓ Final model saved to {FINAL_MODEL_DIR}")
    print(f"✓ State dict saved to {STATE_DICT_PATH}")


if __name__ == "__main__":
    main()

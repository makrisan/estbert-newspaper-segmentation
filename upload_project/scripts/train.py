from pathlib import Path
import sys
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from transformers import Trainer, TrainingArguments

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_jsonl, NewsDataset
from src.model_setup import load_model

# --- PATHS ---
TRAIN_PATH = PROJECT_ROOT / "data" / "large" / "train.jsonl"
VAL_PATH = PROJECT_ROOT / "data" / "large" / "val.jsonl"
CLASS_DISTRIBUTION_PATH = PROJECT_ROOT / "data" / "output" / "class_distribution.png"
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "data" / "output" / "confusion_matrix.png"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"
FINAL_MODEL_DIR = PROJECT_ROOT / "models" / "final_model"

# --- HYPERPARAMETERS ---
BATCH_SIZE = 4
EPOCHS = 2
LEARNING_RATE = 2e-5
DEFAULT_CLASS1_WEIGHT_MULTIPLIER = 1.0


def parse_args() -> argparse.Namespace:
    """
    Parsib käsurea argumendid treenimise konfigureerimiseks.

    Returns:
        Namespace objekt järgmiste väljadega:
            train_path:              treenimisandmete JSONL faili asukoht
            val_path:                valideerimisandmete JSONL faili asukoht
            final_model_dir:         kaust lõpliku mudeli salvestamiseks
            checkpoint_dir:          kaust vahepealsete checkpointide jaoks
            log_dir:                 kaust treenimislogide jaoks
            class_plot_path:         klassijaotuse graafiku salvestuskoht
            cm_plot_path:            confusion matrixi salvestuskoht
            class1_weight_multiplier: klassi 1 kaalu kordaja tasakaalustamiseks
    """
    parser = argparse.ArgumentParser(description="Train article-boundary classifier")
    parser.add_argument("--train-path", type=Path, default=TRAIN_PATH)
    parser.add_argument("--val-path", type=Path, default=VAL_PATH)
    parser.add_argument("--final-model-dir", type=Path, default=FINAL_MODEL_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    parser.add_argument("--class-plot-path", type=Path, default=CLASS_DISTRIBUTION_PATH)
    parser.add_argument("--cm-plot-path", type=Path, default=CONFUSION_MATRIX_PATH)
    parser.add_argument(
        "--class1-weight-multiplier",
        type=float,
        default=DEFAULT_CLASS1_WEIGHT_MULTIPLIER,
        help="Klassi 1 kaalu kordaja (nt 2.0 või 3.0 tugevamaks karistuseks).",
    )
    return parser.parse_args()


def resolve_path(arg_path: Path) -> Path:
    """
    Teisendab suhtelise tee absoluutseks projekti juure suhtes.

    Args:
        arg_path: käsurea argumendist saadud tee

    Returns:
        Absoluutne Path objekt
    """
    return arg_path if arg_path.is_absolute() else PROJECT_ROOT / arg_path


def calculate_class_stats(data: list[dict]) -> tuple[int, int, float, float]:
    """
    Arvutab klasside jaotuse statistika treenimisandmestiku põhjal.

    Klassi 1 kaal arvutatakse valemiga num_zeros / num_ones,
    mis kompenseerib tasakaalustamatust — haruldasemale klassile
    antakse kõrgem kaal loss funktsioonis.

    Args:
        data: JSONL kirjete nimekiri, igaühel väli "label" (0 või 1)

    Returns:
        Neliku (num_zeros, num_ones, ratio, class_1_weight):
            num_zeros:     label 0 näidete arv
            num_ones:      label 1 näidete arv
            ratio:         suhe num_zeros / num_ones
            class_1_weight: soovituslik kaal klassile 1
    """
    labels = [item["label"] for item in data]
    num_zeros = labels.count(0)
    num_ones = labels.count(1)
    ratio = num_zeros / num_ones if num_ones > 0 else float("inf")
    class_1_weight = num_zeros / num_ones if num_ones > 0 else 1.0
    return num_zeros, num_ones, ratio, class_1_weight


def visualize_class_distribution(
    num_zeros: int,
    num_ones: int,
    save_path: Path,
) -> None:
    """
    Salvestab klasside jaotuse tulpdiagrammina PNG failina.

    Kasulik tasakaalustamatuse visuaalseks hindamiseks
    enne treenimist ja lõputöö dokumentatsiooni jaoks.

    Args:
        num_zeros: label 0 näidete arv
        num_ones:  label 1 näidete arv
        save_path: väljundfaili asukoht (kaust luuakse vajadusel)
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.bar(
        ["Label 0 (pole algus)", "Label 1 (artikli algus)"],
        [num_zeros, num_ones]
    )
    plt.ylabel("Arv")
    plt.title("Klasside jaotus treeningandmestikus")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Klasside jaotuse graafik salvestatud: {save_path}")


def compute_metrics(eval_pred) -> dict:
    """
    Arvutab hindamismetrikad Hugging Face Trainer API jaoks.

    Kutsutakse automaatselt iga eval_strategy sammu lõpus.
    Kasutab binary average kuna tegemist on kaheklassilise
    probleemiga, kus label 1 (artikli algus) on põhifookus.

    Args:
        eval_pred: EvalPrediction objekt väljadega
                   predictions (logits) ja label_ids

    Returns:
        Dict järgmiste võtmetega:
            accuracy:  õigete ennustuste osakaal kõigist
            f1:        F1-skoor label 1-le (peamine mõõdik)
            precision: täpsus label 1-le (mitu ennustust oli õige)
            recall:    täielikkus label 1-le (mitu tegelikku leiti)
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, predictions)

    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def plot_confusion_matrix(cm: np.ndarray, save_path: Path) -> None:
    """
    Salvestab confusion matrixi visuaalina PNG failina.

    Confusion matrix näitab nelja arvu:
        TN (True Negative):  label 0 õigesti ennustatud
        FP (False Positive): label 0 valesti ennustatud label 1-ks
        FN (False Negative): label 1 valesti ennustatud label 0-ks
        TP (True Positive):  label 1 õigesti ennustatud

    Args:
        cm:        2x2 numpy maatriks confusion_matrix() väljundist
        save_path: väljundfaili asukoht (kaust luuakse vajadusel)
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pole algus", "Artikli algus"])
    ax.set_yticklabels(["Pole algus", "Artikli algus"])
    ax.set_xlabel("Ennustatud")
    ax.set_ylabel("Tegelik")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix salvestatud: {save_path}")


class WeightedTrainer(Trainer):
    """
    Hugging Face Trainer'i laiendus kaalutud loss funktsiooniga.

    Standardne Trainer kasutab kaalumata CrossEntropyLoss'i,
    mis tugevalt tasakaalustamata andmestikul (94% label 0,
    6% label 1) õpib ennustama peaaegu alati label 0-i ja
    saavutab kõrge accuracy aga madala F1 label 1-le.

    WeightedTrainer annab haruldasemale klassile (label 1)
    kõrgema kaalu, mistõttu vead label 1 ennustamisel
    karistatakse rohkem ja mudel õpib artiklipiire paremini
    tuvastama.

    Attributes:
        class_weights: tensor kujuga [2] kaaludega [w0, w1],
                       kus w1 > w0 tasakaalustamatu andmestiku korral
    """

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        """
        Initsialiseerib WeightedTrainer'i kaalutud loss funktsiooniga.

        Args:
            class_weights: tensor kujuga [2], nt [1.0, 15.72]
                           kus teine väärtus on klassi 1 kaal
            *args:         edastatakse Trainer.__init__-ile
            **kwargs:      edastatakse Trainer.__init__-ile
        """
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Arvutab kaalutud CrossEntropyLoss'i.

        Ylikkirjutab Trainer.compute_loss() meetodi et lisada
        klassi kaalud loss funktsiooni. Kõrgem kaal klassile 1
        tähendab et vale ennustus artikli alguse kohta
        karistatakse rohkem kui vale ennustus jätku kohta.

        Args:
            model:          treenimisalune mudel
            inputs:         batch sisendid (input_ids, attention_mask, labels)
            return_outputs: kui True, tagastab ka mudeli väljundid

        Returns:
            loss tensor, või (loss, outputs) kui return_outputs=True
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(model.device)
        )(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    """
    Peafunktsioon — koordineerib kogu treenimise pipeline'i.

    Sammud:
        1. Laeb treenimis- ja valideerimisandmestiku
        2. Arvutab klasside jaotuse ja optimaalse class_weight
        3. Visualiseerib klasside jaotuse
        4. Laeb eeltreenitud EstBERT mudeli
        5. Treenib WeightedTrainer'iga (kaalutud loss)
        6. Hindab lõplikku mudelit (F1, Precision, Recall, Accuracy)
        7. Salvestab confusion matrixi ja lõpliku mudeli

    Peamine hindamismõõdik on F1-skoor label 1-le,
    kuna täpsus (accuracy) on tasakaalustamata andmestikul
    eksitav — 94% täpsust saab ka alati label 0 ennustades.
    """
    print("1. Alustan treeningut")
    args = parse_args()

    train_path = resolve_path(args.train_path)
    val_path = resolve_path(args.val_path)
    final_model_dir = resolve_path(args.final_model_dir)
    checkpoint_dir = resolve_path(args.checkpoint_dir)
    log_dir = resolve_path(args.log_dir)
    class_plot_path = resolve_path(args.class_plot_path)
    cm_plot_path = resolve_path(args.cm_plot_path)

    train_data = load_jsonl(train_path)
    val_data = load_jsonl(val_path)
    print(f"Train size: {len(train_data)}")
    print(f"Val size:   {len(val_data)}")

    num_zeros, num_ones, ratio, class_1_weight = calculate_class_stats(train_data)
    print(f"\nKlasside jaotus:")
    print(f"  Label 0: {num_zeros}")
    print(f"  Label 1: {num_ones}")
    print(f"  Suhe (0/1): {ratio:.2f}" if num_ones > 0 else "  Suhe (0/1): inf")
    print(f"  Klassi 1 kaal: {class_1_weight:.2f}")

    if args.class1_weight_multiplier <= 0:
        raise ValueError("--class1-weight-multiplier peab olema > 0")

    effective_class1_weight = class_1_weight * args.class1_weight_multiplier
    print(
        f"  Efektiivne klassi 1 kaal: {effective_class1_weight:.2f} "
        f"(kordaja={args.class1_weight_multiplier:.2f})\n"
    )

    visualize_class_distribution(num_zeros, num_ones, save_path=class_plot_path)

    tokenizer, model = load_model()
    train_dataset = NewsDataset(train_data, tokenizer)
    val_dataset = NewsDataset(val_data, tokenizer)

    class_weights = torch.tensor([1.0, effective_class1_weight], dtype=torch.float)

    training_args = TrainingArguments(
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        output_dir=str(checkpoint_dir),
        logging_dir=str(log_dir),
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

    print("=== Alustan treeningut ===\n")
    trainer.train()

    print("\n=== Lõplik hindamine valideerimisandmestikul ===")
    eval_results = trainer.evaluate()
    print(f"  Accuracy:  {eval_results['eval_accuracy']:.4f}")
    print(f"  F1-score:  {eval_results['eval_f1']:.4f}")
    print(f"  Precision: {eval_results['eval_precision']:.4f}")
    print(f"  Recall:    {eval_results['eval_recall']:.4f}")

    predictions = trainer.predict(val_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, save_path=cm_plot_path)

    print("\nConfusion Matrix:")
    print("                    Ennustatud")
    print("                  Pole algus | Artikli algus")
    print(f"Tegelik pole algus:    {cm[0][0]:3d}   |   {cm[0][1]:3d}")
    print(f"Tegelik artikli algus: {cm[1][0]:3d}   |   {cm[1][1]:3d}")

    final_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"\n✓ Mudel salvestatud: {final_model_dir}")


if __name__ == "__main__":
    main()

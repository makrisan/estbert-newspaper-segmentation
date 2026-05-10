from pathlib import Path
import sys
import argparse

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_setup import load_model
from src.dataset import build_triplet_input
from src.text_cleaner import extract_p_tags, split_into_sentences

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "estdagbladet_20110316_lk.txt"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
OUTPUT_PATH = OUTPUT_DIR / "estdagbladet_20110316_lk_predicted.txt"
DEBUG_OUTPUT_PATH = OUTPUT_DIR / "estdagbladet_20110316_lk_debug.txt"
MODEL_PATH = PROJECT_ROOT / "models" / "final_model"

MAX_LENGTH = 512
DEFAULT_THRESHOLD = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description="Sliding window inference artikli piiride leidmiseks")
    parser.add_argument("--input-path", type=Path, default=INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--debug-output-path", type=Path, default=DEBUG_OUTPUT_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Boundary tõenäosuse lävi (default: 0.5). "
             "Kasuta evaluate.py leitud parimat threshold'i.",
    )
    return parser.parse_args()


def predict_boundaries(
    sentences: list[str],
    tokenizer,
    model,
    device,
    threshold: float,
) -> list[dict]:
    """
    Liigub sliding window'ga üle lausete ja ennustab artikli piire.
    Kasutab threshold'i argumendina, mitte hardcoded väärtust.
    """
    results = []

    for i, curr in enumerate(sentences):
        prev_text = sentences[i - 1] if i > 0 else ""
        next_text = sentences[i + 1] if i < len(sentences) - 1 else ""

        if i == 0:
            pred_label = 0
            boundary_prob = 0.0
        else:
            sep_token = tokenizer.sep_token if tokenizer.sep_token else "[SEP]"
            model_input = build_triplet_input(prev_text, curr, next_text, sep_token)

            encoding = tokenizer(
                model_input,
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )

            with torch.no_grad():
                outputs = model(
                    input_ids=encoding["input_ids"].to(device),
                    attention_mask=encoding["attention_mask"].to(device)
                )

            boundary_prob = torch.softmax(outputs.logits, dim=1)[0][1].item()
            pred_label = 1 if boundary_prob >= threshold else 0

        results.append({
            "prev": prev_text,
            "curr": curr,
            "next": next_text,
            "pred_label": pred_label,
            "boundary_prob": boundary_prob
        })

    return results


def write_debug_output(results: list[dict], output_path: Path):
    """
    Kirjutab debug-väljundi faili — prev/curr/next kontekst ja boundary tõenäosus.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(results):
            f.write("=" * 80 + "\n")
            f.write(f"Index: {i}\n")
            f.write(f"Ennustatud label: {item['pred_label']}\n")
            f.write(f"Boundary tõenäosus: {item['boundary_prob']:.4f}\n\n")
            f.write(f"PREV:\n{item['prev']}\n\n")
            f.write(f"CURR:\n{item['curr']}\n\n")
            f.write(f"NEXT:\n{item['next']}\n\n")


def write_tagged_output(results: list[dict], output_path: Path):
    """
    Kirjutab segmenteeritud väljundi faili <p> tagedega.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        article_started = False

        for i, item in enumerate(results):
            if i == 0 or item["pred_label"] == 1:
                if article_started:
                    f.write("</p>\n")
                f.write("<p>\n")
                article_started = True

            f.write(item["curr"] + "\n")

        if article_started:
            f.write("</p>\n")


def main():
    args = parse_args()

    input_path = args.input_path if args.input_path.is_absolute() else PROJECT_ROOT / args.input_path
    model_path = args.model_path if args.model_path.is_absolute() else PROJECT_ROOT / args.model_path

    print("1. Loen sisendfaili")
    if not input_path.exists():
        raise FileNotFoundError(f"Faili ei leitud: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Ridu failis: {len(lines)}")

    all_sentences = []
    for line in lines:
        if "\t" not in line:
            continue
        _, html_text = line.split("\t", 1)
        paragraphs = extract_p_tags(html_text)
        all_sentences.extend(split_into_sentences(paragraphs))

    print(f"Kokku lauseid pärast eeltöötlust: {len(all_sentences)}")

    print("2. Laen mudeli")
    tokenizer, model = load_model(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"3. Ennustan piire (threshold={args.threshold})")
    results = predict_boundaries(all_sentences, tokenizer, model, device, args.threshold)

    total_boundaries = sum(x["pred_label"] for x in results)
    print(f"Leitud piire: {total_boundaries}")

    if total_boundaries == 0:
        print("Mudel ei ennustanud ühtegi piiri.")
    else:
        print("\nEnnustatud piirid:")
        for item in results:
            if item["pred_label"] == 1:
                print("-" * 60)
                print(f"Tõenäosus: {item['boundary_prob']:.4f}")
                print(f"PREV: {item['prev'][:200]}")
                print(f"CURR: {item['curr'][:200]}")

    print("4. Salvestan väljundi")
    write_tagged_output(results, args.output_path)
    write_debug_output(results, args.debug_output_path)

    print(f"Väljundfail: {args.output_path}")
    print(f"Debug fail:  {args.debug_output_path}")


if __name__ == "__main__":
    main()
import html
import json
import os
import re
from pathlib import Path

import torch

from src.dataset import build_triplet_model_input
from src.model_setup import load_model

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = BASE_DIR / "data" / "raw" / "estdagbladet_20110316_lk.txt"
OUTPUT_PATH = BASE_DIR / "data" / "output" / "estdagbladet_20110316_lk_predicted.txt"
DEBUG_PATH = BASE_DIR / "data" / "output" / "estdagbladet_20110316_lk_debug.jsonl"
MODEL_PATH = BASE_DIR / "models" / "model.pt"

MAX_LENGTH = 512
BOUNDARY_THRESHOLD = 0.50
MIN_SENTENCE_LENGTH = 3


def extract_p_tags(html_text: str) -> list[str]:
    """
    Võtab <p>...</p> blokkidest teksti välja.
    Kui <p> tag'e pole, tagastab kogu teksti ühe lõiguna.
    """
    paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", html_text, flags=re.DOTALL | re.IGNORECASE)

    if not paragraphs:
        paragraphs = [html_text]

    cleaned = []
    for p in paragraphs:
        p = html.unescape(p)
        p = re.sub(r"<[^>]+>", " ", p)
        p = re.sub(r"\s+", " ", p).strip()

        if p:
            cleaned.append(p)

    return cleaned


def is_layout_noise(text: str) -> bool:
    """
    Heuristika ilmselge layout-müra eemaldamiseks.
    Ei tohi olla liiga agressiivne.
    """
    if not text:
        return True

    text = text.strip()

    exact_patterns = [
        r"^EESTI PÄEVALEHT$",
        r"^ESTNISKA DAGBLADET$",
        r"^Estniska Dagbladet idag$",
        r"^Sidan \d+$",
        r"^\d+$",
        r"^[IVXLCDM]+$",
        r"^\(?\d+\)?$",
        r"^\d+\.$",
        r"^[\-_=*•·]+$",
        r"^Kolmapäev, \d{1,2}\. märts \d{4}$",
    ]

    for pattern in exact_patterns:
        if re.match(pattern, text, flags=re.IGNORECASE):
            return True

    # Väga lühike numbrite/sümbolite fragment
    if len(text) <= 3 and re.fullmatch(r"[\W\d]+", text):
        return True

    # Väga palju numbreid/sümboleid ja vähe päris teksti
    letters = sum(ch.isalpha() for ch in text)
    non_letters = len(text) - letters
    if len(text) > 0 and letters < 3 and non_letters >= letters:
        return True

    return False


def split_into_sentences(paragraphs: list[str]) -> list[str]:
    """
    Mõõdukalt parem lausejagamine OCR-tekstile.
    Endiselt heuristiline, aga parem kui täiesti toores split.
    """
    sentences = []

    abbreviations = [
        "hr.", "pr.", "dr.", "jne.", "jm.", "jt.", "s.t.", "st.", "nr.", "lk."
    ]

    for paragraph in paragraphs:
        if is_layout_noise(paragraph):
            continue

        text = paragraph.strip()

        for abbr in abbreviations:
            text = text.replace(abbr, abbr.replace(".", "<DOT>"))

        parts = re.split(r"(?<=[.!?])\s+", text)

        restored_parts = []
        for part in parts:
            part = part.replace("<DOT>", ".")
            part = re.sub(r"\s+", " ", part).strip()

            if not part:
                continue

            if is_layout_noise(part):
                continue

            restored_parts.append(part)

        merged_parts = []
        for part in restored_parts:
            if (
                merged_parts
                and len(part) < MIN_SENTENCE_LENGTH
            ):
                merged_parts[-1] = merged_parts[-1] + " " + part
            else:
                merged_parts.append(part)

        sentences.extend(merged_parts)

    return sentences


def load_trained_model(model_path: Path):
    """
    Laeb base tokenizeri + mudeli ja seejärel fine-tuned kaalud.
    Eeldab, et model.pt sisaldab state_dict'i.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Modeli faili ei leitud: {model_path}")

    tokenizer, model = load_model()
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)

    return tokenizer, model


def predict_boundaries(sentences: list[str], tokenizer, model, device) -> list[dict]:
    """
    Liigub sliding window'ga üle lausete ja ennustab,
    kas piir on prev ja curr vahel.
    """
    results = []
    sep_token = tokenizer.sep_token if tokenizer.sep_token else "[SEP]"

    for i, curr in enumerate(sentences):
        prev_text = sentences[i - 1] if i > 0 else ""
        next_text = sentences[i + 1] if i < len(sentences) - 1 else ""

        if i == 0:
            pred_label = 0
            prob_boundary = 0.0
        else:
            model_input = build_triplet_model_input(
                prev_text,
                curr,
                next_text,
                sep_token=sep_token
            )

            encoding = tokenizer(
                model_input,
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )

            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            probs = torch.softmax(outputs.logits, dim=1)
            prob_boundary = probs[0, 1].item()
            pred_label = 1 if prob_boundary >= BOUNDARY_THRESHOLD else 0

        results.append({
            "index": i,
            "prev": prev_text,
            "curr": curr,
            "next": next_text,
            "pred_label": pred_label,
            "prob_boundary": round(prob_boundary, 6)
        })

    return results


def write_tagged_output(results: list[dict], output_path: Path):
    """
    Kirjutab väljundi faili.
    Kui pred_label == 1, alustame uut <p> plokki.
    """
    os.makedirs(output_path.parent, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        article_started = False

        for i, item in enumerate(results):
            curr = item["curr"]
            pred = item["pred_label"]

            if i == 0 or pred == 1:
                if article_started:
                    f.write("</p>\n")
                f.write("<p>\n")
                article_started = True

            f.write(curr + "\n")

        if article_started:
            f.write("</p>\n")


def write_debug_output(results: list[dict], debug_path: Path):
    os.makedirs(debug_path.parent, exist_ok=True)

    with open(debug_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    print("1. Loen sisendfaili")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Faili ei leitud: {INPUT_PATH}")

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print("Ridu failis:", len(lines))

    all_sentences = []

    for line in lines:
        if "\t" not in line:
            continue

        doc_id, html_text = line.split("\t", 1)

        paragraphs = extract_p_tags(html_text)
        sentences = split_into_sentences(paragraphs)

        all_sentences.extend(sentences)

    print("Kokku lauseid pärast eeltöötlust:", len(all_sentences))

    print("2. Laen mudeli")
    tokenizer, model = load_trained_model(MODEL_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("3. Ennustan piire sliding window meetodil")
    results = predict_boundaries(all_sentences, tokenizer, model, device)

    total_boundaries = sum(x["pred_label"] for x in results)
    print("Leitud piire:", total_boundaries)

    found_boundary_examples = [x for x in results if x["pred_label"] == 1][:10]
    if found_boundary_examples:
        print("\nEsimesed boundary ennustused:")
        for item in found_boundary_examples:
            print(
                f"[{item['index']}] prob={item['prob_boundary']:.4f} | curr={item['curr'][:120]}"
            )
    else:
        print("\nBoundary ennustusi ei leitud.")

    print("4. Salvestan väljundi")
    write_tagged_output(results, OUTPUT_PATH)
    write_debug_output(results, DEBUG_PATH)

    print("Valmis.")
    print("Väljundfail:", OUTPUT_PATH)
    print("Debug-fail:", DEBUG_PATH)


if __name__ == "__main__":
    main()

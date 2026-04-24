import os
import re
import json
import torch

from src.model_setup import load_model
from src.dataset import build_triplet_input

INPUT_PATH = "../data/raw/estdagbladet_20110316_lk.txt"
OUTPUT_PATH = "../data/output/estdagbladet_20110316_lk_predicted.txt"
MODEL_PATH = "../models/final_model"

MAX_LENGTH = 512


def extract_p_tags(html_text: str) -> list[str]:
    """
    Võtab <p>...</p> blokkidest teksti välja.
    Säilitame OCR-müra, teeme ainult minimaalse puhastuse.
    """
    paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", html_text, flags=re.DOTALL | re.IGNORECASE)

    cleaned = []
    for p in paragraphs:
        # eemalda üleliigne HTML escaping
        p = p.replace("&amp;", "&")
        p = p.replace("&quot;", '"')
        p = p.replace("&#39;", "'")

        # eemalda võimalikud üleliigsed tagid lõigu seest
        p = re.sub(r"<[^>]+>", "", p)

        # whitespace normaliseerimine
        p = re.sub(r"\s+", " ", p).strip()

        if p:
            cleaned.append(p)

    return cleaned


def is_layout_noise(text: str) -> bool:
    """
    Väga lihtne heuristika, et eemaldada ilmselge layout-müra.
    Ära tee seda liiga agressiivseks.
    """
    noise_patterns = [
        r"^EESTI PÄEVALEHT$",
        r"^ESTNISKA DAGBLADET$",
        r"^Kolmapäev, \d{1,2}\. märts \d{4}$",
        r"^\d+$",
        r"^Sidan \d+$",
        r"^Estniska Dagbladet idag$",
    ]

    for pattern in noise_patterns:
        if re.match(pattern, text, flags=re.IGNORECASE):
            return True

    return False


def split_into_sentences(paragraphs: list[str]) -> list[str]:
    """
    Lihtne lausejagamine.
    OCR tõttu ideaalne ei ole, aga prototüübi jaoks sobib.
    """
    sentences = []

    for paragraph in paragraphs:
        if is_layout_noise(paragraph):
            continue

        # jaga lausete lõppude pealt
        parts = re.split(r"(?<=[.!?])\s+", paragraph)

        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)

    return sentences


def build_model_input(prev_text: str, curr_text: str, next_text: str, tokenizer) -> str:
    """
    Nüüd kasutab inference sama loogikat nagu dataset.
    """
    sep_token = tokenizer.sep_token if tokenizer.sep_token else "[SEP]"
    return build_triplet_input(prev_text, curr_text, next_text, sep_token)


def predict_boundaries(sentences: list[str], tokenizer, model, device) -> list[dict]:
    """
    Liigub sliding window'ga üle lausete ja ennustab,
    kas piir on prev ja curr vahel.
    """
    results = []

    for i, curr in enumerate(sentences):
        prev_text = sentences[i - 1] if i > 0 else ""
        next_text = sentences[i + 1] if i < len(sentences) - 1 else ""

        if i == 0:
            pred_label = 0
        else:
            model_input = build_model_input(prev_text, curr, next_text, tokenizer)

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

            pred_label = torch.argmax(outputs.logits, dim=1).item()

        results.append({
            "prev": prev_text,
            "curr": curr,
            "next": next_text,
            "pred_label": pred_label
        })

    return results


def write_tagged_output(results: list[dict], output_path: str):
    """
    Kirjutab väljundi faili.
    Kui pred_label == 1, alustame uut <p> plokki.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

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


def main():
    print("1. Loen sisendfaili")

    if not os.path.exists(INPUT_PATH):
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
    tokenizer, model = load_model(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("3. Ennustan piire sliding window meetodil")
    results = predict_boundaries(all_sentences, tokenizer, model, device)

    total_boundaries = sum(x["pred_label"] for x in results)
    print("Leitud piire:", total_boundaries)

    print("4. Salvestan väljundi")
    write_tagged_output(results, OUTPUT_PATH)

    print("Valmis.")
    print("Väljundfail:", OUTPUT_PATH)


if __name__ == "__main__":
    main()

import os
import re
import torch

from src.model_setup import load_model
from src.dataset import build_triplet_input

INPUT_PATH = "../data/raw/estdagbladet_20110316_lk.txt"
OUTPUT_PATH = "../data/output/estdagbladet_20110316_lk_predicted.txt"
DEBUG_OUTPUT_PATH = "../data/output/estdagbladet_20110316_lk_debug.txt"
MODEL_PATH = "../models/final_model"

MAX_LENGTH = 512

MONTH_NAMES = [
    "jaanuar", "veebruar", "märts", "aprill", "mai", "juuni",
    "juuli", "august", "september", "oktoober", "november", "detsember"
]

ABBREVIATIONS = [
    "Nr.", "nr.", "lk.", "a.", "st.", "nt.", "jm.", "jne.", "Dr.", "Prof."
]


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
    # üldisemad noise patternid
    noise_patterns = [
        # üksikud tähed / leheküljenumbrid
        r"^[A-ZÕÄÖÜŠŽ]$",
        r"^\d+$",

        # kuupäevaread eri nädalapäevade ja kuudega
        r"^[A-ZÕÄÖÜŠŽa-zõäöüšž]+, \d{1,2}\. [a-zõäöüšž]+ \d{4}( Nr\. \d+ \(\d+\))?$",

        # lehekülje viited
        r"^Sidan \d+$",
        r"^Lk\.? \d+$",
        r"^Lehekülg \d+$",

        # väga lühikesed ajalehe päise tüüpi read
        r"^\|?\s*[A-ZÕÄÖÜŠŽ ]{5,}\s*\|?$",
    ]

    for pattern in noise_patterns:
        if re.match(pattern, text, flags=re.IGNORECASE):
            return True

    # Väga lühikesed ainult suurtähtedest koosnevad päiseread
    if len(text) <= 30 and text.isupper():
        return True

    return False


def protect_sentence_split_cases(text: str) -> str:
    """
    Kaitseb OCR-tekstis kohti, mille pealt ei tohiks lauset poolitada.
    Näiteks kuupäevad nagu "16. märts" ja lühendid nagu "Nr. 11".
    """
    # kaitse kuupäevad ka juhul kui on reavahetus või käänded (nt novembril, jaanuaris)
    for month in MONTH_NAMES:
        text = re.sub(
            rf"(\d{{1,2}})\.\s*\n?\s*({month}\w*)",
            rf"\1<DOT> \2",
            text,
            flags=re.IGNORECASE
        )
    # kaitse aastad nagu "2011. aastal" jne
    text = re.sub(
        r"(\d{4})\.\s*\n?\s*(aastal|aasta|a)",
        r"\1<DOT> \2",
        text,
        flags=re.IGNORECASE
    )
    # lühendid
    for abbreviation in ABBREVIATIONS:
        protected = abbreviation.replace(".", "<DOT>")
        text = text.replace(abbreviation, protected)

    return text


def restore_sentence_split_cases(text: str) -> str:
    """
    Taastab ajutiselt kaitstud punktid tagasi tavalisteks punktideks.
    """
    return text.replace("<DOT>", ".")


def split_into_sentences(paragraphs: list[str]) -> list[str]:
    """
    Lihtne lausejagamine.
    OCR tõttu ideaalne ei ole, aga prototüübi jaoks sobib.
    """
    sentences = []

    for paragraph in paragraphs:
        if is_layout_noise(paragraph):
            continue

        protected_paragraph = protect_sentence_split_cases(paragraph)

        # jaga lausete lõppude pealt
        parts = re.split(r"(?<=[.!?])\s+", protected_paragraph)

        for part in parts:
            part = restore_sentence_split_cases(part)
            part = part.strip()

            if part and not is_layout_noise(part):
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
            boundary_prob = 0.0
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

            probabilities = torch.softmax(outputs.logits, dim=1)
            boundary_prob = probabilities[0][1].item()
            pred_label = torch.argmax(probabilities, dim=1).item()

        results.append({
            "prev": prev_text,
            "curr": curr,
            "next": next_text,
            "pred_label": pred_label,
            "boundary_prob": boundary_prob
        })

    return results


def write_debug_output(results: list[dict], output_path: str):
    """
    Kirjutab debug-väljundi faili.
    Näitab iga ennustuse puhul prev/curr/next konteksti ja boundary tõenäosust.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(results):
            f.write("=" * 80 + "\n")
            f.write(f"Index: {i}\n")
            f.write(f"Pred label: {item['pred_label']}\n")
            f.write(f"Boundary probability: {item['boundary_prob']:.4f}\n\n")

            f.write("PREV:\n")
            f.write(item["prev"] + "\n\n")

            f.write("CURR:\n")
            f.write(item["curr"] + "\n\n")

            f.write("NEXT:\n")
            f.write(item["next"] + "\n\n")


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
    boundary_candidates = [x for x in results if x["pred_label"] == 1]

    print("Leitud piire:", total_boundaries)

    if boundary_candidates:
        print("\nEnnustatud boundary kohad:")
        for item in boundary_candidates:
            print("-" * 80)
            print(f"Boundary probability: {item['boundary_prob']:.4f}")
            print(f"PREV: {item['prev'][:200]}")
            print(f"CURR: {item['curr'][:200]}")
    else:
        print("Mudel ei ennustanud ühtegi boundary't.")

    print("4. Salvestan väljundi")
    write_tagged_output(results, OUTPUT_PATH)
    write_debug_output(results, DEBUG_OUTPUT_PATH)

    print("Valmis.")
    print("Väljundfail:", OUTPUT_PATH)
    print("Debug fail:", DEBUG_OUTPUT_PATH)


if __name__ == "__main__":
    main()

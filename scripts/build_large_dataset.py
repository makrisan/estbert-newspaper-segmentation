import json
import random
import re
from html import unescape
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "segmented"
OUTPUT_PATH = PROJECT_ROOT / "data" / "large_triplet_dataset.jsonl"
SAMPLE_OUTPUT_PATH = PROJECT_ROOT / "data" / "output" / "large_dataset_sample.jsonl"
MAX_SAMPLE_ROWS = 25


def collect_input_files(input_path: Path | None = None) -> tuple[list[Path], list[Path]]:
    """
    Leiab sisendfailid nii uues kui legacy kaustastruktuuris.
    Tagastab (failid, kontrollitud asukohad).
    """
    checked: list[Path] = []

    candidates = [
        input_path,
        INPUT_DIR,
        PROJECT_ROOT / "data" / "raw" / "segmented.txt",
        PROJECT_ROOT / "scripts" / "data" / "segmented",  # legacy üksik fail
        PROJECT_ROOT / "scripts" / "data" / "segmented.txt",
    ]

    for candidate in candidates:
        if candidate is None:
            continue

        candidate = Path(candidate)
        checked.append(candidate)

        if not candidate.exists():
            continue

        if candidate.is_file():
            return [candidate], checked

        # Eelistame .txt faile, aga kui neid pole, võtame kõik failid.
        txt_files = sorted(p for p in candidate.glob("*.txt") if p.is_file())
        if txt_files:
            return txt_files, checked

        any_files = sorted(p for p in candidate.iterdir() if p.is_file())
        if any_files:
            return any_files, checked

    return [], checked


def clean_text(text: str) -> str:
    """
    Eemaldab HTML tagid ja lihtsama OCR/HTML müra.
    """
    text = unescape(text)

    # võta <p> tagide sisu, kui neid on
    paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", text, flags=re.DOTALL | re.IGNORECASE)

    if paragraphs:
        text = "\n".join(paragraphs)
    else:
        text = re.sub(r"<[^>]+>", " ", text)

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def is_layout_noise(text: str) -> bool:
    """
    Eemaldab ilmselge ajalehe layout-müra.
    Ära tee liiga agressiivseks.
    """
    noise_patterns = [
        r"^A$",
        r"^\d+$",
        r"^EESTI PÄEVALEHT$",
        r"^ESTNISKA DAGBLADET$",
        r"^\| ESTNISKA DAGBLADET$",
        r"^Sidan \d+$",
        r"^Estniska Dagbladet idag$",
        r"^Kolmapäev, \d{1,2}\. .+ \d{4}.*$",
    ]

    for pattern in noise_patterns:
        if re.match(pattern, text.strip(), flags=re.IGNORECASE):
            return True

    return False


def split_into_chunks(html_text: str) -> list[str]:
    """
    Teeb ühe artikli HTML tekstist puhastatud tekstijupid.
    Praegu kasutame <p> plokke kui tekstijuppe, aga mitte artikli piiride leidmiseks.
    Artikli piir tuleb ID/rea järgi.
    """
    html_text = unescape(html_text)

    paragraphs = re.findall(
        r"<p[^>]*>(.*?)</p>",
        html_text,
        flags=re.DOTALL | re.IGNORECASE
    )

    chunks = []

    for paragraph in paragraphs:
        cleaned = clean_text(paragraph)

        if not cleaned:
            continue

        if is_layout_noise(cleaned):
            continue

        if len(cleaned) < 3:
            continue

        chunks.append(cleaned)

    return chunks


def parse_segmented_file(file_path: Path) -> list[dict]:
    """
    Loeb segmenteeritud faili.
    Eeldus:
    iga rida = üks artikkel
    formaat:
    article_id<TAB><p>...</p><p>...</p>
    """
    examples = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            if "\t" not in line:
                print(f"Skipping line without tab: {file_path.name}:{line_number}")
                continue

            article_id, html_text = line.split("\t", 1)

            chunks = split_into_chunks(html_text)

            if not chunks:
                continue

            for chunk_idx, chunk in enumerate(chunks):
                label = 1 if chunk_idx == 0 else 0

                examples.append({
                    "id": f"{article_id}_{chunk_idx}",
                    "text": chunk,
                    "label": label
                })

    return examples


def build_triplets(examples: list[dict]) -> list[dict]:
    triplets = []

    for i, item in enumerate(examples):
        triplets.append({
            "id": item["id"],
            "prev": examples[i - 1]["text"] if i > 0 else "",
            "curr": item["text"],
            "next": examples[i + 1]["text"] if i < len(examples) - 1 else "",
            "label": 0 if i == 0 else item["label"]
        })

    return triplets


def save_jsonl(data: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_sample_file(triplets: list[dict], sample_path: Path, sample_size: int = MAX_SAMPLE_ROWS):
    """
    Kirjutab väikese juhuvalimi käsitsi kvaliteedikontrolliks.
    """
    sample_path.parent.mkdir(parents=True, exist_ok=True)

    if not triplets:
        sample = []
    else:
        random.seed(42)
        k = min(sample_size, len(triplets))
        sample = random.sample(triplets, k=k)

    with open(sample_path, "w", encoding="utf-8") as f:
        for item in sample:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Sample rows written: {len(sample)} -> {sample_path}")


def main():
    files, checked_paths = collect_input_files()

    if not files:
        checked_str = "\n - ".join(str(path) for path in checked_paths)
        raise FileNotFoundError(
            "Input data not found. Checked paths:\n"
            f" - {checked_str}\n"
            "Expected either:\n"
            "  1) data/raw/segmented/ (folder with .txt files), or\n"
            "  2) scripts/data/segmented (legacy single file)."
        )

    all_examples = []

    for file_path in files:
        examples = parse_segmented_file(file_path)
        all_examples.extend(examples)

        print(f"{file_path.name}: {len(examples)} examples")

    triplets = build_triplets(all_examples)
    save_jsonl(triplets, OUTPUT_PATH)

    label_1 = sum(1 for item in triplets if item["label"] == 1)
    label_0 = sum(1 for item in triplets if item["label"] == 0)
    total = len(triplets)

    print("\nLarge triplet dataset created.")
    print("Files processed:", len(files))
    print("Total examples:", total)
    print("Label 1:", label_1)
    print("Label 0:", label_0)

    if total > 0:
        print("Label 1 ratio:", round(label_1 / total, 4))
        print("Label 0 ratio:", round(label_0 / total, 4))

    write_sample_file(triplets, SAMPLE_OUTPUT_PATH)

    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
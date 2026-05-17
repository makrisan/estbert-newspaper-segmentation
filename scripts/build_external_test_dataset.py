import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

from src.text_cleaner import extract_p_tags, split_into_sentences

OUTPUT_PATH = PROJECT_ROOT / "data" / "external_test" / "test.jsonl"
SAMPLE_OUTPUT_PATH = PROJECT_ROOT / "data" / "output" / "external_test_sample.jsonl"
INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "external_test" / "test_segmented"
MAX_SAMPLE_ROWS = 25
MIN_CHUNK_LENGTH = 20


def collect_input_files(input_path: Path | None = None) -> tuple[list[Path], list[Path]]:
    """
    Leiab sisendfailid data/raw/segmented/ kaustast.
    Tagastab (failid, kontrollitud asukohad).
    """
    checked: list[Path] = []

    candidates = [
        input_path,
        INPUT_DIR,
        PROJECT_ROOT / "data" / "raw" / "segmented.txt",
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


def split_into_chunks(html_text: str) -> list[str]:
    """
    Teeb ühe artikli HTML tekstist puhastatud tekstijupid.
    Kasutab split_into_sentences() lausepiiride kaitsmiseks,
    et vältida vale lõikamist lühendite ja kuupäevade juures.
    """
    paragraphs = extract_p_tags(html_text)

    # split_into_sentences teeb korraga:
    # - is_layout_noise() filtri
    # - lühendite kaitse
    # - lauseteks jagamise
    # - minimaalse pikkuse kontrolli (>= 20 tähemärki)
    chunks = split_into_sentences(paragraphs)

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
                examples.append({
                    "id": f"{article_id}_{chunk_idx}",
                    "source_file": file_path.stem,
                    "article_id": article_id,
                    "chunk_idx": chunk_idx,
                    "text": chunk,
                    "label": 1 if chunk_idx == 0 else 0
                })

    return examples


def get_article_id(chunk_id: str) -> str:
    return chunk_id.rsplit("_", 1)[0]


def build_triplets(examples: list[dict]) -> list[dict]:
    """
    Ehitab tripletid samas järjestuses nagu tekst inference'i ajal ette tuleb.

    Oluline:
    - Artikli alguse korral ei panda prev="".
    - Artikli alguse korral on prev eelmise artikli viimane chunk.
    - See väldib lihtsat shortcut'i: prev tühi => artikli algus.
    """
    triplets = []

    for i, item in enumerate(examples):
        # Esimene chunk failis on erijuht.
        # Seda ei ole mõistlik treenida, sest inference'is esimene lause
        # ei ole päris boundary eelmise ja praeguse vahel.
        if i == 0:
            continue

        prev_text = examples[i - 1]["text"]

        next_text = ""
        if i < len(examples) - 1:
            next_text = examples[i + 1]["text"]

        triplets.append({
            "id": item["id"],
            "source_file": item["source_file"],
            "article_id": item["article_id"],
            "chunk_idx": item["chunk_idx"],
            "prev": prev_text,
            "curr": item["text"],
            "next": next_text,
            "label": item["label"]
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

    random.seed(42)
    sample = random.sample(triplets, k=min(sample_size, len(triplets))) if triplets else []

    with open(sample_path, "w", encoding="utf-8") as f:
        for item in sample:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Sample rows written: {len(sample)} -> {sample_path}")


def main():
    files, checked_paths = collect_input_files(INPUT_DIR)

    if not files:
        checked_str = "\n - ".join(str(p) for p in checked_paths)
        raise FileNotFoundError(
            f"Sisendfaile ei leitud. Kontrollitud:\n - {checked_str}\n"
            f"Oodatav asukoht: {INPUT_DIR}"
        )

    all_triplets = []

    for file_path in files:
        examples = parse_segmented_file(file_path)
        triplets_for_file = build_triplets(examples)
        all_triplets.extend(triplets_for_file)

        print(
            f"{file_path.name}: "
            f"{len(examples)} examples, "
            f"{len(triplets_for_file)} triplets"
        )

    triplets = all_triplets
    save_jsonl(triplets, OUTPUT_PATH)

    label_1 = sum(1 for t in triplets if t["label"] == 1)
    label_0 = sum(1 for t in triplets if t["label"] == 0)
    total = len(triplets)

    print("\nExternal test andmestik loodud.")
    print(f"Faile töödeldud: {len(files)}")
    print(f"Kokku näiteid:   {total}")

    if total:
        print(f"Label 1:         {label_1} ({label_1 / total:.2%})")
        print(f"Label 0:         {label_0} ({label_0 / total:.2%})")

    # Kontroll: external testis peaks olema mõlemat klassi
    if label_1 == 0:
        raise RuntimeError("External test andmestikus pole ühtegi label 1 näidet.")
    if label_0 == 0:
        raise RuntimeError("External test andmestikus pole ühtegi label 0 näidet.")

    write_sample_file(triplets, SAMPLE_OUTPUT_PATH)
    print(f"Salvestatud: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

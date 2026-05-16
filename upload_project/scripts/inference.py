from pathlib import Path
import sys
import argparse
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_setup import load_model
from src.text_cleaner import extract_p_tags, split_into_sentences

INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "unsegmented"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
MODEL_PATH = PROJECT_ROOT / "models" / "final_model"

MAX_LENGTH = 512
DEFAULT_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    """
    Parsib käsurea argumendid.

    Returns:
        Namespace objekt järgmiste väljadega:
            input_dir:  kaust segmenteerimata .txt failidega
            input_path: üks konkreetne sisendfail (valikuline)
            output_dir: kaust väljundfailide jaoks
            model_path: treenitud mudeli asukoht
            threshold:  boundary tõenäosuse lävi (0.0–1.0)
    """
    parser = argparse.ArgumentParser(
        description="Sliding window inference artikli piiride leidmiseks"
    )
    parser.add_argument(
        "--input-dir", type=Path, default=INPUT_DIR,
        help="Kaust segmenteerimata .txt failidega (vaikimisi: data/raw/unsegmented/)"
    )
    parser.add_argument(
        "--input-path", type=Path, default=None,
        help="Üks konkreetne sisendfail (kui soovid ainult ühte töödelda)"
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help="Boundary tõenäosuse lävi (default: 0.5).",
    )
    return parser.parse_args()


def collect_input_files(args: argparse.Namespace) -> list[Path]:
    """
    Koostab töödeldavate failide nimekirja.

    Kui --input-path on antud, töötleb ainult seda üht faili.
    Muidu leitakse kõik .txt failid --input-dir kaustast.

    Args:
        args: parse_args() tagastatud Namespace objekt

    Returns:
        Sorteeritud nimekiri Path objektidest

    Raises:
        FileNotFoundError: kui üksikfail või kaust ei eksisteeri,
                           või kaustas pole ühtegi .txt faili
    """
    if args.input_path is not None:
        path = (
            args.input_path
            if args.input_path.is_absolute()
            else PROJECT_ROOT / args.input_path
        )
        if not path.exists():
            raise FileNotFoundError(f"Faili ei leitud: {path}")
        return [path]

    input_dir = (
        args.input_dir
        if args.input_dir.is_absolute()
        else PROJECT_ROOT / args.input_dir
    )
    if not input_dir.exists():
        raise FileNotFoundError(f"Kausta ei leitud: {input_dir}")

    files = sorted(input_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"Kaustas pole .txt faile: {input_dir}")

    return files


def predict_boundaries(
    sentences: list[str],
    tokenizer,
    model,
    device,
    threshold: float,
) -> list[dict]:
    """
    Ennustab artikli piire sliding window meetodil.

    Iga lause kohta ehitatakse triplet (prev, curr, next) ja
    saadetakse mudelisse. Mudel tagastab tõenäosuse et curr
    on uue artikli algus (label 1). Esimene lause saab alati
    label 0, kuna eelnevat konteksti pole.

    Args:
        sentences: puhastatud lausete nimekiri
        tokenizer: EstBERT tokenizer
        model:     treenitud BertForSequenceClassification mudel
        device:    torch.device (cpu või cuda)
        threshold: piiri tõenäosuse lävi — kui prob >= threshold,
                   ennustatakse label 1 (artikli algus)

    Returns:
        Nimekiri dict objektidest, igaühes:
            prev:          eelnev lause (tühi kui artikli algus)
            curr:          praegune lause
            next:          järgmine lause (tühi kui viimane)
            pred_label:    0 (jätk) või 1 (uus artikkel)
            boundary_prob: mudeli ennustatud tõenäosus label 1-le
    """
    results = []

    for i, curr in enumerate(sentences):
        prev_text = sentences[i - 1] if i > 0 else ""
        next_text = sentences[i + 1] if i < len(sentences) - 1 else ""

        if i == 0:
            pred_label = 0
            boundary_prob = 0.0
        else:
            # PARANDUS: kasutame tokenize_with_budget() nagu dataset.py
            # Vana lähenemine: tokenizer(full_string) lõikas next täielikult ära
            # Uus lähenemine: igale osale garanteeritud ~170 token eelarve
            from src.dataset import tokenize_with_budget

            encoding = tokenize_with_budget(
                tokenizer,
                prev=prev_text,
                curr=curr,
                next_=next_text,
                max_length=MAX_LENGTH,
            )

            with torch.no_grad():
                outputs = model(
                    input_ids=encoding["input_ids"].unsqueeze(0).to(device),
                    attention_mask=encoding["attention_mask"].unsqueeze(0).to(device),
                )

            boundary_prob = torch.softmax(outputs.logits, dim=1)[0][1].item()
            pred_label = 1 if boundary_prob >= threshold else 0

        results.append({
            "prev": prev_text,
            "curr": curr,
            "next": next_text,
            "pred_label": pred_label,
            "boundary_prob": boundary_prob,
        })

    return results


def write_debug_output(results: list[dict], output_path: Path) -> None:
    """
    Kirjutab inimloetava debug-faili iga lause kohta.

    Iga kirje sisaldab indeksit, ennustatud labelit,
    boundary tõenäosust ning prev/curr/next konteksti.
    Kasulik mudeli käitumise manuaalseks kontrollimiseks.

    Args:
        results:     predict_boundaries() tagastatud nimekiri
        output_path: väljundfaili asukoht (kaust luuakse vajadusel)
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


def write_tagged_output(results: list[dict], output_path: Path) -> None:
    """
    Kirjutab segmenteeritud väljundi <p> tagedega.

    Iga pred_label=1 kohal avatakse uus <p> blokk.
    Tulemus on sama formaat mis segmenteeritud treenimisfailid,
    mis võimaldab tulemusi visuaalselt võrrelda originaaliga.

    Args:
        results:     predict_boundaries() tagastatud nimekiri
        output_path: väljundfaili asukoht (kaust luuakse vajadusel)
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


def process_file(
    input_path: Path,
    args: argparse.Namespace,
    tokenizer,
    model,
    device,
) -> None:
    """
    Töötleb ühe segmenteerimata faili täielikult läbi.

    Loeb faili, puhastab teksti, ennustab piirid ja
    salvestab nii tagged kui debug väljundi. Väljundfailide
    nimed tuletatakse automaatselt sisendi nimest.

    Args:
        input_path: segmenteerimata sisendfaili asukoht
        args:       käsurea argumendid (threshold, output_dir jne)
        tokenizer:  EstBERT tokenizer
        model:      treenitud BertForSequenceClassification mudel
        device:     torch.device (cpu või cuda)
    """
    print(f"\n--- Töötlen: {input_path.name} ---")

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

    print(f"Lauseid pärast eeltöötlust: {len(all_sentences)}")

    results = predict_boundaries(all_sentences, tokenizer, model, device, args.threshold)

    total_boundaries = sum(x["pred_label"] for x in results)
    print(f"Leitud piire: {total_boundaries}")

    stem = input_path.stem
    output_path = args.output_dir / f"{stem}_predicted.txt"
    debug_path = args.output_dir / f"{stem}_debug.txt"

    write_tagged_output(results, output_path)
    write_debug_output(results, debug_path)

    print(f"Väljund: {output_path}")
    print(f"Debug:   {debug_path}")


def main() -> None:
    """
    Peafunktsioon — koordineerib kogu inference pipeline'i.

    Sammud:
        1. Parsib käsurea argumendid
        2. Leiab töödeldavad failid
        3. Laeb mudeli üks kord (efektiivsuse huvides)
        4. Töötleb iga faili järjest läbi process_file()
    """
    args = parse_args()

    input_files = collect_input_files(args)
    print(f"Töödeldavaid faile: {len(input_files)}")

    print("Laen mudeli...")
    model_path = (
        args.model_path
        if args.model_path.is_absolute()
        else PROJECT_ROOT / args.model_path
    )
    tokenizer, model = load_model(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Mudel laetud, kasutan: {device}")
    print(f"Threshold: {args.threshold}")

    for input_path in input_files:
        process_file(input_path, args, tokenizer, model, device)

    print("\nKõik failid töödeldud!")

if __name__ == "__main__":
    main()

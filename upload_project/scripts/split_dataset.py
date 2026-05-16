# standard ratio
# TRAIN_RATIO = 0.8 → õppimine
# VAL_RATIO = 0.1 → kontroll treeningu ajal
# TEST_RATIO = 0.1 → lõplik hindamine

import json
from pathlib import Path
import sys
import argparse

from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_jsonl

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "large_triplet_dataset.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "large"


def save_jsonl(data: list[dict], path: Path) -> None:
    """
    Salvestab andmestiku JSONL formaadis faili.

    Iga dict kirjutatakse eraldi reale JSON kujul.
    Kaust luuakse automaatselt kui see puudub.

    Args:
        data: salvestatavate kirjete nimekiri
        path: väljundfaili asukoht
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def label_stats(dataset: list[dict], name: str) -> None:
    """
    Prindib andmestiku label-jaotuse statistika.

    Näitab label 0 ja label 1 arvu ning protsenti,
    samuti klasside tasakaalustamatuse suhte (0:1).
    Kasulik klasside tasakaalustamatuse hindamiseks
    enne treenimist.

    Args:
        dataset: kirjete nimekiri, igaühel väli "label" (0 või 1)
        name:    andmestiku nimi (nt "Train", "Val", "Test")
    """
    ones = sum(1 for x in dataset if x["label"] == 1)
    zeros = sum(1 for x in dataset if x["label"] == 0)
    total = len(dataset)

    print(f"{name} -> kokku: {total}")

    if total == 0:
        print("Hoiatus: split on tühi")
        return

    print(f"label 0: {zeros} ({zeros / total:.2%})")
    print(f"label 1: {ones} ({ones / total:.2%})")

    if ones > 0:
        print(f"0:1 suhe = {zeros / ones:.2f}:1")


def check_each_split_has_both_labels(
    train_data: list[dict],
    val_data: list[dict],
    test_data: list[dict],
) -> None:
    """
    Kontrollib, et igas splitis oleks olemas nii label 0 kui label 1.
    Kui mõnes splitis puudub üks klass, ei ole F1/precision/recall usaldusväärsed.
    """
    for split_name, split_data in [
        ("Train", train_data),
        ("Val", val_data),
        ("Test", test_data),
    ]:
        ones = sum(1 for x in split_data if x["label"] == 1)
        zeros = sum(1 for x in split_data if x["label"] == 0)

        if ones == 0:
            raise RuntimeError(f"{split_name} splitis pole ühtegi label 1 näidet.")
        if zeros == 0:
            raise RuntimeError(f"{split_name} splitis pole ühtegi label 0 näidet.")

    print("Label check: OK, igas splitis on label 0 ja label 1 olemas")


def parse_args() -> argparse.Namespace:
    """
    Parsib käsurea argumendid andmestiku jagamiseks.

    Returns:
        Namespace objekt järgmiste väljadega:
            input_path:  JSONL sisendfaili asukoht
            output_dir:  kaust train/val/test failide jaoks
            train_ratio: treenimisandmete osakaal (vaikimisi 0.8)
            val_ratio:   valideerimisandmete osakaal (vaikimisi 0.1)
            test_ratio:  testandmete osakaal (vaikimisi 0.1)
            seed:        juhuslikkuse seeme reprodutseeritavuseks
    """
    parser = argparse.ArgumentParser(
        description="Split triplet dataset into train/val/test"
    )
    parser.add_argument(
        "--input-path", type=Path, default=DEFAULT_INPUT_PATH,
        help="Input JSONL dataset path"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for train/val/test JSONL files"
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def extract_group_id(example: dict) -> str:
    """
    Gruppimiseks kasutame source_file väärtust.

    Nii lähevad ühe sisendfaili kõik näited samasse splitti.
    See aitab vältida olukorda, kus sama ajalehefaili näited
    satuvad korraga train, val ja test andmetesse.
    """
    if "source_file" not in example:
        raise KeyError(
            "Näitel puudub 'source_file'. "
            "Kontrolli, et build_large_dataset.py lisab igale näitele source_file välja."
        )

    return str(example["source_file"])


def split_groups(
    group_ids: list[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[set, set, set]:
    """
    Jagab sisendfailid train/val/test gruppidesse source_file tasemel.

    Gruppimine toimub faili järgi, mitte üksikute lausete ega artiklite järgi.
    See tagab, et ühe ajalehefaili kõik näited lähevad samasse splitti.
    Nii väldime andmeleket, kus sama faili sarnane tekst esineb korraga
    treening-, valideerimis- ja testandmetes.

    Jaotus toimub kahes etapis:
        1. train vs (val + test)
        2. val vs test ülejäänud osas
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.0")

    unique_groups = sorted(set(group_ids))

    splitter_1 = GroupShuffleSplit(
        n_splits=1, train_size=train_ratio, random_state=seed
    )
    train_group_idx, temp_group_idx = next(
        splitter_1.split(unique_groups, groups=unique_groups)
    )

    train_groups = {unique_groups[i] for i in train_group_idx}
    temp_groups = [unique_groups[i] for i in temp_group_idx]

    temp_ratio = val_ratio + test_ratio
    val_share_within_temp = val_ratio / temp_ratio if temp_ratio else 0.5

    splitter_2 = GroupShuffleSplit(
        n_splits=1,
        train_size=val_share_within_temp,
        random_state=seed,
    )
    val_group_idx, test_group_idx = next(
        splitter_2.split(temp_groups, groups=temp_groups)
    )

    val_groups = {temp_groups[i] for i in val_group_idx}
    test_groups = {temp_groups[i] for i in test_group_idx}

    return train_groups, val_groups, test_groups


def check_no_group_leakage(
    train_data: list[dict],
    val_data: list[dict],
    test_data: list[dict],
) -> None:
    """
    Kontrollib, et ükski source_file ei esine mitmes splitis korraga.

    Andmeleke tekib siis, kui sama ajalehefaili näited satuvad
    korraga train, val ja test jaotustesse. Sellisel juhul võib mudel
    saada liiga optimistlikud mõõdikud, sest ta näeb treeningus väga
    sarnast teksti sellele, mida hiljem testis hinnatakse.

    Raises:
        RuntimeError: kui leitakse kattuvaid source_file gruppe splittide vahel
    """
    train_groups = {extract_group_id(x) for x in train_data}
    val_groups = {extract_group_id(x) for x in val_data}
    test_groups = {extract_group_id(x) for x in test_data}

    train_val_overlap = train_groups & val_groups
    train_test_overlap = train_groups & test_groups
    val_test_overlap = val_groups & test_groups

    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("Train/Val overlap:", train_val_overlap)
        print("Train/Test overlap:", train_test_overlap)
        print("Val/Test overlap:", val_test_overlap)
        raise RuntimeError("Source file leakage detected across splits")

    print("\nSource file leakage check: OK")
    print(
        f"Files -> train: {len(train_groups)}, "
        f"val: {len(val_groups)}, "
        f"test: {len(test_groups)}"
    )


def main() -> None:
    """
    Peafunktsioon — jagab JSONL andmestiku train/val/test splittideks.

    Sammud:
        1. Laeb täieliku triplet-andmestiku JSONL failist
        2. Eraldab artikli ID-d grupipõhiseks jagamiseks
        3. Jagab artiklid train/val/test gruppidesse
        4. Määrab iga näite vastavasse splitti
        5. Salvestab kolm JSONL faili output kausta
        6. Prindib statistika ja kontrollib andmeleket
    """
    args = parse_args()

    input_path = (
        args.input_path
        if args.input_path.is_absolute()
        else PROJECT_ROOT / args.input_path
    )
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else PROJECT_ROOT / args.output_dir
    )

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    test_path = output_dir / "test.jsonl"

    data = load_jsonl(input_path)

    group_ids = [extract_group_id(x) for x in data]
    unique_group_ids = sorted(set(group_ids))
    print(f"Unikaalseid source_file gruppe: {len(unique_group_ids)}")

    if len(unique_group_ids) < 5:
        print(
            "HOIATUS: source_file gruppe on väga vähe. "
            "Train/val/test split võib olla ebastabiilne."
        )
    train_groups, val_groups, test_groups = split_groups(
        group_ids,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )

    train_data, val_data, test_data = [], [], []
    for item in data:
        group_id = extract_group_id(item)
        if group_id in train_groups:
            train_data.append(item)
        elif group_id in val_groups:
            val_data.append(item)
        else:
            test_data.append(item)

    save_jsonl(train_data, train_path)
    save_jsonl(val_data, val_path)
    save_jsonl(test_data, test_path)

    print("Sisendfail:", input_path)
    print("Väljundkaust:", output_dir)
    print("Kokku näiteid:", len(data))
    print("\nSPLIT STATISTIKA:")
    label_stats(train_data, "Train")
    label_stats(val_data, "Val")
    label_stats(test_data, "Test")

    check_each_split_has_both_labels(train_data, val_data, test_data)
    check_no_group_leakage(train_data, val_data, test_data)

if __name__ == "__main__":
    main()

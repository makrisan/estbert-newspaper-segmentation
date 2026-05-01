import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def run_step(name: str, script_name: str, args: list[str] | None = None):
    print(f"\n==============================")
    print(f"Samm: {name}")
    print(f"==============================\n")

    script_path = SCRIPTS_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"Skripti ei leitud: {script_path}")

    result = subprocess.run(
        [sys.executable, str(script_path)] + (args or []),
        cwd=PROJECT_ROOT,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Samm ebaõnnestus: {name}")

    print(f"\nValmis: {name}")


def main():
    run_step(
        "Ehita triplet-andmestik",
        "build_large_dataset.py",
    )
    run_step(
        "Jaga andmestik train/val/test",
        "split_dataset.py",
        ["--input-path", "data/large_triplet_dataset.jsonl", "--output-dir", "data/large"],
    )
    run_step(
        "Treeni mudel",
        "train.py",
        ["--train-path", "data/large/train.jsonl", "--val-path", "data/large/val.jsonl"],
    )
    run_step(
        "Hinda mudel",
        "evaluate.py",
        ["--val-path", "data/large/val.jsonl", "--test-path", "data/large/test.jsonl"],
    )
    run_step(
        "Inference sliding window meetodil",
        "inference.py",
    )

    print("\nPipeline lõpetatud!")


if __name__ == "__main__":
    main()
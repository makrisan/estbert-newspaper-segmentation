import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def run_step(name: str, script_name: str, args: list[str] | None = None):
    print(f"\n==============================")
    print(f"Running step: {name}")
    print(f"Script: {script_name}")
    print(f"==============================\n")

    script_path = SCRIPTS_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    command = [sys.executable, str(script_path)]
    if args:
        command.extend(args)

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {name}")

    print(f"\nDone: {name}")


def main():
    run_step("Build large triplet dataset", "build_large_dataset.py")
    run_step(
        "Split large dataset",
        "split_dataset.py",
        [
            "--input-path", "data/large_triplet_dataset.jsonl",
            "--output-dir", "data/large",
        ]
    )
    run_step(
        "Train model",
        "train.py",
        [
            "--train-path", "data/large/train.jsonl",
            "--val-path", "data/large/val.jsonl",
        ]
    )
    run_step(
        "Evaluate model",
        "evaluate.py",
        [
            "--val-path", "data/large/val.jsonl",
            "--test-path", "data/large/test.jsonl",
        ]
    )
    run_step("Run sliding window inference", "sliding_window_inference.py")

    print("\nPipeline finished successfully!")


if __name__ == "__main__":
    main()
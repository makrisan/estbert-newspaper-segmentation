import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def run_step(name: str, script_name: str):
    print(f"\n==============================")
    print(f"Running step: {name}")
    print(f"Script: {script_name}")
    print(f"==============================\n")

    script_path = SCRIPTS_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {name}")

    print(f"\nDone: {name}")


def main():
    run_step("Build triplet dataset", "build_triplet_dataset.py")
    run_step("Split dataset", "split_dataset.py")
    run_step("Train model", "train.py")
    run_step("Evaluate model", "evaluate.py")
    run_step("Run sliding window inference", "sliding_window_inference.py")

    print("\nPipeline finished successfully!")


if __name__ == "__main__":
    main()
import argparse
import os
import subprocess

from pathlib import Path

SKETCH_LEARNER_DIR = Path(os.getenv("SKETCH_LEARNER_DIR"))


def run(domain_filepath: Path, problems_directory: Path, workspace: Path, width: int):
    subprocess.call([
        "python3", SKETCH_LEARNER_DIR / "learning" / "main.py",
        "--domain_filepath", str(domain_filepath),
        "--problems_directory", str(problems_directory),
        "--workspace", str(workspace),
        "--width", str(width),
        "--distance_numerical_complexity_limit", "15"]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delivery release experiment.")
    parser.add_argument("--domain_filepath", type=Path, required=True, help="The path to the domain file.")
    parser.add_argument("--problems_directory", type=Path, required=True, help="The directory containing the problem files.")
    parser.add_argument("--workspace", type=Path, required=True, help="The directory containing intermediate files.")
    parser.add_argument("--width", type=int, default=1, help="The upper bound on the sketch width.")

    args = parser.parse_args()

    run(args.domain_filepath, args.problems_directory, args.workspace, args.width)
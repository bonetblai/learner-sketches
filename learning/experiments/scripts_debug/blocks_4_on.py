import argparse
import os
import subprocess

from pathlib import Path

SKETCH_LEARNER_DIR = Path(os.getenv("SKETCH_LEARNER_DIR"))


def run(domain_filepath: Path, problems_directory: Path, workspace: Path, width: int):
    additional_booleans = [
        "b_nullary(arm-empty)",  # 2
        "b_empty(c_and(c_not(c_equal(r_primitive(on,0,1),r_primitive(on_g,0,1))),c_primitive(clear,0)))",
        "b_empty(r_and(r_primitive(on,0,1),r_primitive(on_g,0,1)))"  # goal separating feature
    ]
    additional_numericals = [
        "n_count(c_not(c_primitive(clear,0)))",
        "n_count(r_primitive(on,0,1))",
        "n_count(c_primitive(on-table,0))",
        "n_count(c_primitive(clear,0))",
        "n_count(c_some(r_inverse(r_primitive(on_g,0,1)),c_primitive(holding,0)))"
    ]

    subprocess.call([
        "python3", SKETCH_LEARNER_DIR / "learning" / "main.py",
        "--domain_filepath", str(domain_filepath),
        "--problems_directory", str(problems_directory),
        "--workspace", str(workspace),
        "--width", str(width),
        "--disable_feature_generation"]
        + ["--additional_booleans", ] + additional_booleans
        + ["--additional_numericals",] + additional_numericals
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blocks_4_on debug experiment.")
    parser.add_argument("--domain_filepath", type=Path, required=True, help="The path to the domain file.")
    parser.add_argument("--problems_directory", type=Path, required=True, help="The directory containing the problem files.")
    parser.add_argument("--workspace", type=Path, required=True, help="The directory containing intermediate files.")
    parser.add_argument("--width", type=int, default=1, help="The upper bound on the sketch width.")

    args = parser.parse_args()

    run(args.domain_filepath, args.problems_directory, args.workspace, args.width)
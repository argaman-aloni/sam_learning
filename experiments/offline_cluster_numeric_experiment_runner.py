"""Runs the POL experiments offline on the cluster server."""
import json
import os
import sys
from pathlib import Path

if __name__ == '__main__':
    args = sys.argv
    environment_file_path = Path(args[1]) / "environment.json"
    with open(environment_file_path, "rt") as environment_file:
        environment = json.load(environment_file)
        for env_variable, value in environment.items():
            os.environ[env_variable] = value

    os.system(
        f"nohup bash -c '{sys.executable} numeric_experiment_runner.py "
        f"--working_directory_path {args[2]} --domain_file_name {args[3]} "
        f"--learning_algorithm {args[4]} "
        f"--solver_type {args[5]} --problems_prefix {args[6]} "
        f"--fluents_map_path {args[7]} --fold_number {args[8]} > results-{args[3]}.txt ' &")

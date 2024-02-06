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
        f"nohup bash -c '{sys.executable} conditional_effects_experiment_runner.py "
        f"--working_directory_path {args[2]} --domain_file_name {args[3]} "
        f"--universals_map {args[4]} "
        f"--max_antecedent_size {args[5]} --problems_prefix {args[6]} > results-{args[2]}.txt ' &")

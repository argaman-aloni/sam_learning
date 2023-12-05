"""Runs the POL experiments offline on the cluster server."""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

if __name__ == '__main__':
    now = datetime.now()
    # Format the date and time in a pretty way
    formatted_date_time = now.strftime("%A, %B %d, %Y %I:%M %p")
    print(f"Current date and time: {formatted_date_time}")

    args = sys.argv
    environment_file_path = Path(args[1]) / "environment.json"
    with open(environment_file_path, "rt") as environment_file:
        environment = json.load(environment_file)
        for env_variable, value in environment.items():
            os.environ[env_variable] = value

    os.system(
        f"nohup bash -c '{sys.executable} planning_with_online_learning.py "
        f"--working_directory_path {args[2]} --domain_file_name {args[3]} "
        f"--solver_type {args[4]} --learning_algorithm {args[5]} --fluents_map_path {args[6]}"
        f" --problems_prefix {args[7]} --fold_number {args[8]} > results-{args[3]}.txt ' &")

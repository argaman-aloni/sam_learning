"""Runs the POL experiments offline on the cluster server."""
import os
import sys

if __name__ == '__main__':
    args = sys.argv
    os.system(
        f"nohup bash -c '{sys.executable} numeric_experiment_runner.py "
        f"--working_directory_path {args[1]} --domain_file_name {args[2]} "
        f"--learning_algorithm {args[3]} "
        f"--solver_type {args[4]} --problems_prefix {args[5]} --fluents_map_path {args[6]} > results-{args[2]}.txt ' &")

"""Runs the POL experiments offline on the cluster server."""
import os
import sys

if __name__ == '__main__':
    args = sys.argv
    os.system(
        f"nohup bash -c '{sys.executable} conditional_effects_experiment_runner.py "
        f"--working_directory_path {args[1]} --domain_file_name {args[2]} "
        f"--universals_map {args[3]} "
        f"--max_antecedent_size {args[4]} --problems_prefix {args[5]} > results-{args[2]}.txt ' &")

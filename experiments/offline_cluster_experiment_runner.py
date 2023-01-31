"""Runs the POL experiments offline on the cluster server."""
import os
import sys

if __name__ == '__main__':
    args = sys.argv
    os.system(f"nohup bash -c '{sys.executable} planning_with_offline_learning.py "
              f"--working_directory_path {args[1]} --domain_file_name {args[2]} "
              f"--learning_algorithm {args[3]} "
              f"--solver_type {args[4]} --max_antecedent_size {args[5]} > results-{args[2]}.txt ' &")

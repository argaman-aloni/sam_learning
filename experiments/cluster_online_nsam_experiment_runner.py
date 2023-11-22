"""Runs the POL experiments offline on the cluster server."""
import os
import sys

if __name__ == '__main__':
    args = sys.argv
    os.system(
        f"nohup bash -c '{sys.executable} planning_with_online_learning.py "
        f"--working_directory_path {args[1]} --domain_file_name {args[2]} "
        f"--solver_type {args[3]} --learning_algorithm {args[4]} --fluents_map_path {args[5]}"
        f" --problems_prefix {args[6]} > results-{args[2]}.txt ' &")

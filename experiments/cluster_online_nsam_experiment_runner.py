"""Runs the POL experiments offline on the cluster server."""
import os
import sys

if __name__ == '__main__':
    args = sys.argv
    os.system(
        f"nohup bash -c '{sys.executable} planning_with_online_learning.py "
        f"--working_directory_path {args[1]} --domain_file_name {args[2]} "
        f"--solver_type {args[3]} --problems_prefix {args[4]}  > results-{args[2]}.txt ' &")

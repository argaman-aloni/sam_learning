"""Runs the POL experiments offline on the cluster server."""
import os
import sys

if __name__ == "__main__":
    args = sys.argv
    problems_dir = args[1]
    domain_file_path = args[2]
    problems_prefix = args[3]
    solving_timeouts = args[4]
    os.system(
        f"nohup bash -c '{sys.executable} ../../solvers/enhsp_solver.py {problems_dir} {domain_file_path} {problems_prefix} {solving_timeouts} > /dev/null' &"
    )

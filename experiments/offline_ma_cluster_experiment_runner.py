"""Runs the MA-SAM experiments offline on the cluster server."""
import os
import sys

if __name__ == '__main__':
    args = sys.argv
    os.system(f"nohup bash -c '{sys.executable} ma_planning_with_offline_learning.py "
              f"--working_directory_path {args[1]} --domain_file_name {args[2]} "
              f"--executing_agents {args[3]} > results-{args[2]}.txt ' &")

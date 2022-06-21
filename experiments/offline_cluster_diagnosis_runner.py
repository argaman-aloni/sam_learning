"""Runs the POL experiments offline on the cluster server."""
import os
import sys

if __name__ == '__main__':
    args = sys.argv
    os.system(f"nohup bash -c '{sys.executable} model_fault_diagnosis.py "
              f"--work_dir_path {args[1]} --fluents_map_path {args[2]} --original_domain_file_name {args[3]} > results-{args[2]}.txt' &")

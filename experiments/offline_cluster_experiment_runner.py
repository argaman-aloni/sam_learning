"""Runs the POL experiments offline on the cluster server."""
import os
import sys

if __name__ == '__main__':
    args = sys.argv
    os.system(f"nohup bash -c '{sys.executable} planning_with_offline_learning.py "
              f"{args[1]} {args[2]} {args[3]} > results-{args[2]}.txt' &")

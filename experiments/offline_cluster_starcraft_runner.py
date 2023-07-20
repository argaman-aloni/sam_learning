"""Runs the POL experiments offline on the cluster server."""
import os
import sys

if __name__ == '__main__':
    args = sys.argv
    os.system(
        f"nohup bash -c '{sys.executable} starcraft_experiment_runner.py > /home/mordocha/starcraft/log.txt ' &")

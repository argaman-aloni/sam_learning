import signal
import subprocess
import sys

from experiments.cluster_scripts.common import sigint_handler


def main(min_id, max_id):
    signal.signal(signal.SIGINT, sigint_handler)
    for id in range(min_id, max_id + 1):
        subprocess.check_output(["scancel", str(id)])
        print(f"cancelled job {id}")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))

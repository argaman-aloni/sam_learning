import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import NoReturn

execution_script = """#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition main			### specify partition name where to run a job. short: 7 days limit; gtx1080: 7 days; debug: 2 hours limit and 1 job at a time
#SBATCH --time 0-03:30:00			### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name downward_planner_job			### name of the job
#SBATCH --output job-%J.out			### output log for running job - %J for job number
##SBATCH --mail-user=aaa.bbb@ccc	### user's email for sending job status messages
##SBATCH --mail-type=END			### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE

##SBATCH --mem=32G				### amount of RAM memory
##SBATCH --cpus-per-task=6			### number of CPU cores

module load anaconda
conda info --envs
source activate pol_framework

./sise/home/mordocha/numeric_planning/Metric-FF-v2.1/ff -o {domain_file_path} -f {problem_file_path} > {solution_file_path}
"""

script_execution_lines = """

"""

BATCH_JOB_SUBMISSION_REGEX = re.compile(b"Submitted batch job (?P<batch_id>\d+)")
MAX_RUNNING_TIME = 60  # seconds


class MetricFFSolver:
    logger: logging.Logger

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def write_batch_and_execute_solver(self, script_file_path: Path, problems_directory_path: Path,
                                       domain_file_path: Path) -> NoReturn:
        """

        :param script_file_path:
        :param problems_directory_path:
        :param domain_file_path:
        :return:
        """
        self.logger.info("Changing the current working directory to the MetricFF directory.")
        os.chdir(script_file_path.parent)
        self.logger.info("Starting to solve the input problems using fast downward solver.")
        for problem_file_path in problems_directory_path.glob("pfile*.pddl"):
            solution_path = problems_directory_path / f"{problem_file_path.stem}.solution"
            completed_file_str = execution_script.format(
                domain_file_path=str(domain_file_path.absolute()),
                problem_file_path=str(problem_file_path.absolute()),
                solution_file_path=str(solution_path.absolute())
            )

            with open(script_file_path, "wt") as run_script_file:
                run_script_file.write(completed_file_str)

            submission_str = subprocess.check_output(["sbatch", str(script_file_path)])
            match = BATCH_JOB_SUBMISSION_REGEX.match(submission_str)
            batch_id = match.group("batch_id")
            # waiting for fast downward process to start
            time.sleep(1)
            start_time = time.time()
            execution_state = subprocess.check_output(["squeue", "--me"])
            while batch_id in execution_state:
                self.logger.debug(f"Solver with the id - {batch_id} is still running...")
                if (time.time() - start_time) > MAX_RUNNING_TIME:
                    subprocess.check_output(["scancel", batch_id])

                execution_state = subprocess.check_output(["squeue", "--me"])
                time.sleep(1)
                continue

            self.logger.info("Solver finished its execution!")
            self.logger.debug("Cleaning the sbatch file from the problems directory.")
            os.remove(script_file_path)

        return

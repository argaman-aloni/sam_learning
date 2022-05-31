"""Module responsible for running the Expressive Numeric Heuristic Planner (ENHSP)."""
import logging
import os
import re
import subprocess
import sys
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
#SBATCH --job-name ehsp_planner_job			### name of the job
#SBATCH --output job-%J.out			### output log for running job - %J for job number
##SBATCH --mail-user=aaa.bbb@ccc	### user's email for sending job status messages
##SBATCH --mail-type=END			### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE

##SBATCH --mem=32G				### amount of RAM memory
##SBATCH --cpus-per-task=6			### number of CPU cores

module load anaconda
conda info --envs
source activate pol_framework

java -jar {planner_jar_path} -o {domain_file_path} -f {problem_file_path} -sp {plan_file_path} -anytime -timeout {timeout} > {report_file_path}
"""

ENHSP_FILE_PATH = "/sise/home/mordocha/numeric_planning/ENHSP/enhsp.jar"
BATCH_JOB_SUBMISSION_REGEX = re.compile(b"Submitted batch job (?P<batch_id>\d+)")
MAX_RUNNING_TIME = 60  # seconds


class ENHSPSolver:
    """Class designated to use to activate the metric-FF solver on the cluster and parse its result."""

    logger: logging.Logger

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def write_batch_and_execute_solver(self, script_file_path: Path, problems_directory_path: Path,
                                       domain_file_path: Path, report_file_path: Path) -> NoReturn:
        """Writes the batch file script to run on the cluster and then executes the solver algorithm.

        :param script_file_path: the path to output the learning script.
        :param problems_directory_path: the directory containing the problems that are needed to be solved.
        :param domain_file_path: the path to the domain file that will be used as an input to the planner.
        :param report_file_path: the path to the report file containing the STDOUT.
            Used to understand the outcome of the planner's execution.
        """
        self.logger.info("Starting to solve the input problems using ENHSP solver.")
        for problem_file_path in problems_directory_path.glob("pfile*.pddl"):
            self.logger.debug(f"Starting to work on solving problem - {problem_file_path.stem}")
            solution_path = problems_directory_path / f"{problem_file_path.stem}.solution"
            completed_file_str = execution_script.format(
                planner_jar_path=ENHSP_FILE_PATH,
                domain_file_path=str(domain_file_path.absolute()),
                problem_file_path=str(problem_file_path.absolute()),
                plan_file_path=str(solution_path.absolute()),
                timeout=MAX_RUNNING_TIME,
                report_file_path=report_file_path
            )

            with open(script_file_path, "wt") as run_script_file:
                run_script_file.write(completed_file_str)

            subprocess.check_output(["sbatch", str(script_file_path)])
            self.logger.debug("Waiting for the batch script to start running.")
            time.sleep()
            start_time = time.time()
            while time.time() - start_time < MAX_RUNNING_TIME:
                time.sleep(1)

            self.logger.info("ENHSP finished its execution!")
            self.logger.debug("Cleaning the sbatch file from the problems directory.")
            os.remove(script_file_path)
            for job_file_path in Path(os.getcwd()).glob("job-*.out"):
                self.logger.debug("Removing the temp job file!")
                try:
                    os.remove(job_file_path)

                except FileNotFoundError:
                    continue

        return


if __name__ == '__main__':
    args = sys.argv
    logging.basicConfig(level=logging.DEBUG)
    solver = ENHSPSolver()
    solver.write_batch_and_execute_solver(script_file_path=Path(args[1]),
                                          problems_directory_path=Path(args[2]),
                                          domain_file_path=Path(args[3]),
                                          report_file_path=Path(args[4]))

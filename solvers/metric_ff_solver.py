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
#SBATCH --job-name metric_ff_planner_job			### name of the job
#SBATCH --output job-%J.out			### output log for running job - %J for job number
##SBATCH --mail-user=aaa.bbb@ccc	### user's email for sending job status messages
##SBATCH --mail-type=END			### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE

##SBATCH --mem=32G				### amount of RAM memory
##SBATCH --cpus-per-task=6			### number of CPU cores

module load anaconda
conda info --envs
source activate pol_framework

./ff -o {domain_file_path} -f {problem_file_path} -s 0 > {solution_file_path}
"""

METRIC_FF_DIRECTORY = "/sise/home/mordocha/numeric_planning/Metric-FF-v2.1/"
BATCH_JOB_SUBMISSION_REGEX = re.compile(b"Submitted batch job (?P<batch_id>\d+)")
MAX_RUNNING_TIME = 60  # seconds


class MetricFFSolver:
    logger: logging.Logger

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def write_batch_and_execute_solver(self, script_file_path: Path, problems_directory_path: Path,
                                       domain_file_path: Path) -> NoReturn:
        """Writes the batch file script to run on the cluster and then executes the solver algorithm.

        :param script_file_path: the path to output the learning script.
        :param problems_directory_path: the directory containing the problems that are needed to be solved.
        :param domain_file_path: the path to the domain file that will be used as an input to the planner.
        """
        self.logger.info("Changing the current working directory to the MetricFF directory.")
        os.chdir(METRIC_FF_DIRECTORY)
        self.logger.info("Starting to solve the input problems using MetricFF solver.")
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
                self.logger.debug(f"Solver with the id - {batch_id} is still solving {problem_file_path.stem}...")
                if (time.time() - start_time) > MAX_RUNNING_TIME:
                    subprocess.check_output(["scancel", batch_id])

                execution_state = subprocess.check_output(["squeue", "--me"])
                time.sleep(1)
                continue

            self.logger.info("Solver finished its execution!")
            self.logger.debug("Cleaning the sbatch file from the problems directory.")
            os.remove(script_file_path)
            for job_file_path in Path(METRIC_FF_DIRECTORY).glob("job-*.out"):
                self.logger.debug("Removing the temp job file!")
                try:
                    os.remove(job_file_path)
                except FileNotFoundError:
                    continue

        return


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    solver = MetricFFSolver()
    solver.write_batch_and_execute_solver(
        Path("/sise/home/mordocha/numeric_planning/domains/IPC3/Tests2/Rovers/Numeric/execution_script.sh"),
        Path("/sise/home/mordocha/numeric_planning/domains/IPC3/Tests2/Rovers/Numeric/"),
        Path("/sise/home/mordocha/numeric_planning/domains/IPC3/Tests2/Rovers/Numeric/NumRover.pddl"))

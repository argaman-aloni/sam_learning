import logging
import os
import subprocess
import time
from pathlib import Path
import re

from typing import Tuple

EXECUTION_SCRIPT = """#!/bin/bash

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
scl enable devtoolset-9 bash


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mordocha/.conda/envs/pol_framework2/lib/


./Validate -v -t 0.01 {domain_file_path} {problem_file_path} {solution_file_path} > {validation_log_file_path}

"""

VALIDATOR_DIRECTORY = Path("/sise/home/mordocha/numeric_planning/validators/")

BATCH_JOB_SUBMISSION_REGEX = re.compile(b"Submitted batch job (?P<batch_id>\d+)")

VALID_PLAN = "Plan valid"
INAPPLICABLE_PLAN = "Plan failed to execute"
PLAN_APPLICABLE = "Plan executed successfully"
GOAL_NOT_REACHED = "Goal not satisfied"

MAX_RUNNING_TIME = 60


def write_batch_and_validate_plan(logger: logging.Logger, domain_path: Path, problem_file_path: Path,
                                  solution_file_path: Path) -> Tuple[Path, Path]:
    """Validates that the plan for the input problem.

    :param problem_file_path: the path to the problem file.
    :param solution_file_path: the path to the solution file.
    :return: the path to the script file and the path to the validation log file.
    """
    os.chdir(VALIDATOR_DIRECTORY)
    logger.info("Running VAL to validate the plan's correctness.")
    script_file_path = VALIDATOR_DIRECTORY / "validate_script.sh"
    validation_file_path = domain_path.parent / "validation_log.txt"
    completed_file_str = EXECUTION_SCRIPT.format(
        domain_file_path=str(domain_path),
        problem_file_path=str(problem_file_path.absolute()),
        solution_file_path=str(solution_file_path.absolute()),
        validation_log_file_path=str(validation_file_path.absolute())
    )
    with open(script_file_path, "wt") as run_script_file:
        run_script_file.write(completed_file_str)
    logger.info("Finished writing the script to the cluster. Submitting the job.")
    submission_str = subprocess.check_output(["sbatch", str(script_file_path)])
    match = BATCH_JOB_SUBMISSION_REGEX.match(submission_str)
    batch_id = match.group("batch_id")

    time.sleep(1)
    start_time = time.time()
    execution_state = subprocess.check_output(["squeue", "--me"])
    while batch_id in execution_state:
        logger.debug(f"Validator with the id - {batch_id} is still validating {problem_file_path.stem}...")
        if (time.time() - start_time) > MAX_RUNNING_TIME:
            subprocess.check_output(["scancel", batch_id])

        execution_state = subprocess.check_output(["squeue", "--me"])
        time.sleep(1)
        continue

    logger.info("Finished executing the script.")
    return script_file_path, validation_file_path

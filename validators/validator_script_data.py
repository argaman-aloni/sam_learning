import logging
import os
import subprocess
from pathlib import Path

VALIDATOR_DIRECTORY = Path("/home/mordocha/numeric_planning/VAL/")

VALID_PLAN = "Plan valid"
INAPPLICABLE_PLAN = "Plan failed to execute"
PLAN_APPLICABLE = "Plan executed successfully"
GOAL_NOT_REACHED = "Goal not satisfied"

MAX_RUNNING_TIME = 60

logger = logging.getLogger(__name__)


def write_batch_and_validate_plan(domain_file_path: Path, problem_file_path: Path, solution_file_path: Path) -> Path:
    """Validates that the plan for the input problem.

    :param domain_file_path: the path to the domain file.
    :param problem_file_path: the path to the problem file.
    :param solution_file_path: the path to the solution file.
    :return: the path to the validation log file.
    """
    os.chdir(VALIDATOR_DIRECTORY)
    logger.info("Running VAL to validate the plan's correctness.")
    validation_file_path = domain_file_path.parent / "validation_log.txt"
    try:
        subprocess.check_output(
            f"./Validate -v -t 0.01 {domain_file_path} {problem_file_path} {solution_file_path} > {validation_file_path}")

    except subprocess.CalledProcessError as e:
        logger.error(f"VAL returned status code {e.returncode}.")

    logger.info("Finished validating the solution file.")
    return validation_file_path

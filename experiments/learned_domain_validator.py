"""Module to validate the correctness of the learned action models that were generated."""
import csv
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional, NoReturn, Dict, List, Any

from task import Operator

from sam_learner.core import TrajectoryGenerator
from sam_learner.sam_models import Trajectory
from solvers.fast_downward_solver import FastDownwardSolver

PLAN_SEARCH_TIME_LIMIT = int(os.environ.get('SEARCH_TIME', "300"))
VALID_STATUS = "valid plan!"
INVALID_PLAN_STATUS = "invalid plan!"
FAILED_ACTION_NAME_RE = re.compile(r".* - (\w+-?\w*)")
NOT_POSSIBLE_GROUNDED_ACTION = re.compile(r"KeyError: - '(\w+-?\w*)'")

TIMEOUT_WHILE_SOLVING_ERROR = "Solver could not solve the problem within the given time " \
                              f"limitation ({PLAN_SEARCH_TIME_LIMIT} seconds)."
PROBLEM_NOT_SOLVABLE_ERROR = "Solver could not find a solution to the given problem."
POTENTIAL_PLANNING_ERRORS = [TIMEOUT_WHILE_SOLVING_ERROR, PROBLEM_NOT_SOLVABLE_ERROR]


class ValidationResult:
    """Representation of the plan validation result."""
    status: str
    error: Optional[str]
    failed_action_name: Optional[str]

    def __init__(self, status: str, error: Optional[str] = None, failed_action_name: Optional[str] = None):
        self.status = status
        self.error = error
        self.failed_action_name = failed_action_name


def extract_failed_action_name(error_message) -> str:
    """Extract the failed action name from the error message.

    :param error_message: the message of raised when trying to create a trajectory.
    :return: the name of the failed action.
    """
    match = FAILED_ACTION_NAME_RE.match(error_message)
    return match.group(1)


class DomainValidator:
    """Class that validates the correctness of the domains that were learned by the action model learner.

    The validation process works as follows:
        * Using the learned domain we create plans for the test set problems.
        * For each of the created plans we validate its correctness using the validate URL.
        * We need to validate that the plan that was created is safe, i.e. every action is applicable according to the
            preconditions and effects allowed by the original domain.

    Attributes:
        expected_domain_path: the path to the complete domain containing all of the actions and their preconditions and effects.
    """

    STATISTICS_COLUMNS_NAMES = ["domain_name", "generated_domain_file_name", "problem_file_name", "plan_generated",
                                "plan_validation_status", "validation_error"]

    logger: logging.Logger
    expected_domain_path: str
    fast_downward_solver: FastDownwardSolver

    def __init__(self, expected_domain_path: str):
        self.logger = logging.getLogger(__name__)
        self.expected_domain_path = expected_domain_path
        self.fast_downward_solver = FastDownwardSolver()

    def write_plan_file(self, plan_actions: List[Operator], solution_path: Path) -> NoReturn:
        """Write the plan that was created using the learned domain file.

        :param plan_actions: the actions that were executed in the plan.
        :param solution_path: the path to the file to export the plan to.
        """
        self.logger.info(f"Writing the plan file - {solution_path}")
        with open(solution_path, "wt") as solution_file:
            solution_file.writelines([f"{operator.name}\n" for operator in plan_actions])

    def generate_test_set_plans(
            self, tested_domain_file_path: str, test_set_directory: str) -> Dict[str, Optional[str]]:
        """Generate plans for the given domain file and the given test set directory.

        :param tested_domain_file_path: the path to the tested domain in which we want to create actions for.
        :param test_set_directory: the path to the directory containing the test set problems.
        :return: dictionary containing a mapping between the problem paths and their corresponding solutions.
        """
        successful_plans = {}
        self.logger.info("Solving the test set problems using the learned domain!")
        fast_downward_execution_script_path = Path(test_set_directory) / "solver_execution_script.sh"
        self.fast_downward_solver.write_batch_and_execute_solver(
            output_file_path=fast_downward_execution_script_path,
            test_set_directory_path=Path(test_set_directory),
            problems_regex="*.pddl",
            domain_file_path=tested_domain_file_path
        )
        for problem_file_path in Path(test_set_directory).glob("*.pddl"):
            solution_file_path = problem_file_path.parent / f"{problem_file_path.stem}_plan.solution"
            if solution_file_path.exists():
                successful_plans[str(problem_file_path)] = str(solution_file_path)
            else:
                successful_plans[str(problem_file_path)] = PROBLEM_NOT_SOLVABLE_ERROR

        return successful_plans

    def validate_plan(self, problem_path: str, plan_path: str) -> ValidationResult:
        """Validate the correctness of the learned domain against the domain learned using the learner algorithm.

        :param problem_path: the to the test set problem.
        :param plan_path: the path to the plan generated by a solver.
        :return: an object representing the validation status of the plan.
        """
        self.logger.info(f"Validating the plan generated for the problem - {problem_path}")
        trajectory_generator = TrajectoryGenerator(self.expected_domain_path, problem_path)
        try:
            trajectory_generator.generate_trajectory(plan_path)
            return ValidationResult(VALID_STATUS)

        except AssertionError as error:
            self.logger.warning(f"The plan received is not applicable! {error}")
            # Extracting the failed action name
            error_message = str(error)
            failed_action_name = extract_failed_action_name(error_message)
            return ValidationResult(INVALID_PLAN_STATUS, str(error), failed_action_name)

        except KeyError as error:
            grounded_action = str(error)
            action_name = grounded_action.strip("'()").split(" ")[0]
            self.logger.warning(f"The plan received is not applicable! "
                                f"The operation {grounded_action} is not applicable! The failed action - {action_name}")
            return ValidationResult(INVALID_PLAN_STATUS, grounded_action, action_name)

    def extract_applicable_plan_components(self, problem_path: str, plan_path: str) -> Trajectory:
        """Extract the partial trajectory from the failed plan.

        :param problem_path: the path to the problem file.
        :param plan_path: the path to the failed plan.
        :return: the partial applicable trajectory.
        """
        self.logger.info(f"Extracting the applicable trajectory from the plan file - {plan_path}")
        trajectory_generator = TrajectoryGenerator(self.expected_domain_path, problem_path)
        return trajectory_generator.generate_trajectory(plan_path, should_return_partial_trajectory=True)

    def write_statistics(self, statistics: List[Dict[str, Any]], output_statistics_path: str) -> NoReturn:
        """Writes the statistics of the learned model into a CSV file.

        :param statistics: the object containing the statistics about the learning process.
        :param output_statistics_path: the path to the output file.
        """
        with open(output_statistics_path, 'wt', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.STATISTICS_COLUMNS_NAMES)
            writer.writeheader()
            for data in statistics:
                writer.writerow(data)

    @staticmethod
    def clear_plans(created_plans: List[str], test_set_directory_path: str) -> NoReturn:
        """Clears the directory from the plan files so that the next iterations could continue.

        :param created_plans: the paths to the plans that were created in this iteration.
        :param test_set_directory_path: the path to the test set directory containing the solver execution log.
        """
        for plan_path in created_plans:
            if plan_path is not None:
                os.remove(plan_path)

        for solver_output_path in Path(test_set_directory_path).glob("*.out"):
            os.remove(solver_output_path)

    def on_validation_success(
            self, output_statistics_path: str,
            test_set_statistics: List[Dict[str, Any]]) -> NoReturn:
        """Write the needed statistics if the plans were all validated and were approved.

        :param output_statistics_path: the path to the output file.
        :param test_set_statistics: the statistics object containing the data about the learning process.
        """
        self.logger.info("All plans are valid!")
        self.write_statistics(test_set_statistics, output_statistics_path)
        self.logger.info("Done!")

    def log_model_safety_report(
            self, domain_name: str, generated_domains_directory_path: str, test_set_directory_path: str,
            output_statistics_path: str, is_learning_process_safe: bool = True,
            should_stop_after_first_success: bool = True) -> NoReturn:
        """The main entry point that runs the validation process for the plans that were generated using the learned
            domains.

        :param domain_name: the name of the domain that is being validated.
        :param generated_domains_directory_path: the path to the directory containing the generated domains.
        :param test_set_directory_path: the directory containing the test set problems.
        :param output_statistics_path: the path to the output statistics file.
        :param is_learning_process_safe: an indicator on whether the learning algorithm used creates only safe plans.
            If so, there is no need to validate the created plans of the algorithm.
        :param should_stop_after_first_success: whether or not the algorithm should stop once all of the test set
            problems were solved. This is relevant in cases that the learning is not a monotonic rising function.
        """
        test_set_statistics = []
        for generated_domain_path in Path(generated_domains_directory_path).glob("*.pddl"):
            domain_successful_plans = self.generate_test_set_plans(
                str(generated_domain_path), test_set_directory_path)
            plan_paths = [plan for plan in domain_successful_plans.values() if plan not in POTENTIAL_PLANNING_ERRORS]
            validated_plans = []
            for problem_path in domain_successful_plans:
                plan_path = domain_successful_plans[problem_path]
                domain_statistics = {
                    "domain_name": domain_name,
                    "generated_domain_file_name": generated_domain_path.stem,
                    "problem_file_name": Path(problem_path).stem}

                if domain_successful_plans[problem_path] in POTENTIAL_PLANNING_ERRORS:
                    domain_statistics["plan_generated"] = False
                    domain_statistics["plan_validation_status"] = domain_successful_plans[problem_path]
                    validated_plans.append(False)
                    test_set_statistics.append(domain_statistics)
                    continue

                validation_status = ValidationResult(VALID_STATUS) if is_learning_process_safe \
                    else self.validate_plan(problem_path, plan_path)
                domain_statistics["plan_generated"] = True
                domain_statistics["plan_validation_status"] = validation_status.status
                domain_statistics["validation_error"] = validation_status.error
                validated_plans.append(validation_status.status == VALID_STATUS)
                test_set_statistics.append(domain_statistics)

            self.clear_plans(plan_paths, test_set_directory_path)
            if all(validated_plans) and should_stop_after_first_success:
                self.on_validation_success(output_statistics_path, test_set_statistics)
                return

        self.write_statistics(test_set_statistics, output_statistics_path)
        self.logger.info("Done!")


if __name__ == '__main__':
    try:
        logging.basicConfig(level=logging.DEBUG)
        args = sys.argv
        validator = DomainValidator(expected_domain_path=args[1])
        validator.log_model_safety_report(
            domain_name=args[2],
            generated_domains_directory_path=args[3],
            test_set_directory_path=args[4],
            output_statistics_path=args[5])

    except Exception as e:
        print(e)

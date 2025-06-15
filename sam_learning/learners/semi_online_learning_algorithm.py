"""An online version of the Numeric SAM learner."""

import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List, Tuple, Union, Optional

from pddl_plus_parser.lisp_parsers import ProblemParser, TrajectoryParser
from pddl_plus_parser.models import (
    Domain,
    State,
    ActionCall,
    Observation,
    PDDLObject,
    PDDLFunction,
    Predicate,
    GroundedPredicate,
)

from sam_learning.core import (
    OnlineDiscreteModelLearner,
    PredicatesMatcher,
    NumericFunctionMatcher,
    VocabularyCreator,
    EnvironmentSnapshot,
    EpisodeInfoRecord,
)
from sam_learning.core.online_learning.episode_info_recorder import NUMERIC, DISCRETE, UNKNOWN
from sam_learning.core.online_learning.online_numeric_models_learner import OnlineNumericModelLearner
from sam_learning.core.online_learning.online_utilities import (
    construct_safe_action_model,
    construct_optimistic_action_model,
    export_learned_domain,
    create_plan_actions,
)
from sam_learning.core.online_learning_agents.abstract_agent import AbstractAgent
from solvers import AbstractSolver, SolutionOutputTypes

SAFE_MODEL_TYPE = "safe"
OPTIMISTIC_MODEL_TYPE = "optimistic"

MAX_SUCCESSFUL_STEPS_PER_EPISODE = 50
PROBLEM_SOLVING_TIMEOUT = 60 * 5  # 300 seconds
MIN_EPISODES_TO_PLAN = 50
MIN_EXECUTIONS_PER_ACTION = 10  # Minimum number of executions per action to consider it partially trained

random.seed(42)  # Set seed for reproducibility


class SemiOnlineNumericAMLearner:
    """
    SemiOnlineNumericAMLearner is an unstructured online version of the online numeric action model learning algorithms.

    This class incrementally learns both discrete and numeric action models from online interactions with an environment.
    It supports handling execution failures, updating models based on observed transitions, and refining models through exploration.
    The learner can use external solvers to validate and improve learned models, and records episode statistics for analysis.

    Key features:
    - Maintains separate learners for discrete and numeric action models for each action in the domain.
    - Handles undecided failure observations and attempts to classify the root cause of action failures.
    - Supports exploration to refine models and uses traces to train models.
    - Integrates with external solvers to attempt problem-solving with learned models.
    - Records detailed episode information for evaluation and debugging.
    """

    _discrete_models_learners: Dict[str, OnlineDiscreteModelLearner]
    _numeric_models_learners: Dict[str, OnlineNumericModelLearner]
    agent: AbstractAgent
    episode_recorder: EpisodeInfoRecord
    undecided_failure_observations = Dict[str, Tuple[Set[GroundedPredicate], Dict[str, PDDLFunction]]]

    def __init__(
        self,
        workdir: Path,
        partial_domain: Domain,
        polynomial_degree: int = 0,
        agent: AbstractAgent = None,
        solvers: List[AbstractSolver] = None,
        episode_recorder: EpisodeInfoRecord = None,
    ):
        self.workdir = workdir
        self.logger = logging.getLogger(__name__)
        self.agent = agent
        self.partial_domain = partial_domain
        self._transitions_db = []
        self.triplet_snapshot = EnvironmentSnapshot(partial_domain=partial_domain)
        self._polynomial_degree = polynomial_degree
        self._discrete_models_learners = {}
        self._numeric_models_learners = {}
        self._discrete_predicate_matcher = PredicatesMatcher(partial_domain)
        self._numeric_function_matcher = NumericFunctionMatcher(partial_domain)
        self._vocabulary_creator = VocabularyCreator()
        self._triplet_snapshot = EnvironmentSnapshot(partial_domain=partial_domain)
        self._solvers = solvers
        self.episode_recorder = episode_recorder
        self.undecided_failure_observations = defaultdict(list)
        self._preprocessed_traces_paths = []

    def _handle_execution_failure(
        self,
        action_to_update: Union[ActionCall, str],
        previous_state_pb_functions: Dict[str, PDDLFunction],
        previous_state_pb_predicates: Set[Predicate],
    ):
        """Handles the case when an action execution fails by attempting to classify the root cause
        of the failure. Updates the discrete or numeric model learners based on whether the state
        satisfied the respective model's preconditions. If the cause cannot be determined, the
            observation is added to the undecided failure observations for further analysis.

        :param action_to_update: The action whose execution failed.
        :param previous_state_pb_functions: The numeric functions in the previous state.
        :param previous_state_pb_predicates: The predicates in the previous state.
        """
        action_name = action_to_update.name if isinstance(action_to_update, ActionCall) else action_to_update
        self.logger.debug("The action was not successful, trying to classify the root cause for the failure.")
        if self._discrete_models_learners[action_name].is_state_in_safe_model(state=previous_state_pb_predicates):
            self.logger.debug(
                "The action was not successful, but the discrete model claimed that the state upheld the action's preconditions."
            )
            self._numeric_models_learners[action_name].add_transition_data(previous_state_pb_functions, is_transition_successful=False)
            self.episode_recorder.record_failure_reason(action_name=action_name, failure_reason=NUMERIC)
            return

        if self._numeric_models_learners[action_name].is_state_in_safe_model(state=previous_state_pb_functions):
            self.logger.debug(
                "The action was not successful, but the numeric model claimed that the state upheld the action's preconditions."
            )
            self._discrete_models_learners[action_name].add_transition_data(previous_state_pb_predicates, is_transition_successful=False)
            self.episode_recorder.record_failure_reason(action_name=action_name, failure_reason=DISCRETE)
            return

        self.logger.debug(
            "Cannot decide if the action was not successful due to the discrete or numeric model. Adding to the undecided observations."
        )
        self.undecided_failure_observations[action_name].append((previous_state_pb_predicates, previous_state_pb_functions))
        self.episode_recorder.record_failure_reason(action_name=action_name, failure_reason=UNKNOWN)

    def _eliminate_undecided_observations(self, action_name: str) -> None:
        """Eliminates undecided failure observations for a given action by re-evaluating them
        with the current safe discrete and numeric models. If the cause of failure can now
        be determined, updates the corresponding model learners and removes the observation
        from the undecided list.

        :param action_name: The name of the action whose undecided failure observations should be re-evaluated.
        :returns: None
        """
        self.logger.debug("Eliminating undecided observations.")
        backup_undecided_observations = self.undecided_failure_observations[action_name]
        self.undecided_failure_observations[action_name] = []
        while len(backup_undecided_observations) > 0:
            (previous_state_pb_predicates, previous_state_pb_functions) = backup_undecided_observations.pop()
            self._handle_execution_failure(
                action_to_update=action_name,
                previous_state_pb_functions=previous_state_pb_functions,
                previous_state_pb_predicates=previous_state_pb_predicates,
            )

    def _extract_parameter_bound_state_data(
        self, action: ActionCall, state_predicates: Set[GroundedPredicate], state_functions: Dict[str, PDDLFunction]
    ) -> Tuple[Set[Predicate], Dict[str, PDDLFunction]]:
        """Extracts the parameter bound state data for a given action and state.

        :param action: The action for which the state data is being extracted.
        :param state_predicates: The grounded predicates in the current state.
        :param state_functions: The numeric functions in the current state.
        :return: A tuple containing the propositional predicates and numeric functions relevant to the action.
        """
        pb_predicates = set(self._discrete_predicate_matcher.get_possible_literal_matches(action, list(state_predicates)))
        pb_functions = self._numeric_function_matcher.match_state_functions(action, state_functions)
        return pb_predicates, pb_functions

    def sort_ground_actions_based_on_success_rate(self, grounded_actions: Set[ActionCall]) -> List[ActionCall]:
        """Sorts the grounded actions based on their success rate in the learned models.

        :param grounded_actions: A set of ActionCall instances representing the grounded actions.
        :return: A list of ActionCall instances sorted by their success rate.
        """
        self.logger.debug("Sorting the grounded actions based on their success rate.")
        grouped_actions = defaultdict(list)
        for action in grounded_actions:
            grouped_actions[action.name].append(action)

        for action_list in grouped_actions.values():
            random.shuffle(action_list)

        action_success_rates = {
            action: self.episode_recorder.get_number_successful_action_executions(action_name=action)
            for action in self.partial_domain.actions
        }
        sorted_action_groups = sorted(grouped_actions.items(), key=lambda item: action_success_rates[item[0]])
        # Flatten the sorted, shuffled groups
        final_sorted_actions = [action for _, group in sorted_action_groups for action in group]
        return final_sorted_actions

    def _add_transition_data(self, action_to_update: ActionCall, is_transition_successful: bool = True) -> None:
        """Adds transition data to the relevant models and updates the model learners with
        information about the state changes caused by an action. This method logs the
        process of adding positive transition data to the model for further analysis
        or learning.

        The method involves retrieving the propositional and numeric state representations
        before and after the action and then updating the discrete and numeric models'
        learners accordingly. It also updates the informative states learner with new
        samples from the previous state's predicates and functions.

        :param action_to_update: ActionCall instance representing the action whose
            transition data is to be added.
        :param is_transition_successful: A boolean indicating whether the transition
            caused by the action was successful. Defaults to True.
        """
        previous_state_pb_predicates, previous_state_pb_functions = self._extract_parameter_bound_state_data(
            action=action_to_update,
            state_predicates=self.triplet_snapshot.previous_state_predicates,
            state_functions=self.triplet_snapshot.previous_state_functions,
        )
        next_state_pb_predicates, next_state_pb_functions = self._extract_parameter_bound_state_data(
            action=action_to_update,
            state_predicates=self.triplet_snapshot.next_state_predicates,
            state_functions=self.triplet_snapshot.next_state_functions,
        )

        if not is_transition_successful:
            # we need to know if we can classify why the failure happened
            self._handle_execution_failure(action_to_update, previous_state_pb_functions, previous_state_pb_predicates)
            return

        self.logger.debug("The action was successful, adding the transition data to the model.")
        self._discrete_models_learners[action_to_update.name].add_transition_data(
            previous_state_pb_predicates, next_state_pb_predicates, is_transition_successful=is_transition_successful
        )
        self._numeric_models_learners[action_to_update.name].add_transition_data(
            previous_state_pb_functions, next_state_pb_functions, is_transition_successful=is_transition_successful
        )

        self.logger.debug("Checking if can eliminate some of the undecided observations.")
        self._eliminate_undecided_observations(action_to_update.name)

    def _execute_selected_action(
        self,
        selected_ground_action: ActionCall,
        current_state: State,
        problem_objects: Dict[str, PDDLObject],
        integrate_in_models: bool = True,
    ) -> Tuple[State, bool]:
        """Executes the selected grounded action in the environment using the agent.

        :param selected_ground_action: The ActionCall instance representing the action to execute.
        :param current_state: The current State before executing the action.
        :param problem_objects: A dictionary of problem objects used in the domain.
        :param integrate_in_models: Whether to integrate the observed transition into the learned models (default: True).
        :return: A tuple containing the next State and a boolean indicating if the transition was successful.
        """
        next_state, is_transition_successful, _ = self.agent.observe(state=current_state, action=selected_ground_action)
        self.triplet_snapshot.create_triplet_snapshot(
            previous_state=current_state,
            next_state=next_state,
            current_action=selected_ground_action,
            observation_objects=problem_objects,
        )
        self.episode_recorder.record_single_step(
            action=selected_ground_action, action_applicable=is_transition_successful, previous_state=current_state, next_state=next_state
        )
        if integrate_in_models:
            self.logger.debug("Integrating the transition data into the learned models.")
            self._add_transition_data(selected_ground_action, is_transition_successful=is_transition_successful)

        return next_state, is_transition_successful

    def _select_action_and_execute(
        self, current_state: State, frontier: List[ActionCall], problem_objects: Dict[str, PDDLObject]
    ) -> Tuple[ActionCall, bool, State]:
        """
        Selects an action from the frontier, executes it using the agent, and updates the state.

        :param current_state: The current state before action selection.
        :param frontier: The set of actions available for selection.
        :param problem_objects: A dictionary of problem objects used in the domain.
        :return: A tuple containing the selected ActionCall, a boolean indicating if the transition was successful, and the resulting State.
        """
        self.logger.debug("Selecting an action from the grounded actions set.")
        selected_ground_action = frontier.pop(0)
        self.logger.debug(
            f"Selected the informative action {selected_ground_action.name} with parameters {selected_ground_action.parameters}."
        )
        next_state, is_transition_successful = self._execute_selected_action(
            selected_ground_action, current_state, problem_objects, integrate_in_models=False
        )
        num_unsuccessful_attempts = 0
        while not is_transition_successful and len(frontier) > 0:
            self.logger.debug(
                f"Tried to execute the action {selected_ground_action.name}, but it was not successful. Number of unsuccessful attempts: {num_unsuccessful_attempts + 1}."
            )
            selected_ground_action = frontier.pop(0)
            next_state, is_transition_successful = self._execute_selected_action(
                selected_ground_action, current_state, problem_objects, integrate_in_models=False
            )

        return selected_ground_action, is_transition_successful, next_state

    def initialize_learning_algorithms(self) -> None:
        """Initializes the learning algorithms for each action in the partial domain.

        This method sets up the discrete and numeric model learners, as well as the informative states learner,
        for every action defined in the domain. It creates the necessary vocabularies and associates each learner
        with the corresponding action, preparing the system for online learning.
        """
        self.logger.info("Initializing the learning algorithms.")
        for action_name, action_data in self.partial_domain.actions.items():
            self.logger.debug(f"Initializing the learning algorithms for the action {action_name}.")
            pb_predicates = self._vocabulary_creator.create_lifted_vocabulary(
                domain=self.partial_domain, possible_parameters=action_data.signature
            )
            pb_functions = self._vocabulary_creator.create_lifted_functions_vocabulary(
                domain=self.partial_domain, possible_parameters=action_data.signature
            )
            self._discrete_models_learners[action_name] = OnlineDiscreteModelLearner(action_name=action_name, pb_predicates=pb_predicates)
            self._numeric_models_learners[action_name] = OnlineNumericModelLearner(
                action_name=action_name,
                pb_functions=pb_functions,
                polynom_degree=self._polynomial_degree,
            )

    def _use_solvers_to_solve_problem(
        self, problem_path: Path, domain_path: Path, init_state: State
    ) -> Tuple[SolutionOutputTypes, Optional[int], Optional[State]]:
        """Uses the solvers to solve the problem defined in the given path.

        :param problem_path: The path to the PDDL problem file.
        :param domain_path: The path to the PDDL domain file.
        :param init_state: The initial state from which the problem-solving starts.
        :return: The solution status of the problem-solving attempt.
        """
        solution_path = self.workdir / f"{problem_path.stem}.solution"
        solution_status = SolutionOutputTypes.no_solution
        if self._solvers is None:
            raise ValueError("No solver was provided to the learner.")

        for solver in self._solvers:
            self.logger.info(f"Trying to solve the problem {problem_path.stem} using the solver {solver.name}.")
            solution_status = solver.solve_problem(
                domain_file_path=domain_path,
                problem_file_path=problem_path,
                problems_directory_path=self.workdir,
                solving_timeout=PROBLEM_SOLVING_TIMEOUT,
            )
            if solution_status == SolutionOutputTypes.ok:
                self.logger.info("The problem was solved successfully.")
                plan_actions = create_plan_actions(solution_path)
                trace, goal_reached = self.agent.execute_plan(plan_actions)
                self.train_models_using_trace(trace)
                if not goal_reached:
                    self.logger.debug("The plan created by the solver did not reach the goal.")
                    solution_status = (
                        SolutionOutputTypes.not_applicable if len(trace) < len(plan_actions) else SolutionOutputTypes.goal_not_achieved
                    )
                    return solution_status, len(trace), trace.components[-1].next_state

                return solution_status, len(trace), None

        return solution_status, None, init_state

    def _executed_enough_successful_steps_per_action(self):
        """Checks if enough steps have been executed per action to consider the to be considerably trained."""
        for action in self.partial_domain.actions:
            if self.episode_recorder.get_number_successful_action_executions(action_name=action) < MIN_EXECUTIONS_PER_ACTION:
                self.logger.info(f"Action {action} has NOT been executed enough times.")
                return False

        return True

    def explore_to_refine_models(
        self,
        init_state: State,
        num_steps_till_episode_end: int = MAX_SUCCESSFUL_STEPS_PER_EPISODE,
        problem_objects: Dict[str, PDDLObject] = None,
    ) -> Tuple[bool, int]:
        """
        Explore the environment to refine the action models by executing actions and observing outcomes.

        :param init_state: The initial state from which exploration starts.
        :param num_steps_till_episode_end: The maximum number of steps to perform in the episode.
        :param problem_objects: Optional dictionary of problem objects used in the domain.
        :return: The number of steps taken during exploration or until the goal is reached.
        """
        self.logger.info("Searching for informative actions given the current state.")
        step_number = 0
        grounded_actions = self.agent.get_environment_actions(init_state)
        frontier = self.sort_ground_actions_based_on_success_rate(grounded_actions)
        current_state = init_state.copy()
        while len(frontier) > 0 and step_number < num_steps_till_episode_end:
            self.logger.info(f"Exploring to improve the action model - exploration step {step_number + 1}.")
            action, is_successful, next_state = self._select_action_and_execute(current_state, frontier, problem_objects)
            step_number += 1
            self.logger.info(f"The action {str(action)} was successful. Continuing to the next state.")
            frontier = list(self.sort_ground_actions_based_on_success_rate(grounded_actions))
            current_state = next_state
            if self.agent.goal_reached(current_state):
                self.logger.info("The goal has been reached, returning the learned the number of executed steps.")
                return True, step_number

        self.logger.info("Reached a state with no neighbors to pull an action from, returning the learned model.")
        return False, num_steps_till_episode_end

    def train_models_using_trace(self, trace: Observation) -> None:
        """Train the discrete and numeric action models using a given observation trace.

        :param trace: An Observation object containing a sequence of observed transitions, each with previous and next states,
            the executed action, and success status.
        """
        self.logger.info("Training the models using the trace.")
        for observed_transition in trace.components:
            self.triplet_snapshot.create_triplet_snapshot(
                previous_state=observed_transition.previous_state,
                next_state=observed_transition.next_state,
                current_action=observed_transition.grounded_action_call,
                observation_objects=trace.grounded_objects,
            )
            self._add_transition_data(
                action_to_update=observed_transition.grounded_action_call,
                is_transition_successful=observed_transition.is_successful,
            )

    def _construct_model_and_solve_problem(
        self, model_type: str, problem_path: Path, init_state: State
    ) -> Tuple[SolutionOutputTypes, Optional[int], Optional[State]]:
        """Constructs a model of the specified type and attempts to solve the problem.

        :param model_type: The type of model to construct ('safe' or 'optimistic').
        :param problem_path: The path to the PDDL problem file.
        :param init_state: The initial state from which the problem-solving starts.
        :return: The solution status of the problem-solving attempt.
        """
        if model_type == SAFE_MODEL_TYPE:
            model = construct_safe_action_model(
                partial_domain=self.partial_domain,
                discrete_models_learners=self._discrete_models_learners,
                numeric_models_learners=self._numeric_models_learners,
            )

        elif model_type == OPTIMISTIC_MODEL_TYPE:
            model = construct_optimistic_action_model(
                partial_domain=self.partial_domain,
                discrete_models_learners=self._discrete_models_learners,
                numeric_models_learners=self._numeric_models_learners,
            )

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        domain_path = export_learned_domain(
            workdir=self.workdir, partial_domain=self.partial_domain, learned_domain=model, is_safe_model=(model_type == SAFE_MODEL_TYPE)
        )
        return self._use_solvers_to_solve_problem(domain_path=domain_path, problem_path=problem_path, init_state=init_state)

    def _read_trajectories_and_train_models(self) -> None:
        """"""
        while len(self._preprocessed_traces_paths) > 0:
            trace_path, problem_path = self._preprocessed_traces_paths.pop(0)
            problem = ProblemParser(problem_path=problem_path, domain=self.partial_domain).parse_problem()
            self.logger.info(f"Reading the trajectory from the file {trace_path.stem}.")
            trace = TrajectoryParser(partial_domain=self.partial_domain, problem=problem).parse_trajectory(
                trajectory_file_path=trace_path, contain_transitions_status=True
            )
            self.train_models_using_trace(trace)

    def try_to_solve_problems(self, problems_paths: List[Path]) -> None:
        """Tries to solve the problem using the current domain.

        :param problems_paths: A list of paths to the PDDL problem files to be solved.
        :return: whether the goal was reached and the number of steps taken to solve the problem.
        """
        self.logger.info("Trying to solve the problem.")
        if self._solvers is None:
            raise ValueError("No solver was provided to the learner.")

        for index, problem_path in enumerate(problems_paths):
            safe_model_solution_status, optimistic_model_solution_status = SolutionOutputTypes.no_solution, SolutionOutputTypes.no_solution
            self.episode_recorder.clear_trajectory()
            problem = ProblemParser(problem_path=problem_path, domain=self.partial_domain).parse_problem()
            self.agent.initialize_problem(problem)
            last_state = State(predicates=problem.initial_state_predicates, fluents=problem.initial_state_fluents, is_init=True)
            self.episode_recorder.add_num_grounded_actions(len(self.agent.get_environment_actions(last_state)))
            if index >= MIN_EPISODES_TO_PLAN or self._executed_enough_successful_steps_per_action():
                self._read_trajectories_and_train_models()
                self.logger.info(f"Trying to solve the problem {problem_path.stem} using the safe action model.")
                safe_model_solution_status, trace_len, _ = self._construct_model_and_solve_problem(
                    model_type=SAFE_MODEL_TYPE, problem_path=problem_path, init_state=last_state
                )
                if safe_model_solution_status == SolutionOutputTypes.ok:
                    self.logger.info("The problem was solved using the safe action model.")
                    self.episode_recorder.end_episode(
                        problem_name=problem_path.stem,
                        goal_reached=True,
                        has_solved_solver_problem=True,
                        safe_model_solution_stat=safe_model_solution_status.name,
                    )
                    continue

                (optimistic_model_solution_status, trace_len, last_state) = self._construct_model_and_solve_problem(
                    model_type=OPTIMISTIC_MODEL_TYPE, problem_path=problem_path, init_state=last_state
                )
                if optimistic_model_solution_status == SolutionOutputTypes.ok:
                    self.logger.info("The problem was solved using the optimistic action model.")
                    self.episode_recorder.end_episode(
                        problem_name=problem_path.stem,
                        goal_reached=True,
                        has_solved_solver_problem=True,
                        safe_model_solution_stat=safe_model_solution_status.name,
                        optimistic_model_solution_stat=optimistic_model_solution_status.name,
                    )
                    continue

            self.logger.info(f"Exploring the environment to solve the problem {problem_path.stem}.")
            goal_reached, num_steps_till_episode_end = self.explore_to_refine_models(
                init_state=last_state,
                num_steps_till_episode_end=MAX_SUCCESSFUL_STEPS_PER_EPISODE,
                problem_objects=problem.objects,
            )
            self.episode_recorder.end_episode(
                problem_name=problem_path.stem,
                goal_reached=goal_reached,
                has_solved_solver_problem=False,
                safe_model_solution_stat=safe_model_solution_status.name,
                optimistic_model_solution_stat=optimistic_model_solution_status.name,
            )
            self.logger.info("Training the learning algorithms using the trajectories.")
            self._preprocessed_traces_paths.append((self.episode_recorder.trajectory_path, problem_path))
            self.logger.debug("Exporting the episode statistics to a CSV file.")
            self.episode_recorder.export_statistics(self.workdir / "exploration_statistics.csv")

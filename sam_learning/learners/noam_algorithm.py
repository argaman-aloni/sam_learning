"""An online version of the Numeric SAM learner."""

import logging
import random
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Dict, Set, List, Tuple, Union

from pddl_plus_parser.exporters.numeric_trajectory_exporter import parse_action_call
from pddl_plus_parser.lisp_parsers import ProblemParser
from pddl_plus_parser.models import (
    Domain,
    State,
    ActionCall,
    Precondition,
    Observation,
    PDDLObject,
    PDDLFunction,
    Predicate,
    GroundedPredicate,
)

from sam_learning.core import (
    InformationStatesLearner,
    OnlineDiscreteModelLearner,
    PredicatesMatcher,
    NumericFunctionMatcher,
    VocabularyCreator,
    EnvironmentSnapshot,
    EpisodeInfoRecord,
)
from sam_learning.core.online_learning.online_numeric_models_learner import OnlineNumericModelLearner
from sam_learning.core.online_learning_agents.abstract_agent import AbstractAgent
from solvers import AbstractSolver, SolutionOutputTypes

APPLICABLE_ACTIONS_SELECTION_RATE = 0.2
MAX_STEPS_PER_EPISODE = 1000
PROBLEM_SOLVING_TIMEOUT = 60 * 10  # 1 minutes

random.seed(42)  # Set seed for reproducibility


class ExplorationAlgorithmType(Enum):
    """Enum representing the exploration algorithm types."""

    informative_explorer = "informative_explorer"
    goal_oriented = "goal_oriented"
    combined = "combined"


def create_plan_actions(plan_path: Path) -> List[ActionCall]:
    """Reads a plan file and parses each line into an ActionCall.

    :param plan_path: Path to the plan file containing action calls.
    :return: List of ActionCall objects parsed from the plan file.
    """
    with open(plan_path, "rt") as plan_file:
        plan_lines = plan_file.readlines()

    plan_actions = []
    for line in plan_lines:
        plan_actions.append(parse_action_call(line))

    return plan_actions


class NumericOnlineActionModelLearner:
    """
    Provides functionality for online learning of numeric action models.

    This class is designed to create and manage safe and optimistic action models
    based on transition data extracted during exploration. It also allows for
    construction, optimization, and export of these models. The learner can handle
    state transitions caused by actions, analyze informative states, and update
    learned domain models. Furthermore, it leverages discrete and numeric models
    learners to refine action model accuracy.
    """

    _informative_states_learner: Dict[str, InformationStatesLearner]
    _discrete_models_learners: Dict[str, OnlineDiscreteModelLearner]
    _numeric_models_learners: Dict[str, OnlineNumericModelLearner]
    agent: AbstractAgent
    episode_statistics: Dict[str, int]
    undecided_failure_observations = Dict[str, Tuple[Set[GroundedPredicate], Dict[str, PDDLFunction]]]
    episode_recorder: EpisodeInfoRecord

    def __init__(
        self,
        workdir: Path,
        partial_domain: Domain,
        polynomial_degree: int = 0,
        agent: AbstractAgent = None,
        solvers: List[AbstractSolver] = None,
        exploration_type: ExplorationAlgorithmType = ExplorationAlgorithmType.combined,
        episode_recorder: EpisodeInfoRecord = None,
    ):
        self.workdir = workdir
        self.logger = logging.getLogger(__name__)
        self._informative_states_learner = {}
        self.agent = agent
        self.partial_domain = partial_domain
        self.triplet_snapshot = EnvironmentSnapshot(partial_domain=partial_domain)
        self._polynomial_degree = polynomial_degree
        self._discrete_models_learners = {}
        self._numeric_models_learners = {}
        self._discrete_predicate_matcher = PredicatesMatcher(partial_domain)
        self._numeric_function_matcher = NumericFunctionMatcher(partial_domain)
        self._vocabulary_creator = VocabularyCreator()
        self._triplet_snapshot = EnvironmentSnapshot(partial_domain=partial_domain)
        self._solvers = solvers
        self._exploration_policy = exploration_type
        self.undecided_failure_observations = defaultdict(list)
        self._successful_execution_count = defaultdict(int)
        self._applicable_actions = set()
        self.episode_recorder = episode_recorder

    def construct_safe_action_model(self) -> Domain:
        """Constructs the safe action model for the domain.

        :return: the safe action model.
        """
        self.logger.info("Constructing the safe action model.")
        safe_domain = self.partial_domain.shallow_copy()
        for action_name, action in safe_domain.actions.items():
            preconditions = Precondition("and")
            safe_discrete_preconditions, safe_discrete_effects = self._discrete_models_learners[action_name].get_safe_model()
            for precondition in safe_discrete_preconditions.operands:
                preconditions.add_condition(precondition)

            action.discrete_effects = safe_discrete_effects
            safe_numeric_preconditions, safe_numeric_effects = self._numeric_models_learners[action_name].get_safe_model()
            for precondition in safe_numeric_preconditions.operands:
                preconditions.add_condition(precondition)

            action.numeric_effects = safe_numeric_effects
            action.preconditions.root = preconditions

        return safe_domain

    def construct_optimistic_action_model(self) -> Domain:
        """Constructs the optimistic action model for the domain.

        :return: the safe action model.
        """
        self.logger.info("Constructing the safe action model.")
        optimistic_domain = self.partial_domain.shallow_copy()
        optimistic_domain.requirements.add(":disjunctive-preconditions")
        for action_name, action in optimistic_domain.actions.items():
            preconditions = Precondition("and")
            optimistic_discrete_preconditions, optimistic_discrete_effects = self._discrete_models_learners[
                action_name
            ].get_optimistic_model()
            for precondition in optimistic_discrete_preconditions.operands:
                preconditions.add_condition(precondition)

            action.discrete_effects = optimistic_discrete_effects
            optimistic_numeric_preconditions, optimistic_numeric_effects = self._numeric_models_learners[action_name].get_optimistic_model()
            for precondition in optimistic_numeric_preconditions.operands:
                preconditions.add_condition(precondition)

            action.numeric_effects = optimistic_numeric_effects
            action.preconditions.root = preconditions

        return optimistic_domain

    def _export_learned_domain(self, learned_domain: Domain, is_safe_model: bool = True) -> Path:
        """Exports the learned domain into a file so that it will be used to solve the test set problems.

        :param learned_domain: the domain that was learned by the action model learning algorithm.
        :param is_safe_model: a boolean indicating whether the learned domain is a safe model or an optimistic model.
        :return: The path to the exported domain file.
        """
        domain_file_name = self.partial_domain.name + f"_{'safe' if is_safe_model else 'optimistic'}_learned_domain.pddl"
        domain_path = self.workdir / domain_file_name
        with open(domain_path, "wt") as domain_file:
            domain_file.write(learned_domain.to_pddl())

        return domain_path

    def _calculate_state_action_informative(
        self, current_state: State, action_to_test: ActionCall, problem_objects: Dict[str, PDDLObject]
    ) -> Tuple[bool, bool]:
        """Determines if a given state-action pair is informative based on provided problem objects,
        grounded predicates, and numeric functions. This facilitates evaluating whether the
        state-action pair contributes meaningful information to the learning process.

        :param current_state: The current state of the system, representing the conditions
                              and configurations at the point of evaluation.
        :param action_to_test: The action being evaluated to test its impact on the current state.
        :param problem_objects: A dictionary of problem objects used in the domain, mapping
                                object names to their respective PDDLObject instances.
        :return: A boolean indicating whether the specific state-action pair is informative and whether the action is applicable.
        """
        previous_state_grounded_predicates = self.triplet_snapshot.create_propositional_state_snapshot(
            current_state, action_to_test, {**problem_objects, **self.partial_domain.constants}
        )
        previous_state_grounded_functions = self.triplet_snapshot.create_numeric_state_snapshot(
            current_state, action_to_test, {**problem_objects, **self.partial_domain.constants}
        )
        previous_state_pb_predicates = set(
            self._discrete_predicate_matcher.get_possible_literal_matches(action_to_test, list(previous_state_grounded_predicates))
        )
        previous_state_pb_functions = self._numeric_function_matcher.match_state_functions(
            action_to_test, previous_state_grounded_functions
        )
        state_informative = self._informative_states_learner[action_to_test.name].is_sample_informative(
            new_propositional_sample=previous_state_pb_predicates, new_numeric_sample=previous_state_pb_functions
        )
        action_applicable = self._informative_states_learner[action_to_test.name].is_applicable(
            new_propositional_sample=previous_state_pb_predicates, new_numeric_sample=previous_state_pb_functions
        )
        return state_informative, action_applicable

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
            self._informative_states_learner[action_name].add_new_numeric_failure(new_numeric_sample=previous_state_pb_functions)
            return

        if self._numeric_models_learners[action_name].is_state_in_safe_model(state=previous_state_pb_functions):
            self.logger.debug(
                "The action was not successful, but the numeric model claimed that the state upheld the action's preconditions."
            )
            self._discrete_models_learners[action_name].add_transition_data(previous_state_pb_predicates, is_transition_successful=False)
            return

        self.logger.debug(
            "Cannot decide if the action was not successful due to the discrete or numeric model. Adding to the undecided observations."
        )
        self.undecided_failure_observations[action_name].append((previous_state_pb_predicates, previous_state_pb_functions))

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
        self.episode_recorder.record_single_step(action_name=action_to_update.name, action_applicable=is_transition_successful)
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
        self._informative_states_learner[action_to_update.name].add_new_sample(
            new_propositional_sample=previous_state_pb_predicates,
            new_numeric_sample=previous_state_pb_functions,
            is_successful=is_transition_successful,
        )

        if not is_transition_successful:
            # we need to know if we can classify why the failure happened
            self._handle_execution_failure(action_to_update, previous_state_pb_functions, previous_state_pb_predicates)
            return

        self.logger.debug("The action was successful, adding the transition data to the model.")
        self._successful_execution_count[action_to_update.name] += 1
        self._discrete_models_learners[action_to_update.name].add_transition_data(
            previous_state_pb_predicates, next_state_pb_predicates, is_transition_successful=is_transition_successful
        )
        self._numeric_models_learners[action_to_update.name].add_transition_data(
            previous_state_pb_functions, next_state_pb_functions, is_transition_successful=is_transition_successful
        )

        self.logger.debug("Checking if can eliminate some of the undecided observations.")
        self._eliminate_undecided_observations(action_to_update.name)

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
        random_action = random.choice(frontier)
        if self._exploration_policy == ExplorationAlgorithmType.goal_oriented:
            selected_ground_action = frontier.pop(0)

        else:
            selected_ground_action = frontier.pop(0)
            action_informative, action_applicable = self._calculate_state_action_informative(
                current_state=current_state, action_to_test=selected_ground_action, problem_objects=problem_objects
            )
            while len(frontier) > 0 and not action_informative:
                if action_applicable:
                    if random.random() <= APPLICABLE_ACTIONS_SELECTION_RATE:
                        self.logger.info("Selecting an applicable non-informative action from the grounded actions set.")
                        break

                    self._applicable_actions.add(selected_ground_action)

                selected_ground_action = frontier.pop(0)
                action_informative, action_applicable = self._calculate_state_action_informative(
                    current_state=current_state, action_to_test=selected_ground_action, problem_objects=problem_objects
                )

            if len(frontier) == 0 and not action_informative:
                selected_ground_action = self._applicable_actions.pop() if len(self._applicable_actions) > 0 else random_action

        self.logger.debug(
            f"Selected the informative action {selected_ground_action.name} with parameters {selected_ground_action.parameters}."
        )
        next_state, is_transition_successful, _ = self.agent.observe(state=current_state, action=selected_ground_action)
        self.triplet_snapshot.create_triplet_snapshot(
            previous_state=current_state,
            next_state=next_state,
            current_action=selected_ground_action,
            observation_objects=problem_objects,
        )
        self._add_transition_data(selected_ground_action, is_transition_successful=is_transition_successful)
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
                # qhull_options="QJ Q11 Q12",
            )
            self._informative_states_learner[action_name] = InformationStatesLearner(
                action_name=action_name,
                discrete_model_learner=self._discrete_models_learners[action_name],
                numeric_model_learner=self._numeric_models_learners[action_name],
            )

    def _create_frontier(self, grounded_actions: Set[ActionCall]) -> List[ActionCall]:
        """Creates a frontier of actions to explore.

        :param grounded_actions: The set of grounded actions available for exploration.
        :return: A set of actions to explore.
        """
        if self._exploration_policy == ExplorationAlgorithmType.goal_oriented:
            shuffled_actions = list(grounded_actions)
            random.shuffle(shuffled_actions)
            return shuffled_actions

        weights, actions = [], []
        for action in grounded_actions:
            actions.append(action)
            weights.append(1.0 / (1 + self._successful_execution_count.get(action.name, 0)))

        return random.choices(actions, weights=weights, k=len(actions))

    def _use_solvers_to_solve_problem(self, problem_path: Path, domain_path: Path, solution_path: Path) -> SolutionOutputTypes:
        """Uses the solvers to solve the problem defined in the given path.

        :param problem_path: The path to the PDDL problem file.
        :param domain_path: The path to the PDDL domain file.
        :param solution_path: The path where the solution will be saved.
        :return: The solution status of the problem-solving attempt.
        """
        solution_status = SolutionOutputTypes.no_solution
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
                return solution_status

        return solution_status

    def explore_to_refine_models(
        self,
        init_state: State,
        num_steps_till_episode_end: int = MAX_STEPS_PER_EPISODE,
        problem_objects: Dict[str, PDDLObject] = None,
        debug: bool = False,
    ) -> Tuple[bool, int]:
        """
        Explore the environment to refine the action models by executing actions and observing outcomes.

        :param init_state: The initial state from which exploration starts.
        :param num_steps_till_episode_end: The maximum number of steps to perform in the episode.
        :param problem_objects: Optional dictionary of problem objects used in the domain.
        :param debug: If True, enables to ignore the goal being achieved.
        :return: The number of steps taken during exploration or until the goal is reached.
        """
        self.logger.info("Searching for informative actions given the current state.")
        step_number = 0
        grounded_actions = self.agent.get_environment_actions(init_state)
        frontier = self._create_frontier(grounded_actions)
        current_state = init_state.copy()
        while len(frontier) > 0 and step_number < num_steps_till_episode_end:
            self.logger.info(f"Exploring to improve the action model - exploration step {step_number + 1}.")
            action, is_successful, next_state = self._select_action_and_execute(current_state, frontier, problem_objects)
            step_number += 1
            while not is_successful and step_number < num_steps_till_episode_end and len(frontier) > 0:
                self.logger.debug(f"The action was not successful, trying again for step {step_number + 1}.")
                action, is_successful, next_state = self._select_action_and_execute(current_state, frontier, problem_objects)
                step_number += 1

            if not is_successful:
                self.logger.debug(
                    f"The last explored action was not successful but reached the end of the episode with "
                    f"{step_number} steps / could not execute any more actions..."
                )
                return False, num_steps_till_episode_end

            self.logger.info(f"The action {str(action)} was successful. Continuing to the next state.")
            frontier = self._create_frontier(grounded_actions)
            current_state = next_state
            if self.agent.goal_reached(current_state) and not debug:
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

    def apply_exploration_policy(
        self, problem_path: Path, num_steps_till_episode_end: int = MAX_STEPS_PER_EPISODE
    ) -> Tuple[bool, bool, int]:
        """Applies the exploration policy to the current state.

        :return: the number of steps taken to solve the problem.
        """
        problem = ProblemParser(problem_path=problem_path, domain=self.partial_domain).parse_problem()
        initial_state = State(predicates=problem.initial_state_predicates, fluents=problem.initial_state_fluents, is_init=True)
        if self._exploration_policy == ExplorationAlgorithmType.informative_explorer:
            self.logger.info("Applying the informative explorer exploration policy - starting to explore the environment.")
            goal_reached, executed_steps = self.explore_to_refine_models(
                init_state=initial_state,
                num_steps_till_episode_end=num_steps_till_episode_end,
                problem_objects=problem.objects,
            )
            return goal_reached, False, executed_steps

        self.logger.debug("Applying goal oriented exploration approach.")
        solution_file_path = self.workdir / f"{problem_path.stem}.solution"
        self.logger.info(
            f"Could not solve the problem {problem_path.stem} using the safe action model, transitioning to the optimistic model."
        )
        optimistic_model = self.construct_optimistic_action_model()
        domain_path = self._export_learned_domain(optimistic_model, is_safe_model=False)
        solution_status = self._use_solvers_to_solve_problem(
            domain_path=domain_path,
            problem_path=problem_path,
            solution_path=solution_file_path,
        )

        if solution_status == SolutionOutputTypes.ok:
            self.logger.info("The problem was solved using the optimistic action model.")
            plan_actions = create_plan_actions(solution_file_path)
            trace, goal_reached = self.agent.execute_plan(plan_actions)
            self.train_models_using_trace(trace)
            if goal_reached:
                return goal_reached, True, len(plan_actions)

            else:
                self.logger.info(f"The plan created using the optimistic model could not reach the goal. Valid Plan length: {len(trace)}")
                goal_reached, executed_steps = self.explore_to_refine_models(
                    trace.components[-1].next_state if len(trace) > 0 else initial_state,
                    num_steps_till_episode_end=num_steps_till_episode_end - len(trace),
                    problem_objects=trace.grounded_objects,
                )
                return goal_reached, False, executed_steps + len(trace)

        self.logger.info(f"The problem {problem_path.stem} could not be solved using the optimistic model.")
        goal_reached, executed_steps = self.explore_to_refine_models(
            init_state=initial_state,
            num_steps_till_episode_end=num_steps_till_episode_end,
            problem_objects=problem.objects,
        )
        return goal_reached, False, executed_steps

    def try_to_solve_problem(self, problem_path: Path, num_steps_till_episode_end: int = MAX_STEPS_PER_EPISODE) -> Tuple[bool, int]:
        """Tries to solve the problem using the current domain.

        :param problem_path: the path to the problem to solve.
        :param num_steps_till_episode_end: the number of steps to take until the end of the episode.
        :return: whether the goal was reached and the number of steps taken to solve the problem.
        """
        self.logger.info("Trying to solve the problem.")
        if self._solvers is None:
            raise ValueError("No solver was provided to the learner.")

        solution_file_path = self.workdir / f"{problem_path.stem}.solution"
        self.logger.info("Trying to solve the problem using the safe action model.")
        safe_model = self.construct_safe_action_model()
        domain_path = self._export_learned_domain(safe_model, is_safe_model=True)
        solution_status = self._use_solvers_to_solve_problem(
            domain_path=domain_path,
            problem_path=problem_path,
            solution_path=solution_file_path,
        )
        if solution_status == SolutionOutputTypes.ok:
            self.logger.info("The problem was solved using the safe action model.")
            plan_actions = create_plan_actions(solution_file_path)
            if len(plan_actions) > 0:
                self.logger.info("The plan created using the safe action model.")
                trace, goal_reached = self.agent.execute_plan(plan_actions)
                if not goal_reached:
                    raise ValueError("The goal should have been reached when used the safe action model!")

                self.train_models_using_trace(trace)
                self.episode_recorder.end_episode(
                    undecided_states=self.undecided_failure_observations,
                    goal_reached=goal_reached,
                    num_steps_in_episode=len(trace),
                    has_solved_solver_problem=True,
                )
                return True, len(trace)

        goal_reached, solver_reached_goal, num_steps_till_episode_end = self.apply_exploration_policy(
            problem_path, num_steps_till_episode_end
        )
        self.episode_recorder.end_episode(
            undecided_states=self.undecided_failure_observations,
            goal_reached=goal_reached,
            num_steps_in_episode=num_steps_till_episode_end,
            has_solved_solver_problem=solver_reached_goal,
        )
        return goal_reached, num_steps_till_episode_end

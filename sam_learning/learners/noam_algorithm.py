"""An online version of the Numeric SAM learner."""

import random
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Dict, Set, List, Tuple, Union

from pddl_plus_parser.lisp_parsers import ProblemParser
from pddl_plus_parser.models import (
    Domain,
    State,
    ActionCall,
    Observation,
    PDDLObject,
    PDDLFunction,
    Predicate,
)

from sam_learning.core import (
    InformationStatesLearner,
    EnvironmentSnapshot,
    EpisodeInfoRecord,
)
from sam_learning.core.online_learning_agents.abstract_agent import AbstractAgent
from sam_learning.learners.semi_online_learning_algorithm import SemiOnlineNumericAMLearner, OPTIMISTIC_MODEL_TYPE, SAFE_MODEL_TYPE
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


class NumericOnlineActionModelLearner(SemiOnlineNumericAMLearner):
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
        super().__init__(workdir, partial_domain, polynomial_degree, agent, solvers, episode_recorder)
        self._informative_states_learner = {}
        self.triplet_snapshot = EnvironmentSnapshot(partial_domain=partial_domain)
        self._exploration_policy = exploration_type
        self._successful_execution_count = defaultdict(int)
        self._applicable_actions = set()

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
        super()._handle_execution_failure(action_to_update, previous_state_pb_functions, previous_state_pb_predicates)
        action_name = action_to_update.name if isinstance(action_to_update, ActionCall) else action_to_update
        if not self._discrete_models_learners[action_name].is_state_in_safe_model(state=previous_state_pb_predicates):
            return

        self.logger.debug("The action was not successful and it was due to the numeric part of the action!")
        self._informative_states_learner[action_name].add_new_numeric_failure(new_numeric_sample=previous_state_pb_functions)

    def _add_transition_data(
        self, action_to_update: ActionCall, is_transition_successful: bool = True, previous_state: State = None, next_state: State = None
    ) -> None:
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
        super()._add_transition_data(action_to_update, is_transition_successful, previous_state, next_state)
        previous_state_pb_predicates, previous_state_pb_functions = self._extract_parameter_bound_state_data(
            action=action_to_update,
            state_predicates=self.triplet_snapshot.previous_state_predicates,
            state_functions=self.triplet_snapshot.previous_state_functions,
        )
        self._informative_states_learner[action_to_update.name].add_new_sample(
            new_propositional_sample=previous_state_pb_predicates,
            new_numeric_sample=previous_state_pb_functions,
            is_successful=is_transition_successful,
        )

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
            next_state, is_transition_successful = self._execute_selected_action(selected_ground_action, current_state, problem_objects)
            return selected_ground_action, is_transition_successful, next_state

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
        next_state, is_transition_successful = self._execute_selected_action(selected_ground_action, current_state, problem_objects)
        return selected_ground_action, is_transition_successful, next_state

    def initialize_learning_algorithms(self) -> None:
        """Initializes the learning algorithms for each action in the partial domain.

        This method sets up the discrete and numeric model learners, as well as the informative states learner,
        for every action defined in the domain. It creates the necessary vocabularies and associates each learner
        with the corresponding action, preparing the system for online learning.
        """
        super().initialize_learning_algorithms()
        for action_name, action_data in self.partial_domain.actions.items():
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
            weights.append(1.0 / (1 + self.episode_recorder.get_number_successful_action_executions(action_name=action.name)))

        return random.choices(actions, weights=weights, k=len(actions))

    def explore_to_refine_models(
        self,
        init_state: State,
        num_steps_till_episode_end: int = MAX_STEPS_PER_EPISODE,
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
        self.logger.info(
            f"Could not solve the problem {problem_path.stem} using the safe action model, transitioning to the optimistic model."
        )
        solution_status, trace_length, last_state = self._construct_model_and_solve_problem(OPTIMISTIC_MODEL_TYPE, problem_path)
        if solution_status == SolutionOutputTypes.ok:
            self.logger.info("The problem was solved using the optimistic action model.")
            return True, True, trace_length

        elif solution_status in [SolutionOutputTypes.not_applicable, SolutionOutputTypes.goal_not_achieved]:
            self.logger.info("The optimistic action model could not solve the problem, exploring to refine models.")
            goal_reached, executed_steps = self.explore_to_refine_models(
                init_state=last_state,
                num_steps_till_episode_end=num_steps_till_episode_end - trace_length,
                problem_objects=problem.objects,
            )
            return goal_reached, False, executed_steps

        self.logger.info(f"The optimistic action model could not solve the problem, exploring to refine models. Status: {solution_status}")
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

        self.logger.info("Trying to solve the problem using the safe action model.")
        solution_status, trace_length, _ = self._construct_model_and_solve_problem(SAFE_MODEL_TYPE, problem_path)
        if solution_status == SolutionOutputTypes.ok:
            self.episode_recorder.end_episode(
                undecided_states=self.undecided_failure_observations,
                goal_reached=True,
                num_steps_in_episode=trace_length,
                has_solved_solver_problem=True,
            )
            return True, trace_length

        elif solution_status in [SolutionOutputTypes.not_applicable, SolutionOutputTypes.goal_not_achieved]:
            raise ValueError("The goal should have been reached when used the safe action model!")

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

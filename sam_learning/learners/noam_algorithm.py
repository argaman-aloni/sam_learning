"""An online version of the Numeric SAM learner."""

import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List, Tuple, Union

from pddl_plus_parser.lisp_parsers import ProblemParser
from pddl_plus_parser.models import (
    Domain,
    State,
    ActionCall,
    PDDLObject,
    PDDLFunction,
    Predicate,
)

from sam_learning.core import (
    InformationStatesLearner,
    EnvironmentSnapshot,
    EpisodeInfoRecord,
    contains_duplicates,
)
from sam_learning.core.online_learning_agents.abstract_agent import AbstractAgent
from sam_learning.learners.semi_online_learning_algorithm import (
    SemiOnlineNumericAMLearner,
    OPTIMISTIC_MODEL_TYPE,
    SAFE_MODEL_TYPE,
    MAX_SUCCESSFUL_STEPS_PER_EPISODE,
    MAX_FAILED_STEPS_PER_EPISODE,
)
from solvers import AbstractSolver, SolutionOutputTypes
from utilities import LearningAlgorithmType

NUM_STEPS_FOR_STARVATION = 1000  # Number of steps after which an action is considered starved
random.seed(42)  # Set seed for reproducibility


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
        exploration_type: LearningAlgorithmType = LearningAlgorithmType.noam_learning,
        episode_recorder: EpisodeInfoRecord = None,
    ):
        super().__init__(workdir, partial_domain, polynomial_degree, agent, solvers, episode_recorder, exploration_type=exploration_type)
        self._informative_states_learner = {}
        self.triplet_snapshot = EnvironmentSnapshot(partial_domain=partial_domain)
        self._successful_execution_count = defaultdict(int)
        self._action_last_executed_step = defaultdict(int)
        self._global_step_counter = 0

    @staticmethod
    def _create_random_actions_frontier(grounded_actions: Set[ActionCall]) -> List[ActionCall]:
        """

        :param grounded_actions:
        :return:
        """
        injective_actions = [action for action in grounded_actions if not contains_duplicates(action.parameters)]
        shuffled_actions = list(injective_actions)
        random.shuffle(shuffled_actions)
        return shuffled_actions

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
        super()._add_transition_data(action_to_update, is_transition_successful)
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
        # Build a mapping from action name to list of actions in the frontier
        frontier_by_name = {}
        for action in frontier:
            frontier_by_name.setdefault(action.name, []).append(action)

        # Find action names in the frontier that have not been executed in the last 100 steps
        starved_action_names = [
            name
            for name in frontier_by_name
            if self._action_last_executed_step.get(name, -float("inf")) <= self._global_step_counter - NUM_STEPS_FOR_STARVATION
        ]
        if starved_action_names:
            # Pick the first starved action name and pop an action with that name from the frontier
            starved_name = starved_action_names[0]
            selected_ground_action = random.choice(frontier_by_name[starved_name])
            frontier.remove(selected_ground_action)
            next_state, is_transition_successful = self._execute_selected_action(selected_ground_action, current_state, problem_objects)
            return selected_ground_action, is_transition_successful, next_state

        selected_ground_action = frontier.pop(0)
        action_informative, action_applicable = self._calculate_state_action_informative(
            current_state=current_state, action_to_test=selected_ground_action, problem_objects=problem_objects
        )
        while len(frontier) > 0 and not action_informative:
            selected_ground_action = frontier.pop(0)
            action_informative, action_applicable = self._calculate_state_action_informative(
                current_state=current_state, action_to_test=selected_ground_action, problem_objects=problem_objects
            )

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
        return self.sort_ground_actions_based_on_success_rate(grounded_actions)

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
        frontier = self._create_frontier(grounded_actions)
        current_state = init_state.copy()
        self.episode_step_failure_counter = 0
        while (
            len(frontier) > 0
            and step_number < num_steps_till_episode_end
            and self.episode_step_failure_counter < MAX_FAILED_STEPS_PER_EPISODE
        ):
            self.logger.info(f"Exploring to improve the action model - exploration step {step_number + 1}.")
            action, is_successful, next_state = self._select_action_and_execute(current_state, frontier, problem_objects)
            self._action_last_executed_step[action.name] = step_number
            self._global_step_counter += 1
            while not is_successful and len(frontier) > 0:
                self.logger.debug(f"The action was not successful, trying again for step {step_number + 1}.")
                action, is_successful, next_state = self._select_action_and_execute(current_state, frontier, problem_objects)
                self._action_last_executed_step[action.name] = step_number
                self._global_step_counter += 1

            if not is_successful:
                self.logger.debug(f"Informative search failed, trying to execute a random action to move forward.")
                frontier = self._create_frontier(grounded_actions)
                action, is_successful, next_state = super()._select_action_and_execute(current_state, frontier, problem_objects)
                self._action_last_executed_step[action.name] = step_number
                self._global_step_counter += 1

                if not is_successful:
                    self.logger.info("Reached a dead end - ending the episode...")
                    return False, num_steps_till_episode_end

            step_number += 1
            self.logger.info(f"The action {str(action)} was successful. Continuing to the next state.")
            frontier = self._create_frontier(grounded_actions)
            current_state = next_state
            if self.agent.goal_reached(current_state):
                self.logger.info("The goal has been reached, returning the learned the number of executed steps.")
                return True, step_number

        self.logger.info("Reached a state with no neighbors to pull an action from, returning the learned model.")
        return False, num_steps_till_episode_end

    def _explore_and_terminate_episode(
        self,
        initial_state: State,
        problem_path: Path,
        num_steps_till_episode_end: int,
        problem_objects: Dict[str, PDDLObject],
        safe_model_solution_stat: SolutionOutputTypes = SolutionOutputTypes.no_solution,
        optimistic_model_solution_stat: SolutionOutputTypes = None,
    ) -> Tuple[bool, bool, int]:
        """

        :param initial_state:
        :param problem_path:
        :param num_steps_till_episode_end:
        :param problem_objects:
        :param safe_model_solution_stat:
        :param optimistic_model_solution_stat:
        :return:
        """
        exploration_start_time = time.time()
        goal_reached, executed_steps = self.explore_to_refine_models(
            init_state=initial_state,
            num_steps_till_episode_end=num_steps_till_episode_end,
            problem_objects=problem_objects,
        )
        for action_name in self.partial_domain.actions:
            self.logger.debug(f"Checking if can eliminate some of the undecided observations for the action {action_name}.")
            self._eliminate_undecided_observations(action_name)

        self.episode_recorder.end_episode(
            problem_name=problem_path.stem,
            goal_reached=goal_reached,
            has_solved_solver_problem=False,
            safe_model_solution_stat=safe_model_solution_stat.name,
            optimistic_model_solution_stat=None if not optimistic_model_solution_stat else optimistic_model_solution_stat.name,
            exploration_time=time.time() - exploration_start_time,
            export_trajectory=False,
        )
        return goal_reached, False, executed_steps

    def apply_exploration_policy(
        self,
        problem_path: Path,
        num_steps_till_episode_end: int = MAX_SUCCESSFUL_STEPS_PER_EPISODE,
        safe_model_solution_stat: SolutionOutputTypes = SolutionOutputTypes.no_solution,
    ) -> Tuple[bool, bool, int]:
        """Applies the exploration policy to the current state.

        :return: the number of steps taken to solve the problem.
        """
        problem = ProblemParser(problem_path=problem_path, domain=self.partial_domain).parse_problem()
        initial_state = State(predicates=problem.initial_state_predicates, fluents=problem.initial_state_fluents, is_init=True)
        self.logger.info(
            f"Could not solve the problem {problem_path.stem} using the safe action model, transitioning to the optimistic model."
        )
        solution_status, trace_length, last_state = self._construct_model_and_solve_problem(
            OPTIMISTIC_MODEL_TYPE, problem_path, init_state=initial_state
        )

        if solution_status == SolutionOutputTypes.ok:
            self.logger.info("The problem was solved using the optimistic action model.")
            self.episode_recorder.end_episode(
                problem_name=problem_path.stem,
                goal_reached=True,
                has_solved_solver_problem=True,
                safe_model_solution_stat=safe_model_solution_stat.name,
                optimistic_model_solution_stat=solution_status.name,
                export_trajectory=False,
            )
            return True, True, trace_length

        self.logger.info("The optimistic action model could not solve the problem, exploring to refine models.")
        return self._explore_and_terminate_episode(
            initial_state=last_state,
            problem_path=problem_path,
            num_steps_till_episode_end=num_steps_till_episode_end,
            problem_objects=problem.objects,
            safe_model_solution_stat=safe_model_solution_stat,
            optimistic_model_solution_stat=solution_status,
        )

    def try_to_solve_problem(
        self, problem_path: Path, num_steps_till_episode_end: int = MAX_SUCCESSFUL_STEPS_PER_EPISODE
    ) -> Tuple[bool, int]:
        """Tries to solve the problem using the current domain.

        :param problem_path: the path to the problem to solve.
        :param num_steps_till_episode_end: the number of steps to take until the end of the episode.
        :return: whether the goal was reached and the number of steps taken to solve the problem.
        """
        self.logger.info("Trying to solve the problem.")
        if self._solvers is None:
            raise ValueError("No solver was provided to the learner.")

        self.logger.info("Trying to solve the problem using the safe action model.")
        problem = ProblemParser(problem_path=problem_path, domain=self.partial_domain).parse_problem()
        initial_state = State(predicates=problem.initial_state_predicates, fluents=problem.initial_state_fluents, is_init=True)
        solution_status, trace_length, _ = self._construct_model_and_solve_problem(SAFE_MODEL_TYPE, problem_path, init_state=initial_state)
        if solution_status == SolutionOutputTypes.ok:
            self.episode_recorder.end_episode(
                problem_name=problem_path.stem,
                goal_reached=True,
                has_solved_solver_problem=True,
                safe_model_solution_stat=solution_status.name,
                export_trajectory=False,
            )
            return True, trace_length

        if solution_status == SolutionOutputTypes.not_applicable:
            raise ValueError("The goal should have been reached when used the safe action model!")

        goal_reached, solver_reached_goal, num_steps_till_episode_end = self.apply_exploration_policy(
            problem_path, num_steps_till_episode_end, solution_status
        )
        return goal_reached, num_steps_till_episode_end


class GoalOrientedExplorer(NumericOnlineActionModelLearner):
    """
    A specialized learner that focuses on goal-oriented exploration in the domain.

    This class extends the NumericOnlineActionModelLearner to implement a goal-oriented
    exploration strategy, allowing for targeted learning and refinement of action models
    based on specific goals within the domain.
    """

    def __init__(
        self,
        workdir: Path,
        partial_domain: Domain,
        polynomial_degree: int = 0,
        agent: AbstractAgent = None,
        solvers: List[AbstractSolver] = None,
        episode_recorder: EpisodeInfoRecord = None,
        exploration_type: LearningAlgorithmType = LearningAlgorithmType.goal_oriented_explorer,
    ):
        super().__init__(
            workdir=workdir,
            partial_domain=partial_domain,
            polynomial_degree=polynomial_degree,
            agent=agent,
            solvers=solvers,
            exploration_type=exploration_type,
            episode_recorder=episode_recorder,
        )

    def _create_frontier(self, grounded_actions: Set[ActionCall]) -> List[ActionCall]:
        """Creates a frontier of actions to explore.

        :param grounded_actions: The set of grounded actions available for exploration.
        :return: A set of actions to explore.
        """
        return self._create_random_actions_frontier(grounded_actions)

    def _select_action_and_execute(
        self, current_state: State, frontier: List[ActionCall], problem_objects: Dict[str, PDDLObject]
    ) -> Tuple[ActionCall, bool, State]:
        selected_ground_action = frontier.pop(0)
        next_state, is_transition_successful = self._execute_selected_action(selected_ground_action, current_state, problem_objects)
        return selected_ground_action, is_transition_successful, next_state


class InformativeExplorer(GoalOrientedExplorer):
    """
    A specialized learner that focuses on informative exploration in the domain.

    This class extends the NumericOnlineActionModelLearner to implement an informative
    exploration strategy, allowing for targeted learning and refinement of action models
    based on informative states within the domain.
    """

    def __init__(
        self,
        workdir: Path,
        partial_domain: Domain,
        polynomial_degree: int = 0,
        agent: AbstractAgent = None,
        solvers: List[AbstractSolver] = None,
        episode_recorder: EpisodeInfoRecord = None,
        exploration_type: LearningAlgorithmType = LearningAlgorithmType.informative_explorer,
    ):
        super().__init__(
            workdir=workdir,
            partial_domain=partial_domain,
            polynomial_degree=polynomial_degree,
            agent=agent,
            solvers=solvers,
            exploration_type=exploration_type,
            episode_recorder=episode_recorder,
        )

    def apply_exploration_policy(
        self,
        problem_path: Path,
        num_steps_till_episode_end: int = MAX_SUCCESSFUL_STEPS_PER_EPISODE,
        safe_model_solution_stat: SolutionOutputTypes = None,
    ) -> Tuple[bool, bool, int]:
        problem = ProblemParser(problem_path=problem_path, domain=self.partial_domain).parse_problem()
        initial_state = State(predicates=problem.initial_state_predicates, fluents=problem.initial_state_fluents, is_init=True)
        self.logger.info("Applying the informative explorer exploration policy - starting to explore the environment.")
        return self._explore_and_terminate_episode(
            initial_state=initial_state,
            problem_path=problem_path,
            num_steps_till_episode_end=num_steps_till_episode_end,
            problem_objects=problem.objects,
            safe_model_solution_stat=safe_model_solution_stat,
            optimistic_model_solution_stat=SolutionOutputTypes.irrelevant,
        )


class OptimisticExplorer(GoalOrientedExplorer):
    """
    A specialized learner that focuses on optimistic exploration in the domain.

    This class extends the NumericOnlineActionModelLearner to implement an optimistic
    exploration strategy, allowing for targeted learning and refinement of action models
    based on optimistic assumptions within the domain.
    """

    def __init__(
        self,
        workdir: Path,
        partial_domain: Domain,
        polynomial_degree: int = 0,
        agent: AbstractAgent = None,
        solvers: List[AbstractSolver] = None,
        episode_recorder: EpisodeInfoRecord = None,
        exploration_type: LearningAlgorithmType = LearningAlgorithmType.optimistic_explorer,
    ):
        super().__init__(
            workdir=workdir,
            partial_domain=partial_domain,
            polynomial_degree=polynomial_degree,
            agent=agent,
            solvers=solvers,
            exploration_type=exploration_type,
            episode_recorder=episode_recorder,
        )

    def try_to_solve_problem(
        self, problem_path: Path, num_steps_till_episode_end: int = MAX_SUCCESSFUL_STEPS_PER_EPISODE
    ) -> Tuple[bool, int]:
        """Tries to solve the problem using the current domain.

        :param problem_path: the path to the problem to solve.
        :param num_steps_till_episode_end: the number of steps to take until the end of the episode.
        :return: whether the goal was reached and the number of steps taken to solve the problem.
        """
        self.logger.info("Trying to solve the problem.")
        if self._solvers is None:
            raise ValueError("No solver was provided to the learner.")

        self.logger.info("Trying to solve the problem using the optimistic action model.")
        problem = ProblemParser(problem_path=problem_path, domain=self.partial_domain).parse_problem()
        initial_state = State(predicates=problem.initial_state_predicates, fluents=problem.initial_state_fluents, is_init=True)
        solution_status, trace_length, _ = self._construct_model_and_solve_problem(
            OPTIMISTIC_MODEL_TYPE, problem_path, init_state=initial_state
        )
        if solution_status == SolutionOutputTypes.ok:
            self.episode_recorder.end_episode(
                problem_name=problem_path.stem,
                goal_reached=True,
                has_solved_solver_problem=True,
                optimistic_model_solution_stat=solution_status.name,
                export_trajectory=False,
            )
            return True, trace_length

        goal_reached, solver_reached_goal, num_steps_till_episode_end = self._explore_and_terminate_episode(
            initial_state=initial_state,
            problem_path=problem_path,
            num_steps_till_episode_end=num_steps_till_episode_end,
            problem_objects=problem.objects,
            optimistic_model_solution_stat=solution_status,
        )
        return goal_reached, num_steps_till_episode_end

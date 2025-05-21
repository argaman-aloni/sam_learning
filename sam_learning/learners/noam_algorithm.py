"""An online version of the Numeric SAM learner."""
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, Set, List, Any, Tuple

from pddl_plus_parser.exporters.numeric_trajectory_exporter import parse_action_call
from pddl_plus_parser.lisp_parsers import ProblemParser
from pddl_plus_parser.models import (
    Domain,
    State,
    ActionCall,
    Precondition,
    NumericalExpressionTree,
    Observation,
    PDDLObject,
)

from sam_learning.core import (
    InformationStatesLearner,
    PriorityQueue,
    EpisodeInfoRecord,
    OnlineDiscreteModelLearner,
    PredicatesMatcher,
    NumericFunctionMatcher,
    VocabularyCreator,
    EnvironmentSnapshot,
)
from sam_learning.core.online_learning.online_numeric_models_learner import OnlineNumericModelLearner
from sam_learning.core.online_learning_agents.abstract_agent import AbstractAgent
from solvers import AbstractSolver, SolutionOutputTypes

NON_INFORMATIVE_IG = 0
MAX_STEPS_PER_EPISODE = 100
PROBLEM_SOLVING_TIMEOUT = 60 * 5  # 5 minutes


class ExplorationAlgorithmType(Enum):
    """Enum representing the exploration algorithm types."""

    informative_explorer = "informative_explorer"
    goal_oriented = "goal_oriented"
    combined = "combined"


class NumericOnlineActionModelLearner:
    """"An online version of the Numeric SAM learner."""

    DUMMY_ACTION_CALL = ActionCall(name="dummy_action", grounded_parameters=[])

    _informative_states_learner: Dict[str, InformationStatesLearner]
    agent: AbstractAgent
    episode_statistics: Dict[str, int]
    _episode_recorder: EpisodeInfoRecord
    safe_perfect_numeric_effects = Dict[str, Set[NumericalExpressionTree]]

    def __init__(
        self,
        workdir: Path,
        partial_domain: Domain,
        polynomial_degree: int = 0,
        agent: AbstractAgent = None,
        solver: AbstractSolver = None,
        exploration_type: ExplorationAlgorithmType = ExplorationAlgorithmType.combined,
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
        self._solver = solver
        self._exploration_policy = exploration_type

    def _construct_safe_action_model(self) -> Domain:
        """Constructs the safe action model for the domain.

        :return: the safe action model.
        """
        self.logger.info("Constructing the safe action model.")
        for action_name, action in self.partial_domain.actions.items():
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

        return self.partial_domain

    def _construct_optimistic_action_model(self) -> Domain:
        """Constructs the optimistic action model for the domain.

        :return: the safe action model.
        """
        self.logger.info("Constructing the safe action model.")
        for action_name, action in self.partial_domain.actions.items():
            preconditions = Precondition("and")
            optimistic_discrete_preconditions, optimistic_discrete_effects = self._discrete_models_learners[action_name].get_optimistic_model()
            for precondition in optimistic_discrete_preconditions.operands:
                preconditions.add_condition(precondition)

            action.discrete_effects = optimistic_discrete_effects
            optimistic_numeric_preconditions, optimistic_numeric_effects = self._numeric_models_learners[action_name].get_optimistic_model()
            for precondition in optimistic_numeric_preconditions.operands:
                preconditions.add_condition(precondition)

            action.numeric_effects = optimistic_numeric_effects
            action.preconditions.root = preconditions

        return self.partial_domain

    def _export_learned_domain(self, learned_domain: Domain, is_safe_model: bool = True) -> Path:
        """Exports the learned domain into a file so that it will be used to solve the test set problems.

        :param learned_domain: the domain that was learned by the action model learning algorithm.
        :param test_set_path: the path to the test set directory where the domain would be exported to.
        :param file_name: the name of the file to export the domain to.
        """
        domain_file_name = self.partial_domain.name + f"{'safe' if is_safe_model else 'optimistic'}_learned_domain.pddl"
        domain_path = self.workdir / domain_file_name
        with open(domain_path, "wt") as domain_file:
            domain_file.write(learned_domain.to_pddl())

        return domain_path

    def _create_plan_actions(self, plan_path: Path) -> List[ActionCall]:
        """

        :param plan_path:
        :return:
        """
        self.logger.debug(f"Reading the plan in the path {plan_path}")
        with open(plan_path, "rt") as plan_file:
            plan_lines = plan_file.readlines()

        plan_actions = []
        for line in plan_lines:
            plan_actions.append(parse_action_call(line))

        return plan_actions

    def _calculate_state_action_informative_state(
        self, current_state: State, action_to_test: ActionCall, problem_objects: Dict[str, PDDLObject]
    ) -> bool:
        """

        :param current_state:
        :param action_to_test:
        :param problem_objects:
        :return:
        """
        previous_state_grounded_predicates = self.triplet_snapshot.create_propositional_state_snapshot(current_state, action_to_test, problem_objects)
        previous_state_grounded_functions = self.triplet_snapshot.create_numeric_state_snapshot(current_state, action_to_test, problem_objects)
        previous_state_pb_predicates = set(
            self._discrete_predicate_matcher.get_possible_literal_matches(action_to_test, list(previous_state_grounded_predicates))
        )
        previous_state_pb_functions = self._numeric_function_matcher.match_state_functions(action_to_test, previous_state_grounded_functions)
        state_informative = self._informative_states_learner[action_to_test.name].is_sample_informative(
            new_propositional_sample=previous_state_pb_predicates, new_numeric_sample=previous_state_pb_functions
        )
        return state_informative

    def _add_transition_data(self, action_to_update: ActionCall, is_transition_successful: bool = True) -> None:
        """
        Adds transition data to the relevant models and updates the model learners with
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
        self.logger.debug("Adding the positive transition data to the model.")
        previous_state_pb_predicates = set(
            self._discrete_predicate_matcher.get_possible_literal_matches(action_to_update, list(self.triplet_snapshot.previous_state_predicates))
        )
        next_state_pb_predicates = set(
            self._discrete_predicate_matcher.get_possible_literal_matches(action_to_update, list(self.triplet_snapshot.next_state_predicates))
        )
        previous_state_pb_functions = self._numeric_function_matcher.match_state_functions(
            action_to_update, self.triplet_snapshot.previous_state_functions
        )
        next_state_pb_functions = self._numeric_function_matcher.match_state_functions(action_to_update, self.triplet_snapshot.next_state_functions)
        self._discrete_models_learners[action_to_update.name].add_transition_data(
            previous_state_pb_predicates, next_state_pb_predicates, is_transition_successful=is_transition_successful
        )
        self._numeric_models_learners[action_to_update.name].add_transition_data(
            previous_state_pb_functions, next_state_pb_functions, is_transition_successful=is_transition_successful
        )
        self._informative_states_learner[action_to_update.name].add_new_sample(
            new_propositional_sample=previous_state_pb_predicates,
            new_numeric_sample=previous_state_pb_functions,
            is_successful=is_transition_successful,
        )

    def _select_action_and_execute(
        self, current_state: State, frontier: PriorityQueue, problem_objects: Dict[str, PDDLObject]
    ) -> Tuple[ActionCall, bool, State]:
        """

        :param current_state:
        :param frontier:
        :param problem_objects:
        :return:
        """
        action = frontier.get_item()
        next_state, is_transition_successful, _ = self.agent.observe(state=current_state, action=action)
        self.triplet_snapshot.create_triplet_snapshot(
            previous_state=current_state, next_state=next_state, current_action=action, observation_objects=problem_objects
        )
        self._add_transition_data(action, is_transition_successful=is_transition_successful)
        return action, is_transition_successful, next_state

    def initialize_learning_algorithms(self) -> None:
        """

        :return:
        """
        self.logger.info("Initializing the learning algorithms.")
        for action_name, action_data in self.partial_domain.actions.items():
            self.logger.debug(f"Initializing the learning algorithms for the action {action_name}.")
            pb_predicates = self._vocabulary_creator.create_lifted_vocabulary(domain=self.partial_domain, possible_parameters=action_data.signature)
            pb_functions = self._vocabulary_creator.create_lifted_functions_vocabulary(
                domain=self.partial_domain, possible_parameters=action_data.signature
            )
            self._discrete_models_learners[action_name] = OnlineDiscreteModelLearner(action_name=action_name, pb_predicates=pb_predicates)
            self._numeric_models_learners[action_name] = OnlineNumericModelLearner(
                action_name=action_name, pb_functions=pb_functions, polynom_degree=self._polynomial_degree
            )
            self._informative_states_learner[action_name] = InformationStatesLearner(
                action_name=action_name,
                discrete_model_learner=self._discrete_models_learners[action_name],
                numeric_model_learner=self._numeric_models_learners[action_name],
            )

    def calculate_valid_neighbors(
        self, grounded_actions: Set[ActionCall], current_state: State, problem_objects: Dict[str, PDDLObject]
    ) -> PriorityQueue:
        """Calculates the valid action neighbors for the current state that the learner is in.

        :param grounded_actions: all possible grounded actions.
        :param current_state: the current state that the learner is in.
        :return: a priority queue of the valid neighbors for the current state, the priority of the action is based
            on their IG.
        """
        self.logger.info("Calculating the valid neighbors for the current state.")
        valid_frontier = PriorityQueue()
        if self._exploration_policy == ExplorationAlgorithmType.goal_oriented:
            self.logger.debug("The exploration policy is goal oriented, adding all the actions to the priority queue.")
            for action in grounded_actions:
                valid_frontier.insert(item=action, priority=1.0, selection_probability=1.0)

            return valid_frontier

        for action in grounded_actions:
            state_informative = self._calculate_state_action_informative_state(current_state, action, problem_objects)
            if state_informative:
                self.logger.debug(f"The action {action.name} is informative, adding it to the priority queue.")
                valid_frontier.insert(item=action, priority=1.0, selection_probability=1.0)

        return valid_frontier

    def update_failed_action_neighbors(
        self, neighbors: PriorityQueue, current_state: State, action: ActionCall, problem_objects: Dict[str, PDDLObject]
    ) -> PriorityQueue:
        """Calculates the new neighbor queue based on the new information of the failed action.

        :param neighbors: the previously calculated neighbors queue.
        :param current_state: the state in which the action had failed.
        :param action: the failed grounded action.
        :param problem_objects: the objects in the observation.
        :return: the new neighbors queue with the failed lifted action updated.
        """
        self.logger.debug("Updating the failed action's frontier with the new data.")
        new_neighbors = PriorityQueue()
        while len(neighbors) > 0:
            neighbor, information_gain, probability = neighbors.get_queue_item_data()
            if neighbor.name != action.name:
                new_neighbors.insert(item=neighbor, priority=information_gain, selection_probability=probability)
                continue

            state_informative = self._calculate_state_action_informative_state(current_state, neighbor, problem_objects)
            if state_informative:
                self.logger.info(f"The action {neighbor.name} is informative, adding it to the priority queue.")
                new_neighbors.insert(item=neighbor, priority=0.5, selection_probability=probability)

        return new_neighbors

    def search_to_learn_action_model(
        self, init_state: State, num_steps_till_episode_end: int = MAX_STEPS_PER_EPISODE, problem_objects: Dict[str, PDDLObject] = None
    ) -> int:
        """Searches for informative actions to learn an action model that solves the problem.

        :param init_state: the current state of the environment.
        :return: the learned domain with the number of steps done in the episode and whether the goal was achieved.
        """
        self.logger.info("Searching for informative actions given the current state.")
        grounded_actions = self.agent.get_environment_actions(init_state)
        self._episode_recorder.add_num_grounded_actions(len(grounded_actions))
        step_number = 0
        frontier = self.calculate_valid_neighbors(grounded_actions, init_state, problem_objects)
        current_state = init_state.copy()
        while len(frontier) > 0 and step_number < num_steps_till_episode_end:
            action, is_transition_successful, next_state = self._select_action_and_execute(current_state, frontier, problem_objects)
            step_number += 1

            while not is_transition_successful and step_number < MAX_STEPS_PER_EPISODE:
                self.logger.debug("The action was not successful, trying again.")
                frontier = self.update_failed_action_neighbors(frontier, current_state, action)
                action, is_transition_successful, next_state = self._select_action_and_execute(current_state, frontier, problem_objects)
                step_number += 1

            grounded_actions = self.agent.get_environment_actions(next_state)
            current_state = next_state
            if self.agent.goal_reached(current_state):
                self.logger.info("The goal has been reached, returning the learned model.")
                return step_number

            frontier = self.calculate_valid_neighbors(grounded_actions, current_state, problem_objects)

        self.logger.info("Reached a state with no neighbors to pull an action from, returning the learned model.")
        return num_steps_till_episode_end

    def train_models_using_trace(self, trace: Observation) -> None:
        """

        :param trace:
        :return:
        """
        self.logger.info("Training the models using the trace.")
        for observed_transition in trace.components:
            self.triplet_snapshot.create_triplet_snapshot(
                previous_state=observed_transition.previous_state,
                next_state=observed_transition.next_state,
                current_action=observed_transition.grounded_action_call,
                observation_objects=trace.grounded_objects,
            )
            self._add_transition_data(action_to_update=observed_transition.grounded_action_call, is_transition_successful=True)

    def apply_exploration_policy(self, problem_path: Path) -> int:
        """Applies the exploration policy to the current state.

        :return: the number of steps taken to solve the problem.
        """
        problem = ProblemParser(problem_path=problem_path, domain=self.partial_domain).parse_problem()
        initial_state = State(predicates=problem.initial_state_predicates, fluents=problem.initial_state_fluents, is_init=True)
        if self._exploration_policy == ExplorationAlgorithmType.informative_explorer:
            self.logger.info("Applying the informative explorer exploration policy - starting to explore the environment.")
            return self.search_to_learn_action_model(init_state=initial_state, num_steps_till_episode_end=MAX_STEPS_PER_EPISODE)

        self.logger.debug("Applying goal oriented exploration approach.")
        solution_file_path = self.workdir / f"{problem_path.stem}.solution"
        self.logger.info(f"Could not solve the problem {problem_path.stem} using the safe action model, transitioning to the optimistic model.")
        optimistic_model = self._construct_optimistic_action_model()
        domain_path = self._export_learned_domain(optimistic_model, is_safe_model=False)
        solution_status = self._solver.solve_problem(
            domain_file_path=domain_path,
            problem_file_path=problem_path,
            problems_directory_path=self.workdir,
            solving_timeout=PROBLEM_SOLVING_TIMEOUT,
        )

        trace = Observation()
        trace.add_problem_objects(problem.objects)
        trace.add_component(previous_state=initial_state, call=self.DUMMY_ACTION_CALL, next_state=initial_state)

        if solution_status == SolutionOutputTypes.ok:
            self.logger.info("The problem was solved using the safe action model.")
            plan_actions = self._create_plan_actions(solution_file_path)
            trace, goal_reached = self.agent.execute_plan(plan_actions)
            self.train_models_using_trace(trace)
            if goal_reached:
                return len(plan_actions)

        self.logger.info(f"The plan created using the optimistic model could not reach the goal.")
        return (
            len(trace)
            + self.search_to_learn_action_model(
                trace.components[-1].next_state,
                num_steps_till_episode_end=MAX_STEPS_PER_EPISODE - len(trace),
                problem_objects=trace.grounded_objects,
            )
            - 1
        )

    def try_to_solve_problem(self, problem_path: Path) -> int:
        """Tries to solve the problem using the current domain.

        :param problem_path: the path to the problem to solve.
        :return: the number of steps taken to solve the problem.
        """
        self.logger.info("Trying to solve the problem.")
        if self._solver is None:
            raise ValueError("No solver was provided to the learner.")

        solution_file_path = self.workdir / f"{problem_path.stem}.solution"
        self.logger.info("Trying to solve the problem using the safe action model.")
        safe_model = self._construct_safe_action_model()
        domain_path = self._export_learned_domain(safe_model, is_safe_model=True)
        solution_status = self._solver.solve_problem(
            domain_file_path=domain_path,
            problem_file_path=problem_path,
            problems_directory_path=self.workdir,
            solving_timeout=PROBLEM_SOLVING_TIMEOUT,
        )
        if solution_status == SolutionOutputTypes.ok:
            self.logger.info("The problem was solved using the safe action model.")
            plan_actions = self._create_plan_actions(solution_file_path)
            return len(plan_actions)

        return self.apply_exploration_policy(problem_path)

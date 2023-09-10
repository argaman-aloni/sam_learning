"""An online version of the Numeric SAM learner."""
import random
from typing import Dict, Set, Optional, Tuple

from pddl_plus_parser.models import Domain, State, ActionCall, PDDLObject, Precondition, Predicate

from sam_learning.core import InformationGainLearner, NotSafeActionError, LearnerDomain, AbstractAgent, \
    PriorityQueue
from sam_learning.learners.numeric_sam import PolynomialSAMLearning

LABEL = "label"

NON_INFORMATIVE_IG = 0
MIN_FEATURES_TO_CONSIDER = 1
MAX_STEPS_PER_EPISODE = 300


class OnlineNSAMLearner(PolynomialSAMLearning):
    """"An online version of the Numeric SAM learner."""

    ig_learner: Dict[str, InformationGainLearner]
    agent: AbstractAgent
    _action_observation_rate: Dict[str, float]
    _state_failure_rate: int
    _state_applicable_actions: PriorityQueue

    def __init__(self, partial_domain: Domain, polynomial_degree: int = 0, agent: AbstractAgent = None):
        super().__init__(partial_domain=partial_domain, polynomial_degree=polynomial_degree)
        self.ig_learner = {}
        self.agent = agent
        self._action_observation_rate = {action: 1 for action in self.partial_domain.actions}
        self._state_failure_rate = 0
        self._state_applicable_actions = PriorityQueue()

    def _extract_objects_from_state(self, state: State) -> Dict[str, PDDLObject]:
        """Extracts the objects from the state.

        :param state: the state to extract the objects from.
        :return: a dictionary mapping object names to their matching PDDL object .
        """
        self.logger.debug("Extracting the objects from the state.")
        state_objects = {}
        for grounded_predicates_set in state.state_predicates.values():
            for grounded_predicate in grounded_predicates_set:
                for param_name, obj_type in grounded_predicate.signature.items():
                    object_name = grounded_predicate.object_mapping[param_name]
                    state_objects[object_name] = PDDLObject(name=object_name, type=obj_type)

        for grounded_function in state.state_fluents.values():
            for obj_name, obj_type in grounded_function.signature.items():
                state_objects[obj_name] = PDDLObject(name=obj_name, type=obj_type)

        self.logger.debug(f"Extracted the following objects - {list(state_objects.keys())}")
        return state_objects

    def _is_successful_action(self, previous_state: State, next_state: State) -> bool:
        """Checks whether the action was successful.

        :param previous_state: the previous state.
        :param next_state: the next state.
        :return: whether the action was successful.
        """
        self.logger.debug("Checking whether or not the action was successful.")
        return self.are_states_different(previous_state, next_state)

    def _select_next_action_to_execute(self, frontier_actions: PriorityQueue) -> ActionCall:
        """Selects the next action to execute from both the informative frontier and the applicable actions.

        Note:
            To be able to traverse over different states we raise the probability of the applicable actions to be with
            inversed ratio to the failure rate.

        :param frontier_actions: the actions that are informative and their execution helps the learner.
        :return: the next action to execute.
        """
        select_from_applicable_prob = 1 - 1 / (1 + self._state_failure_rate)
        select_from_frontier_prob = 1 - select_from_applicable_prob
        queue_to_select_from = random.choices(
            [self._state_applicable_actions, frontier_actions],
            weights=(select_from_applicable_prob, select_from_frontier_prob), k=1)[0]
        if len(queue_to_select_from) == 0:  # assuming that the frontier is not empty
            self.logger.debug("There are no actions to select from, returning a random action from the frontier.")
            return frontier_actions.get_item()

        return queue_to_select_from.get_item()

    def reset_current_epoch_numeric_data(self) -> None:
        """Resets the numeric part of the domain's data."""
        self.logger.debug("Resetting the numeric part of the domain's data.")
        for action in self.partial_domain.actions.values():
            discrete_preconditions = {op for op in action.preconditions.root.operands if isinstance(op, Predicate)}
            action.preconditions.root = Precondition("and")
            for discrete_precondition in discrete_preconditions:
                action.preconditions.add_condition(discrete_precondition)

            action.numeric_effects = set()

    def init_online_learning(self) -> None:
        """Initializes the online learning algorithm."""
        for action_name, action_data in self.partial_domain.actions.items():
            action_predicate_lifted_vocabulary = self.vocabulary_creator.create_lifted_vocabulary(
                self.partial_domain, action_data.signature)
            lifted_functions = self.vocabulary_creator.create_lifted_functions_vocabulary(
                self.partial_domain, action_data.signature)
            lifted_predicate_names = [p.untyped_representation for p in action_predicate_lifted_vocabulary]
            lifted_function_names = [func for func in lifted_functions]
            self.ig_learner[action_name] = InformationGainLearner(
                action_name=action_name, lifted_functions=lifted_function_names,
                lifted_predicates=lifted_predicate_names)

    def calculate_state_action_information_gain(
            self, state: State, action: ActionCall, action_already_calculated: bool = False) -> float:
        """Calculates the information gain of a state.

        :param state: the state to calculate the information gain of.
        :param action: the action that we calculate the information gain of executing in the state.
        :param action_already_calculated: whether the action's information gain had been calculated already,
            to reduce calculation efforts.
        :return: the information gain of the state.
        """
        self.logger.info(f"Calculating the information gain of applying {str(action)} on the state.")
        state_objects = self._extract_objects_from_state(state)
        grounded_state_propositions = self.triplet_snapshot.create_propositional_state_snapshot(
            state, action, state_objects)
        lifted_predicates = self.matcher.get_possible_literal_matches(action, list(grounded_state_propositions))
        grounded_state_functions = self.triplet_snapshot.create_numeric_state_snapshot(state, action, state_objects)
        lifted_functions = self.function_matcher.match_state_functions(action, grounded_state_functions)
        if self._action_observation_rate[action.name] == 1:
            self.logger.debug(f"Action {action.name} has yet to be observed. Updating the relevant lifted functions.")
            self.ig_learner[action.name].remove_non_existing_functions(list(lifted_functions.keys()))

        is_informative = self.ig_learner[action.name].is_sample_informative(
            lifted_functions, lifted_predicates, use_cache=action_already_calculated)
        IG = NON_INFORMATIVE_IG if not is_informative else self.ig_learner[action.name].calculate_information_gain(
            lifted_functions, lifted_predicates)
        if not is_informative:
            self.logger.debug(f"The action {action.name} is not informative, checking if it is an applicable one.")
            if self.ig_learner[action.name].is_applicable_and_new_state(lifted_functions, lifted_predicates):
                selection_prob = (1 - self._action_observation_rate[action.name] /
                                  sum([rate for rate in self._action_observation_rate.values()]))
                self._state_applicable_actions.insert(item=action, priority=1, selection_probability=selection_prob)

        return IG

    def create_all_grounded_actions(self, observed_objects: Dict[str, PDDLObject]) -> Set[ActionCall]:
        """Creates all the grounded actions for the domain given the current possible objects.

        :param observed_objects: the objects that the learner has observed so far.
        :return: a set of all the possible grounded actions.
        """
        self.logger.info("Creating all the grounded actions for the domain given the current possible objects.")
        grounded_action_calls = self.vocabulary_creator.create_grounded_actions_vocabulary(
            domain=self.partial_domain, observed_objects=observed_objects)
        return grounded_action_calls

    def update_agent(self, new_agent: AbstractAgent) -> None:
        """Updates the agent that the learner is using."""
        self.logger.info(f"Updating the agent.")
        self.agent = new_agent

    def calculate_valid_neighbors(
            self, grounded_actions: Set[ActionCall], current_state: State) -> PriorityQueue:
        """Calculates the valid action neighbors for the current state that the learner is in.

        :param grounded_actions: all possible grounded actions.
        :param current_state: the current state that the learner is in.
        :return: a priority queue of the valid neighbors for the current state, the priority of the action is based
            on their IG.
        """
        self.logger.info("Calculating the valid neighbors for the current state.")
        neighbors = PriorityQueue()
        action_calculation_cache = {action_name: 0 for action_name in self.partial_domain.actions}
        for grounded_action in grounded_actions:
            self.logger.debug(f"Checking the action {grounded_action.name}.")
            # Setting to a negative value since priority queue is works from smallest to largest.
            action_info_gain = self.calculate_state_action_information_gain(
                state=current_state, action=grounded_action,
                action_already_calculated=action_calculation_cache[grounded_action.name] > 0)
            action_calculation_cache[grounded_action.name] += 1
            selection_prob = (1 - self._action_observation_rate[grounded_action.name] /
                              sum([observation_rate for observation_rate in self._action_observation_rate.values()]))

            if abs(action_info_gain) > NON_INFORMATIVE_IG:  # IG is a negative number.
                self.logger.info(f"The action {grounded_action.name} is informative, adding it to the priority queue.")
                neighbors.insert(item=grounded_action, priority=action_info_gain, selection_probability=selection_prob)

        for action in self.partial_domain.actions:
            self.ig_learner[action].clear_convex_hull_cache()

        return neighbors

    def update_failed_action_neighbors(
            self, neighbors: PriorityQueue, current_state: State, action: ActionCall) -> PriorityQueue:
        """Calculates the new neighbor queue based on the new information of the failed action.

        :param neighbors: the previously calculated neighbors queue.
        :param current_state: the state in which the action had failed.
        :param action: the failed grounded action.
        :return: the new neighbors queue with the failed lifted action updated.
        """
        self.logger.info("Updating the failed action's frontier with the new data.")
        new_neighbors = PriorityQueue()
        failed_action_observed = False
        self.ig_learner[action.name].clear_convex_hull_cache()
        while len(neighbors) > 0:
            neighbor, information_gain, probability = neighbors.get_queue_item_data()
            if neighbor.name != action.name:
                new_neighbors.insert(item=neighbor, priority=information_gain, selection_probability=probability)

            else:
                new_ig = self.calculate_state_action_information_gain(
                    state=current_state, action=neighbor, action_already_calculated=failed_action_observed)
                failed_action_observed = True
                selection_prob = (1 - self._action_observation_rate[neighbor.name] /
                                  sum([rate for rate in self._action_observation_rate.values()]))
                new_neighbors.insert(item=neighbor, priority=new_ig, selection_probability=selection_prob)

        return new_neighbors

    def execute_action(
            self, action_to_execute: ActionCall, previous_state: State, next_state: State) -> None:
        """Executes an action in the environment and updates the action model accordingly.

        :param action_to_execute: the action to execute in the environment.
        :param previous_state: the state prior to the action's execution.
        :param next_state: the state following the action's execution.
        """
        self.logger.info(f"Executing the action {action_to_execute.name} in the environment.")
        self._action_observation_rate[action_to_execute.name] += 1
        observation_objects = self._extract_objects_from_state(next_state)
        self.triplet_snapshot.create_triplet_snapshot(
            previous_state=previous_state, next_state=next_state, current_action=action_to_execute,
            observation_objects=observation_objects)

        pre_state_lifted_numeric_functions = self.function_matcher.match_state_functions(
            action_to_execute, self.triplet_snapshot.previous_state_functions)
        pre_state_lifted_predicates = self.matcher.get_possible_literal_matches(
            action_to_execute, list(self.triplet_snapshot.previous_state_predicates))

        if not self._is_successful_action(previous_state, next_state):
            self.logger.debug("The action was not successful, adding the negative sample to the learner.")
            self.ig_learner[action_to_execute.name].add_negative_sample(
                numeric_negative_sample=pre_state_lifted_numeric_functions,
                negative_propositional_sample=pre_state_lifted_predicates)
            return

        self.logger.debug("The action was successful, adding the positive sample to the learner.")
        self.ig_learner[action_to_execute.name].add_positive_sample(
            positive_numeric_sample=pre_state_lifted_numeric_functions,
            positive_propositional_sample=pre_state_lifted_predicates)
        if action_to_execute.name in self.observed_actions:
            super().update_action(action_to_execute, previous_state, next_state)
            return

        super().add_new_action(action_to_execute, previous_state, next_state)

    def create_safe_model(self) -> LearnerDomain:
        """Creates a safe model from the currently learned data."""
        for action_name, action in self.partial_domain.actions.items():
            if action_name not in self.storage:
                self.logger.debug(f"The action - {action_name} has not been observed in the trajectories!")
                continue

            self.storage[action_name].filter_out_inconsistent_state_variables()
            try:
                self._construct_safe_numeric_preconditions(action)
                self._construct_safe_numeric_effects(action)
                self.logger.info(f"Done learning the action - {action_name}!")

            except NotSafeActionError as e:
                self.logger.warning(f"The action - {e.action_name} is not safe for execution, reason - {e.reason}")

        return self.partial_domain

    def search_to_learn_action_model(
            self, init_state: State, problem_objects: Optional[Dict[str, PDDLObject]] = None) -> Tuple[
        LearnerDomain, int, bool]:
        """Searches for informative actions to learn an action model that solves the problem.

        :param init_state: the current state of the environment.
        :param problem_objects: the objects in the problem - an optional parameter (helps with type hierarchy).
        :return: the learned domain with the number of steps done in the episode and whether the goal was achieved.
        """
        self.logger.info("Searching for informative actions given the current state.")
        observed_objects = problem_objects or self._extract_objects_from_state(init_state)
        grounded_actions = self.create_all_grounded_actions(observed_objects=observed_objects)
        neighbors = self.calculate_valid_neighbors(grounded_actions, init_state)
        current_state = init_state.copy()
        num_steps = 0
        while len(neighbors) > 0:
            action = self._select_next_action_to_execute(neighbors)
            next_state = self.agent.observe(state=current_state, action=action)
            self.execute_action(action_to_execute=action, previous_state=current_state, next_state=next_state)
            num_steps += 1
            while not self._is_successful_action(current_state, next_state) and len(neighbors) > 0:
                self._state_failure_rate += 1
                self.logger.debug("The action was not successful, trying again.")
                neighbors = self.update_failed_action_neighbors(neighbors, current_state, action)
                action = self._select_next_action_to_execute(neighbors)
                next_state = self.agent.observe(state=current_state, action=action)
                self.execute_action(action_to_execute=action, previous_state=current_state, next_state=next_state)
                num_steps += 1
                if num_steps == MAX_STEPS_PER_EPISODE:
                    self.logger.warning("Reached the maximum number of steps per epoch, returning the safe model.")
                    return self.create_safe_model(), num_steps, False

            self.logger.debug("The action changed the state of the environment, updating the possible neighbors.")
            self._state_applicable_actions.clear()
            self._state_failure_rate = 0
            neighbors = self.calculate_valid_neighbors(grounded_actions, next_state)

            current_state = next_state
            if self.agent.get_reward(current_state) == 1:
                return self.create_safe_model(), num_steps, True

        self.logger.info("Reached a state with no neighbors to pull an action from, returning the learned model.")
        return self.create_safe_model(), num_steps, False

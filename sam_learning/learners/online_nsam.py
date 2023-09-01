"""An online version of the Numeric SAM learner."""
from queue import PriorityQueue
from typing import Dict, Set, Optional

from pddl_plus_parser.models import Domain, State, ActionCall, PDDLObject, Precondition, Predicate

from sam_learning.core import InformationGainLearner, NotSafeActionError, LearnerDomain, AbstractAgent
from sam_learning.learners.numeric_sam import PolynomialSAMLearning

LABEL = "label"

NON_INFORMATIVE_IG = 0
MIN_FEATURES_TO_CONSIDER = 1
MAX_STEPS_PER_EPOCH = 100


class OnlineNSAMLearner(PolynomialSAMLearning):
    """"An online version of the Numeric SAM learner."""

    ig_learner: Dict[str, InformationGainLearner]
    agent: AbstractAgent
    _action_observation_rate: Dict[str, float]

    def __init__(self, partial_domain: Domain, polynomial_degree: int = 0, agent: AbstractAgent = None):
        super().__init__(partial_domain=partial_domain, polynomial_degree=polynomial_degree)
        self.ig_learner = {}
        self.agent = agent
        self._action_observation_rate = {action: 1 for action in self.partial_domain.actions}

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
        """Checks whether or not the action was successful.

        :param previous_state: the previous state.
        :param next_state: the next state.
        :return: whether or not the action was successful.
        """
        self.logger.debug("Checking whether or not the action was successful.")
        return self.are_states_different(previous_state, next_state)

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

    def calculate_state_action_information_gain(self, state: State, action: ActionCall) -> float:
        """Calculates the information gain of a state.

        :param state: the state to calculate the information gain of.
        :param action: the action that we calculate the information gain of executing in the state.
        :return: the information gain of the state.
        """
        self.logger.info(f"Calculating the information gain of applying {str(action)} on the state.")
        state_objects = self._extract_objects_from_state(state)
        grounded_state_propositions = self.triplet_snapshot.create_propositional_state_snapshot(
            state, action, state_objects)
        lifted_state_propositions = self.matcher.get_possible_literal_matches(action, list(grounded_state_propositions))
        grounded_state_functions = self.triplet_snapshot.create_numeric_state_snapshot(state, action, state_objects)
        lifted_state_functions = self.function_matcher.match_state_functions(action, grounded_state_functions)
        if self._action_observation_rate[action.name] == 1:
            self.logger.debug(f"Action {action.name} has yet to be observed. Updating the relevant lifted functions.")
            self.ig_learner[action.name].remove_non_existing_functions(list(lifted_state_functions.keys()))

        numeric_ig = self.ig_learner[action.name].calculate_sample_information_gain(
            lifted_state_functions, lifted_state_propositions)
        return numeric_ig

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
            self, grounded_actions: Set[ActionCall], current_state: State) -> PriorityQueue[ActionCall]:
        """Calculates the valid action neighbors for the current state that the learner is in.

        :param grounded_actions: all possible grounded actions.
        :param current_state: the current state that the learner is in.
        :return: a priority queue of the valid neighbors for the current state, the priority of the action is based
            on their IG.
        """
        self.logger.info("Calculating the valid neighbors for the current state.")
        neighbors = PriorityQueue()
        for grounded_action in grounded_actions:
            self.logger.debug(f"Checking the action {grounded_action.name}.")
            # Setting to a negative value since priority queue is works from smallest to largest.
            action_info_gain = -self.calculate_state_action_information_gain(
                state=current_state, action=grounded_action)
            action_info_gain *= 1 / self._action_observation_rate[grounded_action.name]

            if abs(action_info_gain) > NON_INFORMATIVE_IG:  # IG is a negative number.
                self.logger.info(f"The action {grounded_action.name} is informative, adding it to the priority queue.")
                neighbors.put((action_info_gain, str(grounded_action), grounded_action))

        return neighbors

    def execute_action(
            self, action_to_execute: ActionCall, previous_state: State, next_state: State) -> None:
        """Executes an action in the environment and updates the action model accordingly.

        :param action_to_execute: the action to execute in the environment.
        :param previous_state: the state prior to the action's execution.
        :param next_state: the state following the action's execution.
        """
        self.logger.info(f"Executing the action {action_to_execute.name} in the environment.")
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
            self._action_observation_rate[action_to_execute.name] += 0.1
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
            self._action_observation_rate[action_to_execute.name] += 0.5
            return

        super().add_new_action(action_to_execute, previous_state, next_state)
        self._action_observation_rate[action_to_execute.name] += 1

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

    def search_for_informative_actions(
            self, init_state: State, problem_objects: Optional[Dict[str, PDDLObject]] = None) -> LearnerDomain:
        """Searches for informative actions given the current state.

        :param init_state: the current state of the environment.
        :param problem_objects: the objects in the problem - an optional parameter (helps with type hierarchy).
        :return: the set of informative actions.
        """
        self.logger.info("Searching for informative actions given the current state.")
        observed_objects = problem_objects or self._extract_objects_from_state(init_state)
        grounded_actions = self.create_all_grounded_actions(observed_objects=observed_objects)
        neighbors = self.calculate_valid_neighbors(grounded_actions, init_state)
        current_state = init_state.copy()
        num_steps = 0
        while not neighbors.empty():
            _, _, action = neighbors.get()
            next_state = self.agent.observe(state=current_state, action=action)
            self.execute_action(action_to_execute=action, previous_state=current_state, next_state=next_state)
            num_steps += 1
            while not self._is_successful_action(current_state, next_state):
                self.logger.debug("The action was not successful, trying again.")
                _, _, action = neighbors.get()
                next_state = self.agent.observe(state=current_state, action=action)
                self.execute_action(action_to_execute=action, previous_state=current_state, next_state=next_state)
                num_steps += 1
                if num_steps == MAX_STEPS_PER_EPOCH:
                    self.logger.warning("Reached the maximum number of steps per epoch, returning the safe model.")
                    return self.create_safe_model()

            self.logger.debug("The action changed the state of the environment, updating the possible neighbors.")
            neighbors = self.calculate_valid_neighbors(grounded_actions, next_state)

            current_state = next_state
            if self.agent.get_reward(current_state) == 1 or num_steps == MAX_STEPS_PER_EPOCH:
                return self.create_safe_model()

        self.logger.info("Reached a state with no neighbors to pull an action from, returning the learned model.")
        return self.create_safe_model()

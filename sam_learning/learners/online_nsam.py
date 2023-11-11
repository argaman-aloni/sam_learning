"""An online version of the Numeric SAM learner."""
from typing import Dict, Set, Optional, Tuple, List, Any

from pddl_plus_parser.models import Domain, State, ActionCall, PDDLObject, Precondition, Predicate, PDDLFunction, \
    Action, Operator

from sam_learning.core import InformationGainLearner, LearnerDomain, AbstractAgent, \
    PriorityQueue, LearnerAction
from sam_learning.learners.numeric_sam import PolynomialSAMLearning

LABEL = "label"

NON_INFORMATIVE_IG = 0
MIN_FEATURES_TO_CONSIDER = 1
MAX_STEPS_PER_EPISODE = 500

EXECUTION_DB_COLUMNS = ["lifted_action", "lifted_state_predicates", "lifted_function_values", "execution_result"]
SUCCESS_RESULT = 1
FAIL_RESULT = -1


class OnlineNSAMLearner(PolynomialSAMLearning):
    """"An online version of the Numeric SAM learner."""

    ig_learner: Dict[str, InformationGainLearner]
    agent: AbstractAgent
    _action_observation_rate: Dict[str, float]
    _action_failure_rate: Dict[str, int]
    _state_applicable_actions: PriorityQueue
    _state_action_execution_db: Dict[str, List[Any]]
    _unsafe_domain: Domain

    def __init__(self, partial_domain: Domain, polynomial_degree: int = 0, agent: AbstractAgent = None,
                 fluents_map: Optional[Dict[str, List[str]]] = None):
        super().__init__(partial_domain=partial_domain, polynomial_degree=polynomial_degree,
                         preconditions_fluent_map=fluents_map)
        self.ig_learner = {}
        self.agent = agent
        self._action_observation_rate = {action: 1 for action in self.partial_domain.actions}
        self._action_failure_rate = {action: 0 for action in self.partial_domain.actions}
        self._state_applicable_actions = PriorityQueue()
        self._state_action_execution_db = {col: [] for col in EXECUTION_DB_COLUMNS}
        self._unsafe_domain = partial_domain.shallow_copy()
        for action_name, action_data in self.partial_domain.actions.items():
            self.ig_learner[action_name] = InformationGainLearner(action_name=action_name)

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

        state_objects.update(self.partial_domain.constants)
        self.logger.debug(f"Extracted the following objects - {list(state_objects.keys())}")
        return state_objects

    def _add_action_execution_to_db(self, lifted_action: str, lifted_predicates: List[Predicate],
                                    lifted_numeric_functions: dict[str, PDDLFunction], execution_result: int) -> None:
        """Adds the trace of the execution of the action including its outcome.

        :param lifted_action:  the lifted action that was executed.
        :param lifted_predicates: the lifted bounded predicates of the state matching the lifted actions.
        :param lifted_numeric_functions: the lifted numeric functions that were observed in the state.
        :param execution_result: the result of the execution of the action - success (1) or fail (-1).
        """
        self.logger.info("Adding the action execution to the database.")
        lifted_predicates_str = [pred.untyped_representation for pred in lifted_predicates]
        lifted_numeric_functions = [func for func in lifted_numeric_functions.values()]
        self._state_action_execution_db["lifted_action"].append(lifted_action)
        self._state_action_execution_db["lifted_state_predicates"].append(lifted_predicates_str)
        self._state_action_execution_db["lifted_function_values"].append(lifted_numeric_functions)
        self._state_action_execution_db["execution_result"].append(execution_result)

    def calculate_novelty_rate(self, action: ActionCall, state: State) -> float:
        """Checks if the lifted bounded action was already executed in the state.

        :param action: the action to check whether visited in the state.
        :param state: the current state of the environment.
        :return: whether the action was already executed in the state.
        """
        self.logger.info("Trying to calculate the novelty rate of the action.")
        if len(self.partial_domain.actions[action.name].numeric_effects) == 0 and \
                len(self.partial_domain.actions[action.name].discrete_effects) == 0:
            self.logger.debug("The action was not observed and its effects were not yet learned.")
            return 100

        self.logger.debug("The action was partially learned. Calculating the novelty rate of the action.")
        # starting to calculate the novelty rate.
        operator = Operator(action=self._unsafe_domain.actions[action.name], domain=self._unsafe_domain,
                            grounded_action_call=action.parameters)
        assumed_next_state = operator.apply(state, allow_inapplicable_actions=True)
        assumed_lifted_functions, assumed_lifted_predicates = self._get_lifted_bounded_state(action, assumed_next_state)
        assumed_next_state_predicates_str = {pred.untyped_representation for pred in assumed_lifted_predicates}
        discrete_novelty_rate = sum([len(assumed_next_state_predicates_str.difference(pred_set))
                                     for pred_set in self._state_action_execution_db["lifted_state_predicates"]])
        numeric_novelty_rate = 0
        for observed_numeric_state in self._state_action_execution_db["lifted_function_values"]:
            for observed_numeric_function in observed_numeric_state:
                numeric_novelty_rate += abs(
                    observed_numeric_function.value - assumed_lifted_functions[
                        observed_numeric_function.untyped_representation].value)

        novelty_rate = ((discrete_novelty_rate + numeric_novelty_rate) /
                        len(self._state_action_execution_db["lifted_action"]))

        return novelty_rate

    def _apply_feature_selection(self, action: ActionCall) -> List[str]:
        """Applies feature selection to the action's numeric features.

        Note:
            This approach is used to reduce the calculation efforts of the learner. The idea is to select features in a
            greedy manner, starting from the minimal number of features, i.e. zero and increasing the number over time.
            There might be a more informative way to apply feature selection, but this is for future work.


        :param action: the action being applied.
        :return: the features to explore.
        """
        return self.preconditions_fluent_map[action.name] if \
            self.preconditions_fluent_map else self.ig_learner[action.name].numeric_positive_samples.columns.tolist()

    def _get_lifted_bounded_state(
            self, action: ActionCall, state: State) -> Tuple[Dict[str, PDDLFunction], List[Predicate]]:
        """Gets the lifted bounded predicates and functions that match the action being executed.

        :param action: the action being executed.
        :param state: the state in which the action is being executed.
        :return: the lifted bounded functions and predicates.
        """
        state_objects = self._extract_objects_from_state(state)
        grounded_state_propositions = self.triplet_snapshot.create_propositional_state_snapshot(
            state, action, state_objects)
        lifted_predicates = self.matcher.get_possible_literal_matches(action, list(grounded_state_propositions))
        grounded_state_functions = self.triplet_snapshot.create_numeric_state_snapshot(state, action, state_objects)
        lifted_functions = self.function_matcher.match_state_functions(action, grounded_state_functions)
        return lifted_functions, lifted_predicates

    def _calculate_selection_probability(self, grounded_action: ActionCall) -> float:
        """Calculates the selection probability of the action.

        :param grounded_action: the action to calculate the selection probability of.
        :return: the selection probability of the action.
        """
        normalized_observation_rate = self._action_observation_rate[grounded_action.name] / \
                                      sum([rate for rate in self._action_observation_rate.values()])
        # we prefer actions that failed more than others since they might be more dependent on others to succeed.
        failure_rate = self._action_failure_rate[grounded_action.name] / \
                       (sum([rate for rate in self._action_failure_rate.values()]) + 1)
        selection_prob = (1 - normalized_observation_rate) + failure_rate
        return selection_prob

    def _update_unsafe_model(self, action_name: str) -> None:
        """

        :param action_name:
        :return:
        """
        self.logger.info(f"Updating the unsafe model after observing a successful execution of {action_name}.")
        action = self._create_safe_action(action_name)
        updated_action = Action()
        updated_action.name = action.name
        updated_action.signature = action.signature
        updated_action.preconditions = action.preconditions
        updated_action.discrete_effects = {effect.copy() for effect in action.discrete_effects}
        updated_action.numeric_effects = {effect for effect in action.numeric_effects}
        self._unsafe_domain.actions[action.name] = updated_action

    def _reset_action_numeric_data(self, action_name: str) -> None:
        """

        :param action_name:
        :return:
        """
        self.logger.debug("Resetting the numeric part of the action's data.")
        action = self.partial_domain.actions[action_name]
        discrete_preconditions = {op for op in action.preconditions.root.operands if isinstance(op, Predicate)}
        action.preconditions.root = Precondition("and")
        for discrete_precondition in discrete_preconditions:
            action.preconditions.add_condition(discrete_precondition)

    def _create_safe_action_model(self) -> Tuple[Dict[str, LearnerAction], Dict[str, str]]:
        """Overriding the method to create the safe action model - includes clearing the temporary action information.

        :return: the actions that are allowed to execute and the metadata about the learning.
        """
        for action in self.partial_domain.actions:
            self._reset_action_numeric_data(action)

        return super()._create_safe_action_model()

    def create_all_grounded_actions(self, observed_objects: Dict[str, PDDLObject]) -> Set[ActionCall]:
        """Creates all the grounded actions for the domain given the current possible objects.

        :param observed_objects: the objects that the learner has observed so far.
        :return: a set of all the possible grounded actions.
        """
        self.logger.info("Creating all the grounded actions for the domain given the current possible objects.")
        self._action_failure_rate = {action: 0 for action in self.partial_domain.actions}
        grounded_action_calls = self.vocabulary_creator.create_grounded_actions_vocabulary(
            domain=self.partial_domain, observed_objects=observed_objects)
        return grounded_action_calls

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
        lifted_functions, lifted_predicates = self._get_lifted_bounded_state(action, state)
        if not self.ig_learner[action.name].are_dataframes_initialized():
            self.logger.debug(f"Action {action.name} has yet to be observed. Updating the relevant lifted functions.")
            self.ig_learner[action.name].init_dataframes(
                valid_lifted_functions=list([func for func in lifted_functions.keys()]),
                lifted_predicates=[pred.untyped_representation for pred in lifted_predicates])

        # TODO: Remove this code to the feature selection values.
        features_to_explore = self._apply_feature_selection(action)
        is_informative = self.ig_learner[action.name].is_sample_informative(
            lifted_functions, lifted_predicates, use_cache=action_already_calculated,
            relevant_numeric_features=features_to_explore)
        if not is_informative:
            return NON_INFORMATIVE_IG

        return self.calculate_novelty_rate(action, state)

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
            action_info_gain = self.calculate_state_action_information_gain(
                state=current_state, action=grounded_action,
                action_already_calculated=action_calculation_cache[grounded_action.name] > 0)
            action_calculation_cache[grounded_action.name] += 1
            selection_prob = self._calculate_selection_probability(grounded_action)
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
                continue

            new_ig = self.calculate_state_action_information_gain(
                state=current_state, action=neighbor, action_already_calculated=failed_action_observed)
            failed_action_observed = True
            selection_prob = self._calculate_selection_probability(neighbor)
            new_neighbors.insert(item=neighbor, priority=new_ig, selection_probability=selection_prob)

        return new_neighbors

    def execute_action(
            self, action_to_execute: ActionCall, previous_state: State, next_state: State, reward: int) -> None:
        """Executes an action in the environment and updates the action model accordingly.

        :param action_to_execute: the action to execute in the environment.
        :param previous_state: the state prior to the action's execution.
        :param next_state: the state following the action's execution.
        :param reward: the reward for executing the action.
        """
        self.logger.info(f"Executing the action {action_to_execute.name} in the environment.")
        self._action_observation_rate[action_to_execute.name] += 1
        observation_objects = self._extract_objects_from_state(next_state)
        self.triplet_snapshot.create_triplet_snapshot(
            previous_state=previous_state, next_state=next_state, current_action=action_to_execute,
            observation_objects=observation_objects)

        pre_state_functions, pre_state_predicates = self._get_lifted_bounded_state(action_to_execute, previous_state)
        action_signature = str(self.partial_domain.actions[action_to_execute.name])
        if reward < 0:
            self.logger.debug("The action was not successful, adding the negative sample to the learner.")
            self._action_failure_rate[action_to_execute.name] += 1
            self._add_action_execution_to_db(action_signature, pre_state_predicates, pre_state_functions, FAIL_RESULT)
            self.ig_learner[action_to_execute.name].add_negative_sample(
                numeric_negative_sample=pre_state_functions, negative_propositional_sample=pre_state_predicates)
            return

        self._reset_action_numeric_data(action_to_execute.name)
        self.logger.debug("The action was successful, adding the positive sample to the learner.")
        self._add_action_execution_to_db(action_signature, pre_state_predicates, pre_state_functions, SUCCESS_RESULT)
        self.ig_learner[action_to_execute.name].add_positive_sample(
            positive_numeric_sample=pre_state_functions, positive_propositional_sample=pre_state_predicates)
        if action_to_execute.name in self.observed_actions:
            super().update_action(action_to_execute, previous_state, next_state)
            return

        super().add_new_action(action_to_execute, previous_state, next_state)
        self._update_unsafe_model(action_to_execute.name)

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
        while len(neighbors) > 0 and num_steps < MAX_STEPS_PER_EPISODE:
            action = neighbors.get_item()
            next_state, reward = self.agent.observe(state=current_state, action=action)
            self.execute_action(action, current_state, next_state, reward=reward)
            num_steps += 1
            while reward < 0 < len(neighbors) and num_steps < MAX_STEPS_PER_EPISODE:
                self.logger.debug("The action was not successful, trying again.")
                neighbors = self.update_failed_action_neighbors(neighbors, current_state, action)
                if len(neighbors) == 0:
                    break

                action = neighbors.get_item()
                next_state, reward = self.agent.observe(state=current_state, action=action)
                self.execute_action(action, current_state, next_state, reward=reward)
                num_steps += 1

            if num_steps >= MAX_STEPS_PER_EPISODE or len(neighbors) == 0:
                break

            self.logger.debug("The action changed the state of the environment, updating the possible neighbors.")
            self._state_applicable_actions.clear()
            neighbors = self.calculate_valid_neighbors(grounded_actions, next_state)
            current_state = next_state
            if self.agent.goal_reached(current_state):
                self.logger.info("The goal has been reached, returning the learned model.")
                self._create_safe_action_model()
                return self.partial_domain, num_steps, True

        self.logger.info("Reached a state with no neighbors to pull an action from, returning the learned model.")
        self._create_safe_action_model()
        return self.partial_domain, num_steps, False

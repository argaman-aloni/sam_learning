"""An online version of the Numeric SAM learner."""
import random
from typing import Dict, Set, Optional, Tuple, List, Any

from pddl_plus_parser.models import Domain, State, ActionCall, Precondition, Predicate, PDDLFunction, \
    Action, NumericalExpressionTree, CompoundPrecondition

from sam_learning.core import InformationGainLearner, LearnerDomain, AbstractAgent, \
    PriorityQueue, LearnerAction, EpisodeInfoRecord
from sam_learning.core.online_learning.novelty_learner import NoveltyLearner, FAIL_RESULT, SUCCESS_RESULT
from sam_learning.learners.numeric_sam import NumericSAMLearner

NON_INFORMATIVE_IG = 0
MAX_STEPS_PER_EPISODE = 100


class OnlineNSAMLearner(NumericSAMLearner):
    """"An online version of the Numeric SAM learner."""

    ig_learner: Dict[str, InformationGainLearner]
    agent: AbstractAgent
    episode_statistics: Dict[str, int]
    _action_observation_rate: Dict[str, float]
    _action_failure_rate: Dict[str, int]
    _state_action_execution_db: Dict[str, List[Any]]
    _unsafe_domain: Domain
    _episode_recorder: EpisodeInfoRecord
    safe_perfect_numeric_effects = Dict[str, Set[NumericalExpressionTree]]

    def __init__(self, partial_domain: Domain, polynomial_degree: int = 0, agent: AbstractAgent = None,
                 episode_recorder: EpisodeInfoRecord = None,
                 fluents_map: Optional[Dict[str, List[str]]] = None):
        super().__init__(partial_domain=partial_domain, polynomial_degree=polynomial_degree,
                         relevant_fluents=fluents_map)
        self.ig_learner = {}
        self.agent = agent
        self.applicable_actions = PriorityQueue()
        self._episode_recorder = episode_recorder
        self._action_observation_rate = {action: 1 for action in self.partial_domain.actions}
        self._action_failure_rate = {action: 0 for action in self.partial_domain.actions}
        self.novelty_calculator = NoveltyLearner()
        self._unsafe_domain = partial_domain.shallow_copy()
        self.safe_perfect_numeric_effects = {}

        for action_name, action_data in self.partial_domain.actions.items():
            self.ig_learner[action_name] = InformationGainLearner(action_name=action_name)

    def _apply_feature_selection(self, action: ActionCall) -> List[str]:
        """Applies feature selection to the action's numeric features.

        Note:
            This approach is used to reduce the calculation efforts of the learner. The idea is to select features in a
            greedy manner, starting from the minimal number of features, i.e. zero and increasing the number over time.
            There might be a more informative way to apply feature selection, but this is for future work.


        :param action: the action being applied.
        :return: the features to explore.
        """
        return self.relevant_fluents[action.name] if \
            self.relevant_fluents else self.ig_learner[action.name].numeric_positive_samples.columns.tolist()

    def _get_lifted_bounded_state(
            self, action: ActionCall, state: State) -> Tuple[Dict[str, PDDLFunction], List[Predicate]]:
        """Gets the lifted bounded predicates and functions that match the action being executed.

        :param action: the action being executed.
        :param state: the state in which the action is being executed.
        :return: the lifted bounded functions and predicates.
        """
        state_objects = state.get_state_objects()
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

    def _construct_safe_numeric_effects(self, action: LearnerAction, allow_unsafe_learning: bool = False) -> bool:
        """Constructs the safe numeric effects for the input action.

        :param action: the action that its effects are constructed for.
        :param allow_unsafe_learning: whether to allow unsafe learning of the effects.
        """
        if action.name in self.safe_perfect_numeric_effects and allow_unsafe_learning:
            self.logger.debug(f"The effects of action - {action.name} were already learned perfectly, "
                              f"and can use it for internal calculations.")
            action.numeric_effects = self.safe_perfect_numeric_effects[action.name]
            return True

        learned_perfectly = super()._construct_safe_numeric_effects(action, allow_unsafe_learning)
        if learned_perfectly:
            self.safe_perfect_numeric_effects[action.name] = action.numeric_effects

    def _update_unsafe_model(self, action_name: str) -> None:
        """Updates the unsafe model after observing a successful execution of the action.

        :param action_name: the action that had been successfully executed.
        """
        self.logger.info(f"Updating the unsafe model after observing a successful execution of {action_name}.")
        action = self.partial_domain.actions[action_name]
        if action_name in self.safe_perfect_numeric_effects:
            numeric_effects = self.safe_perfect_numeric_effects[action_name]

        else:
            self._construct_safe_numeric_effects(action, allow_unsafe_learning=True)
            numeric_effects = {effect for effect in action.numeric_effects}

        updated_action = Action()
        updated_action.name = action.name
        updated_action.signature = action.signature
        updated_action.preconditions = CompoundPrecondition()
        updated_action.discrete_effects = {effect.copy() for effect in action.discrete_effects}
        updated_action.numeric_effects = numeric_effects
        self._unsafe_domain.actions[action.name] = updated_action

    def _reset_action_numeric_data(self, action_name: str) -> None:
        """Resets the numeric part of the action's data.

        :param action_name: the name of the action to reset.
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

    def _handle_non_informative_action(self, action: ActionCall, state: State) -> None:
        """Handles the insertion of non-informative actions to the applicable actions queue.

        :param action: the action that is not informative and possibly applicable.
        :param state: the state in which the action is being evaluated.
        """
        self.logger.info(f"Handling the non-informative applicable action {action.name}.")
        applicable_action_novelty = self.novelty_calculator.calculate_novelty(
            action, state, self._unsafe_domain, self.observed_actions)
        self.applicable_actions.insert(
            item=action, priority=applicable_action_novelty,
            selection_probability=self._calculate_selection_probability(action))

    def _select_action_to_execute(self, frontier: PriorityQueue) -> ActionCall:
        """Selects the next action to execute from both the frontier and the applicable actions.

        :param frontier: the informative actions that are available to execute.
        :return: the next action to execute which is the action with the largest novelty chosen from one of the queues.
        """
        self.logger.info("Selecting the next action to execute...")
        if len(frontier) == 0 or len(self.applicable_actions) == 0:
            queue_to_select_from = frontier if len(frontier) > 0 else self.applicable_actions
            return queue_to_select_from.get_item()

        queue_to_select_from = random.choices([self.applicable_actions, frontier], weights=[0.2, 0.8], k=1)[0]
        return queue_to_select_from.get_item()

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
        all_possible_predicates = self.vocabulary_creator.create_lifted_vocabulary(
            domain=self.partial_domain, possible_parameters=self.partial_domain.actions[action.name].signature)
        if not self.ig_learner[action.name].are_dataframes_initialized():
            self.logger.debug(f"Action {action.name} has yet to be observed. Updating the relevant lifted functions.")
            self.ig_learner[action.name].init_dataframes(
                valid_lifted_functions=[func for func in lifted_functions.keys()],
                lifted_predicates=[pred.untyped_representation for pred in all_possible_predicates])

        features_to_explore = self._apply_feature_selection(action)
        is_informative, is_applicable = self.ig_learner[action.name].is_sample_informative(
            lifted_functions, lifted_predicates, use_cache=action_already_calculated,
            relevant_numeric_features=features_to_explore)
        if not is_informative:
            self.logger.debug(f"The action {action.name} is not informative checking if it is applicable in the state.")
            if is_applicable:
                self.logger.debug(f"The action {action.name} is applicable, adding to the applicable actions queue.")
                self._handle_non_informative_action(action, state)

            return NON_INFORMATIVE_IG

        return self.novelty_calculator.calculate_novelty(action, state, self._unsafe_domain, self.observed_actions)

    def update_agent(self, new_agent: AbstractAgent) -> None:
        """Updates the agent that the learner is using."""
        self.logger.info(f"Updating the agent.")
        self._action_failure_rate = {action: 0 for action in self.partial_domain.actions}
        self.agent = new_agent

    def calculate_valid_neighbors(
            self, grounded_actions: Set[ActionCall], current_state: State, step_number: int) -> PriorityQueue:
        """Calculates the valid action neighbors for the current state that the learner is in.

        :param grounded_actions: all possible grounded actions.
        :param current_state: the current state that the learner is in.
        :return: a priority queue of the valid neighbors for the current state, the priority of the action is based
            on their IG.
        """
        self.logger.info("Calculating the valid neighbors for the current state.")
        neighbors = PriorityQueue()

        action_calculation_cache = {action_name: 0 for action_name in self.partial_domain.actions}
        lifted_to_grounded = {action_name: [action for action in grounded_actions if action.name == action_name]
                              for action_name in self.partial_domain.actions}
        actions_to_explore = []
        for action, groundings in lifted_to_grounded.items():
            actions_to_explore.extend(random.sample(groundings, k=min(len(groundings), 2)))

        for grounded_action in actions_to_explore:
            self.logger.debug(f"Checking the action {grounded_action.name}.")
            if step_number % 10 == 0:   # TODO: Delete this abomination.
                action_info_gain = self.calculate_state_action_information_gain(
                    state=current_state, action=grounded_action,
                    action_already_calculated=action_calculation_cache[grounded_action.name] > 0)
            else:
                action_info_gain = 1

            action_calculation_cache[grounded_action.name] += 1
            if abs(action_info_gain) > NON_INFORMATIVE_IG:  # IG is a negative number.
                selection_prob = self._calculate_selection_probability(grounded_action)
                self.logger.info(f"The action {grounded_action.name} is informative, adding it to the priority queue.")
                neighbors.insert(item=grounded_action, priority=action_info_gain, selection_probability=selection_prob)

        for action in self.partial_domain.actions:
            self.ig_learner[action].clear_convex_hull_cache()

        self._episode_recorder.add_num_informative_actions_in_step(len(neighbors))
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
            if abs(new_ig) > NON_INFORMATIVE_IG:
                self.logger.info(f"The action {neighbor.name} is informative, adding it to the priority queue.")
                new_neighbors.insert(item=neighbor, priority=new_ig, selection_probability=selection_prob)

        self._episode_recorder.add_num_informative_actions_in_step(len(new_neighbors))
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
        self.triplet_snapshot.create_triplet_snapshot(
            previous_state=previous_state, next_state=next_state, current_action=action_to_execute,
            observation_objects=next_state.get_state_objects())

        pre_state_functions, pre_state_predicates = self._get_lifted_bounded_state(action_to_execute, previous_state)
        if reward < 0:
            self.logger.debug("The action was not successful, adding the negative sample to the learner.")
            self._action_failure_rate[action_to_execute.name] += 1
            self.novelty_calculator.add_sample_to_execution_db(
                action_to_execute.name, pre_state_functions, pre_state_predicates, FAIL_RESULT)
            self._episode_recorder.add_step_data(action_to_execute.name, FAIL_RESULT)
            self.ig_learner[action_to_execute.name].add_negative_sample(
                numeric_negative_sample=pre_state_functions, negative_propositional_sample=pre_state_predicates)
            return

        self._reset_action_numeric_data(action_to_execute.name)
        self.logger.debug("The action was successful, adding the positive sample to the learner.")
        self.novelty_calculator.add_sample_to_execution_db(
            action_to_execute.name, pre_state_functions, pre_state_predicates, SUCCESS_RESULT)
        self._episode_recorder.add_step_data(action_to_execute.name, SUCCESS_RESULT)
        self.ig_learner[action_to_execute.name].add_positive_sample(
            positive_numeric_sample=pre_state_functions, positive_propositional_sample=pre_state_predicates)
        if action_to_execute.name in self.observed_actions:
            super().update_action(action_to_execute, previous_state, next_state)
            self._update_unsafe_model(action_to_execute.name)
            return

        super().add_new_action(action_to_execute, previous_state, next_state)
        self._update_unsafe_model(action_to_execute.name)

    def search_to_learn_action_model(self, init_state: State) -> Tuple[LearnerDomain, int, bool]:
        """Searches for informative actions to learn an action model that solves the problem.

        :param init_state: the current state of the environment.
        :return: the learned domain with the number of steps done in the episode and whether the goal was achieved.
        """
        self.logger.info("Searching for informative actions given the current state.")
        grounded_actions = self.agent.get_environment_actions(init_state)
        self._episode_recorder.add_num_grounded_actions(len(grounded_actions))
        step_number = 0
        frontier = self.calculate_valid_neighbors(grounded_actions, init_state, step_number)
        current_state = init_state.copy()
        while len(frontier) + len(self.applicable_actions) > 0 and step_number < MAX_STEPS_PER_EPISODE:
            action = self._select_action_to_execute(frontier)
            next_state, reward = self.agent.observe(state=current_state, action=action)
            self.execute_action(action, current_state, next_state, reward=reward)
            step_number += 1
            while reward < 0 < len(frontier) + len(self.applicable_actions) and step_number < MAX_STEPS_PER_EPISODE:
                self.logger.debug("The action was not successful, trying again.")
                frontier = self.update_failed_action_neighbors(frontier, current_state, action)
                action = self._select_action_to_execute(frontier)
                next_state, reward = self.agent.observe(state=current_state, action=action)
                self.execute_action(action, current_state, next_state, reward=reward)
                step_number += 1

            grounded_actions = self.agent.get_environment_actions(next_state)
            self.applicable_actions.clear()
            frontier = self.calculate_valid_neighbors(grounded_actions, next_state, step_number)
            current_state = next_state
            if self.agent.goal_reached(current_state):
                self.logger.info("The goal has been reached, returning the learned model.")
                self._create_safe_action_model()
                return self.partial_domain, step_number, True

        self.logger.info("Reached a state with no neighbors to pull an action from, returning the learned model.")
        self._create_safe_action_model()
        return self.partial_domain, step_number, False
